#ifndef ALLOCATOR_SOA_ALLOCATOR_H
#define ALLOCATOR_SOA_ALLOCATOR_H

#include <assert.h>
#include <tuple>
#include <typeinfo>
#include <type_traits>

#define __DEV__ __device__

#include "bitmap/bitmap.h"

#include "allocator/soa_base.h"
#include "allocator/soa_block.h"
#include "allocator/soa_defrag_record.h"
#include "allocator/soa_executor.h"
#include "allocator/soa_field.h"
#include "allocator/tuple_helper.h"
#include "allocator/util.h"


// TODO: Fix visibility.
template<BlockIndexT N_Objects, class... Types>
class SoaAllocator {
 public:
  using ThisAllocator = SoaAllocator<N_Objects, Types...>;

  static const ObjectIndexT kNumBlockElements = 64;
  static const uint64_t kBlockAddrBitmask = 0xFFFFFFFFFFC0;
  static_assert(kNumBlockElements == 64,
                "Not implemented: Block size != 64.");
  static const BlockIndexT N = N_Objects / kNumBlockElements;

  static_assert(N_Objects % kNumBlockElements == 0,
                "N_Objects Must be divisible by BlockSize.");


#ifdef OPTION_DEFRAG
  // ---- Defragmentation (soa_defrag.inc) ----
  template<typename T, int NumRecords>
  __DEV__ void defrag_choose_source_block(int min_remaining_records);

  template<typename T, int NumRecords>
  __DEV__ void defrag_choose_target_blocks();

  template<typename T, int NumRecords>
  __DEV__ void defrag_move();

  template<typename T, int NumRecords>
  __DEV__ void defrag_store_forwarding_ptr();

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  template<typename T, int NumRecords>
  __DEV__ void defrag_update_block_state();
#endif  // OPTION_DEFRAG_FORWARDING_POINTER

  template<int NumRecords>
  __DEV__ void load_records_to_shared_mem();

  void DBG_print_defrag_time();

  // Should be invoked from host side.
  template<typename T, int NumRecords>
  void parallel_defrag(int min_num_compactions);

  template<typename T>
  void parallel_defrag(int min_num_compactions);
  // ---- END ----
#endif  // OPTION_DEFRAG

  // ---- Debugging (soa_debug.inc) ----
  template<class T>
  __DEV__ BlockIndexT DBG_allocated_slots();

  template<class T>
  __DEV__ BlockIndexT DBG_used_slots();

  static void DBG_print_stats();

  __DEV__ float DBG_calculate_fragmentation();

  __DEV__ void DBG_print_state_stats();

  __DEV__ void DBG_collect_stats();

  __DEV__ void DBG_print_collected_stats();

  // TODO: Consider moving out of soa_debug.inc.
  template<class T>
  __DEV__ static bool is_type(const T* ptr);

  long unsigned int DBG_get_enumeration_time() {
    // Microseconds to milliseconds.
    return bench_prefix_sum_time/1000;
  }
  // ---- END ----


  template<typename T>
  struct BlockHelper {
    static const int kIndex = TYPE_INDEX(Types..., T);

    static const int kSize =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::k64BlockMinSize>::kSize;

    // Data segment size. TODO: Rename.
    static const int kBytes =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::k64BlockMinSize>::kBytes;

    using BlockType = SoaBlock<T, kIndex, kSize>;

#ifdef OPTION_DEFRAG
    static const int kLeq50Threshold = BlockType::kLeq50Threshold;
#endif  // OPTION_DEFRAG
  };

  using BlockBitmapT = unsigned long long int;

  template<typename T>
  struct TypeId {
    static const TypeIndexT value = BlockHelper<T>::kIndex;
  };

  __DEV__ static TypeIndexT get_type(const void* ptr) {
    auto type_id = PointerHelper::get_type(ptr);
    assert(type_id < kNumTypes);
    return type_id;
  }

  __DEV__ void initialize(char* data_buffer) {
    global_free_.initialize(true);
    for (int i = 0; i < kNumTypes; ++i) {
      allocated_[i].initialize(false);
      active_[i].initialize(false);
#ifdef OPTION_DEFRAG
      leq_50_[i].initialize(false);
#endif  // OPTION_DEFRAG

      if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef OPTION_DEFRAG
        num_leq_50_[i] = 0;
#endif  // OPTION_DEFRAG
      }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      data_ = data_buffer;

      // Check alignment of data storage buffer.
      assert(reinterpret_cast<uintptr_t>(data_) % 64 == 0);
    }
  }

  __DEV__ SoaAllocator(const ThisAllocator&) = delete;

  // Use initialize() instead of constructor to avoid zero-initialization.
  __DEV__ SoaAllocator() = delete;

  // Allocate location and return pointer.
  template<class T>
  __DEV__ T* allocate_new() {
    T* result = nullptr;

    do {
      const auto active = __activemask();
      // Leader thread is the first thread whose mask bit is set to 1.
      const auto leader = __ffs(active) - 1;
      assert(leader >= 0 && leader < 32);
      // Use lane mask to empty all bits higher than the current thread.
      // The rank of this thread is the number of bits set to 1 in the result.
      const auto rank = __lane_id();
      assert(rank < 32);

      // Values to be calculated by the leader.
      BlockIndexT block_idx;
      BlockBitmapT allocation_bitmap;
      if (rank == leader) {
        assert(__popc(__activemask()) == 1);    // Only one thread executing.

        block_idx = find_active_block<T>();
        auto* block = get_block<T>(block_idx);
        BlockBitmapT* free_bitmap = &block->free_bitmap;
        allocation_bitmap = allocate_in_block<T>(
            free_bitmap, __popc(active), block_idx);
        auto num_allocated = __popcll(allocation_bitmap);

        auto actual_type_id = block->type_id;
        if (actual_type_id != BlockHelper<T>::kIndex) {
          // TODO: Check correctness. This code is rarely executed.

          // Block deallocated and initialized for a new type between lookup
          // from active bitmap and here. This is extremely unlikely!
          // But possible.
          // Undo allocation and update bitmaps accordingly.

          // Note: Cannot be in invalidation here, because we are certain that
          // we allocated the bits that we are about to deallocate here.
          auto before_undo = atomicOr(free_bitmap, allocation_bitmap);
          auto slots_before_undo = __popcll(before_undo);

#ifdef OPTION_DEFRAG
          if (BlockHelper<T>::kSize - slots_before_undo
                  > BlockHelper<T>::kLeq50Threshold
              && BlockHelper<T>::kSize - slots_before_undo - num_allocated
                  <= BlockHelper<T>::kLeq50Threshold) {
            ASSERT_SUCCESS(leq_50_[actual_type_id].allocate<true>(block_idx));
            atomicAdd(&num_leq_50_[actual_type_id], 1);
          }
#endif  // OPTION_DEFRAG

          // Cases to handle: block now active again or block empty now.
          if (slots_before_undo == 0) {
            // Block became active. (Was full.)
            ASSERT_SUCCESS(active_[actual_type_id].allocate<true>(block_idx));
          } else if (slots_before_undo == BlockHelper<T>::kSize - 1) {
            // Block now empty.
            if (invalidate_block<T>(block_idx)) {
              // Block is invalidated and no new allocations can be performed.
              ASSERT_SUCCESS(active_[actual_type_id].deallocate<true>(block_idx));
#ifdef OPTION_DEFRAG
              ASSERT_SUCCESS(leq_50_[actual_type_id].deallocate<true>(block_idx));
              atomicSub(&num_leq_50_[actual_type_id], 1);
#endif // OPTION_DEFRAG
              ASSERT_SUCCESS(allocated_[actual_type_id].deallocate<true>(block_idx));
              ASSERT_SUCCESS(global_free_.allocate<true>(block_idx));
            }
          }
        }
      }

      assert(__activemask() == active);
      // Get pointer from allocation (nullptr if no allocation).
      allocation_bitmap = __shfl_sync(active, allocation_bitmap, leader);
      block_idx = __shfl_sync(active, block_idx, leader);
      assert(block_idx < N);
      result = get_ptr_from_allocation<T>(
          block_idx, __popc(__lanemask_lt() & active), allocation_bitmap);
    } while (result == nullptr);

    return result;
  }

  // Allocate location, construct object and return pointer.
  template<typename T, typename... Args>
  __DEV__ T* make_new(Args&&... args) {
    return new(allocate_new<T>()) T(std::forward<Args>(args)...);
  }

  template<class T>
  __DEV__ void free(T* obj) {
    auto type_id = obj->get_type();
    if (type_id == BlockHelper<T>::kIndex) {
      free_typed(obj);
    } else {
      ASSERT_SUCCESS(TupleHelper<Types...>
          ::template dev_for_all<FreeHelper<T>::InnerHelper>(this, obj));
    }
  }

  // Free object by type ID instead of type.
  template<int TypeIndex>
  __DEV__ void free_untyped(void* obj) {
    auto* typed = static_cast<TYPE_ELEMENT(Types, TypeIndex)*>(obj);
    free(typed);
  }

  // Call a member functions on all objects of type IterT.
  template<class IterT, class T, void(T::*func)()>
  void parallel_do_single_type() {
    ParallelExecutor<ThisAllocator, IterT, T, void, T>
        ::template FunctionWrapper<func>
        ::parallel_do(this, /*shared_mem_size=*/ 0);
  }

  // Call a member function on all objects of type T and its subclasses.
  template<class T, void(T::*func)()>
  void parallel_do() {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelper<ThisAllocator, T, func>
        ::template InnerHelper>(this);
  }

  // Call a member function on all objects of type.
  // Device version (sequential).
  // TODO: This does not enumerate subtypes.
  // TODO: This also enumerates newly-created objects.
  template<class T, typename F, typename... Args>
  __DEV__ void device_do(F func, Args... args) {
    // device_do iterates over objects in a block.
    allocated_[BlockHelper<T>::kIndex].enumerate(
      &SequentialExecutor<T, F, ThisAllocator, Args...>::device_do,
      func, this, args...);
  }

  template<typename T>
  __DEV__ void initialize_iteration() {
    const auto num_blocks = allocated_[BlockHelper<T>::kIndex].scan_num_bits();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks;
         i += blockDim.x * gridDim.x) {
      auto idx = allocated_[BlockHelper<T>::kIndex].scan_get_index(i);
      assert(allocated_[BlockHelper<T>::kIndex][idx]);

      // Initialize block for iteration.
      get_block<T>(idx)->initialize_iteration();
    }
  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  template<typename T>
  __DEV__ BlockBitmapT allocate_in_block(BlockBitmapT* free_bitmap_ptr,
                                         int alloc_size, BlockIndexT block_idx) {
    // Allocation bits.
    BlockBitmapT selected_bits = 0;
    // Set to true if block is full.
    bool block_full;

    BlockBitmapT free_bitmap = *free_bitmap_ptr;

    do {
      // Bit set to 1 if slot is free.
      // TODO: Try different ones.
      const auto rotation_len = warp_id() % 64;
      // TODO: Can we use return value from atomic update in second iteration?
      BlockBitmapT updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      auto free_slots = __popcll(updated_mask);
      auto allocation_size = min(free_slots, alloc_size);
      BlockBitmapT newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        auto next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        auto next_bit_pos_unrot = (next_bit_pos - rotation_len) % 64;
        newly_selected_bits |= 1ULL << next_bit_pos_unrot;
      }

      assert(__popcll(newly_selected_bits) == allocation_size);
      // Count the number of bits that were selected but already set to false
      // by another thread.
      BlockBitmapT before_update = atomicAnd(free_bitmap_ptr, ~newly_selected_bits);
      free_bitmap = before_update & ~newly_selected_bits;
      BlockBitmapT successful_alloc = newly_selected_bits & before_update;
      block_full = (before_update & ~successful_alloc) == 0;

      if (successful_alloc > 0ULL) {
        // At least one slot allocated.
        auto num_successful_alloc = __popcll(successful_alloc);
        alloc_size -= num_successful_alloc;
        selected_bits |= successful_alloc;

        if (block_full) {
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
        }

#ifdef OPTION_DEFRAG
        // Check if more than 50% full now.
        auto prev_full = BlockHelper<T>::kSize - __popcll(before_update);
        if (prev_full <= BlockHelper<T>::kLeq50Threshold
            && prev_full + num_successful_alloc > BlockHelper<T>::kLeq50Threshold) {
          ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
          atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        }
#endif  // OPTION_DEFRAG
      }

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (alloc_size > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return selected_bits;
  }

  // Note: Assuming that the block is leq_50_!
  template<class T>
  __DEV__ void deallocate_block(BlockIndexT block_idx, bool dealloc_leq_50 = true) {
    // Precondition: Block is invalidated.
    assert(get_block<T>(block_idx)->free_bitmap == 0);
    ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));

#ifdef OPTION_DEFRAG
    if (dealloc_leq_50) {
      ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
      atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
    }
#endif  // OPTION_DEFRAG

    ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
    ASSERT_SUCCESS(global_free_.allocate<true>(block_idx));
  }

  template<class T>
  __DEV__ void free_typed(T* obj) {
    obj->~T();
    const auto block_idx = get_block_idx<T>(obj);
    const auto obj_id = get_object_id<T>(obj);
    const auto dealloc_state = get_block<T>(block_idx)->deallocate(obj_id);

    // Note: Different ordering of branches can lead to deadlock!
    if (dealloc_state == kBlockNowActive) {
      ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
#ifdef OPTION_DEFRAG
    } else if (dealloc_state == kBlockNowLeq50Full) {
      ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
#endif  // OPTION_DEFRAG
    } else if (dealloc_state == kBlockNowEmpty) {
      // Assume that block is empty.
      // TODO: Special case if N == 2 or N == 1 for leq_50_.
      if (invalidate_block<T>(block_idx)) {
        // Block is invalidated and no new allocations can be performed.
        deallocate_block<T>(block_idx, /*dealloc_leq_50=*/ true);
      }
    }
  }

  // Helper data structure for freeing objects whose types are subtypes of the
  // declared type. BaseClass is the declared type.
  template<typename BaseClass>
  struct FreeHelper {
    // Iterating over all types T in the allocator.
    template<typename T>
    struct InnerHelper {
      // T is a subclass of BaseClass. Check if same type.
      template<bool Check, int Dummy>
      struct ClassSelector {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          if (obj->get_type() == BlockHelper<T>::kIndex) {
            allocator->free_typed(static_cast<T*>(obj));
            return false;  // No need to check other types.
          } else {
            return true;   // true means "continue processing".
          }
        }
      };

      // T is not a subclass of BaseClass. Skip.
      template<int Dummy>
      struct ClassSelector<false, Dummy> {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          return true;
        }
      };

      __DEV__ bool operator()(ThisAllocator* allocator, BaseClass* obj) {
        return ClassSelector<std::is_base_of<BaseClass, T>::value, 0>::call(
            allocator, obj);
      }
    };
  };

  template<typename T>
  __DEV__ bool is_block_allocated(BlockIndexT index) {
    return allocated_[BlockHelper<T>::kIndex][index];
  }

  template<class T>
  __DEV__ BlockIndexT get_block_idx(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t data_as_int = reinterpret_cast<uintptr_t>(data_);

    assert(((ptr_as_int & kBlockAddrBitmask) - data_as_int) % kBlockSizeBytes == 0);
    return ((ptr_as_int & kBlockAddrBitmask) - data_as_int) / kBlockSizeBytes;
  }

  template<class T>
  __DEV__ static ObjectIndexT get_object_id(T* ptr) {
    return PointerHelper::obj_id_from_obj_ptr(ptr);
  }

  template<class T>
  __DEV__ static T* get_object(typename BlockHelper<T>::BlockType* block,
                               ObjectIndexT obj_id) {
    assert(obj_id < 64);
    return block->make_pointer(obj_id);
  }

  template<class T>
  __DEV__ typename BlockHelper<T>::BlockType* get_block(BlockIndexT block_idx)
      const {
    assert(block_idx < N && block_idx >= 0);
    uintptr_t increment = static_cast<uintptr_t>(block_idx)*kBlockSizeBytes;
    auto* result = reinterpret_cast<typename BlockHelper<T>::BlockType*>(
        data_ + increment);
    assert(reinterpret_cast<char*>(result) >= data_);
    return result;
  }

  template<class T>
  __DEV__ BlockIndexT find_active_block() {
    BlockIndexT block_idx;

    do {
      // Retry a couple of times. May reduce fragmentation.
      int retries = kFindActiveBlockRetries;   // retries=2 before
      do {
        block_idx = active_[BlockHelper<T>::kIndex]
            .template find_allocated<false>(retries + blockIdx.x);
      } while (block_idx == Bitmap<BlockIndexT, N>::kIndexError
               && --retries > 0);

      if (block_idx == Bitmap<BlockIndexT, N>::kIndexError) {
        // TODO: May be out of memory here.
        block_idx = global_free_.deallocate_seed(blockIdx.x);
        assert(block_idx != (Bitmap<BlockIndexT, N>::kIndexError));  // OOM
        initialize_block<T>(block_idx);
        ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
#ifdef OPTION_DEFRAG
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
        atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
#endif  // OPTION_DEFRAG
        ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      }
    } while (block_idx == Bitmap<BlockIndexT, N>::kIndexError);

    assert(block_idx < N);
    return block_idx;
  }

  template<class T>
  __DEV__ void initialize_block(BlockIndexT block_idx) {
    static_assert(sizeof(typename BlockHelper<T>::BlockType)
          % kNumBlockElements == 0,
        "Internal error: SOA block not aligned to 64 bytes.");
    assert(block_idx >= 0);
    auto* block_ptr = get_block<T>(block_idx);
    assert(reinterpret_cast<char*>(block_ptr) >= data_);
    new(block_ptr) typename BlockHelper<T>::BlockType();
  }

  template<class T>
  __DEV__ T* get_ptr_from_allocation(BlockIndexT block_idx, int rank,
                                     BlockBitmapT allocation) {
    assert(block_idx < N);
    assert(rank < 32);

    // Get index of rank-th first bit set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      allocation &= allocation - 1;
    }

    auto position = __ffsll(allocation);

    if (position > 0) {
      // Allocation successful.
      return get_block<T>(block_idx)->make_pointer(position - 1);
    } else {
      return nullptr;
    }
  }

  // Precondition: Block is active.
  // Postcondition: Do not change active status.
  template<class T>
  __DEV__ bool invalidate_block(BlockIndexT block_idx) {
    auto* block = get_block<T>(block_idx);

    while (true) {
      auto old_free_bitmap = atomicExch(&block->free_bitmap, 0ULL);
      if (old_free_bitmap == BlockHelper<T>::BlockType::kBitmapInitState) {
        return true;
      } else if (old_free_bitmap != 0ULL) {
        // TODO: Check correctness of this code path. It is rarely executed.

        // block->free_bitmap = old_free_bitmap;

        // free_bitmap now 0. We should deactivate the block here, but since
        // the block deallocation procedure expects the invalid block to be
        // active, we omit this and only deactivate it in the specical case
        // described below.

        // At least one bit was modified. Rollback invalidation.
        auto before_rollback = atomicOr(&block->free_bitmap, old_free_bitmap);
        if (before_rollback > 0ULL) {
#ifdef OPTION_DEFRAG
          if ((BlockHelper<T>::kSize - __popcll(old_free_bitmap)
                  > BlockHelper<T>::kLeq50Threshold)
              && (BlockHelper<T>::kSize - __popcll(before_rollback)
                  <= BlockHelper<T>::kLeq50Threshold)) {
            // Some thread is trying to set the bit in the leq_50 bitmap.
            ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
            atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
          }
#endif  // OPTION_DEFRAG

          // At least 1 other thread deallocated an object (set a bit). That
          // thread is attempting to make the block active. For this to
          // succeed, we have to deactivate the block here.
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
        }

        if ((before_rollback | old_free_bitmap) !=
            BlockHelper<T>::BlockType::kBitmapInitState) {
          break;
        }  // else: Block emptied again. Try invalidating it again.
      }
    }

    return false;
  }

  static const int kNumTypes = sizeof...(Types);

  static const int kBlockSizeBytes =
      sizeof(typename BlockHelper<typename TupleHelper<Types...>
             ::Type64BlockSizeMin>::BlockType);

  Bitmap<BlockIndexT, N> global_free_;

  Bitmap<BlockIndexT, N> allocated_[kNumTypes];

  Bitmap<BlockIndexT, N> active_[kNumTypes];

#ifdef OPTION_DEFRAG
  // Bit set if block is <= 50% full and active.
  Bitmap<BlockIndexT, N> leq_50_[kNumTypes];
  BlockIndexT num_leq_50_[kNumTypes];

  // Temporary storage for defragmentation records.
  SoaDefragRecords<BlockBitmapT, kMaxDefragRecords> defrag_records_;
#endif  // OPTION_DEFRAG

  char* data_;

  static const BlockIndexT kN = N;

  static const size_t kDataBufferSize = static_cast<size_t>(N)*kBlockSizeBytes;
};


template<typename T, typename F, typename AllocatorT, typename... Args>
__DEV__ void SequentialExecutor<T, F, AllocatorT, Args...>::device_do(
    BlockIndexT block_idx, F func, AllocatorT* allocator, Args... args) {
  auto* block = allocator->template get_block<T>(block_idx);
  auto bitmap = block->allocation_bitmap();

  while (bitmap != 0ULL) {
    auto pos = __ffsll(bitmap) - 1;
    bitmap &= bitmap - 1;

    auto* obj = AllocatorT::template get_object<T>(block, pos);
    (obj->*func)(args...);
  }
}


// This are textual headers. Must be included at the end of the file.
#ifdef OPTION_DEFRAG
#include "allocator/soa_defrag.inc"
#endif  // OPTION_DEFRAG

#include "allocator/soa_debug.inc"

#endif  // ALLOCATOR_SOA_ALLOCATOR_H
