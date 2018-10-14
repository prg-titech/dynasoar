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
#include "allocator/soa_defrag.h"
#include "allocator/soa_executor.h"
#include "allocator/soa_field.h"
#include "allocator/tuple_helper.h"
#include "allocator/util.h"


// TODO: Fix visibility.
template<uint32_t N_Objects, class... Types>
class SoaAllocator {
 public:
  using ThisAllocator = SoaAllocator<N_Objects, Types...>;

  static const uint8_t kObjectAddrBits = 6;
  static const uint32_t kNumBlockElements = 64;
  static const uint64_t kObjectAddrBitmask = kNumBlockElements - 1;
  static const uint64_t kBlockAddrBitmask = 0xFFFFFFFFFFC0;
  static_assert(kNumBlockElements == 64,
                "Not implemented: Block size != 64.");
  static const int N = N_Objects / kNumBlockElements;

  static_assert(N_Objects % kNumBlockElements == 0,
                "N_Objects Must be divisible by BlockSize.");

  template<typename T>
  struct BlockHelper {
    static const int kIndex = TYPE_INDEX(Types..., T);

    static const int kSize =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::kPadded64BlockMinSize>::kSize;

    static const int kBytes =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::kPadded64BlockMinSize>::kBytes;

    using BlockType = SoaBlock<T, kIndex, kSize, 64>;

    static const int kLeq50Threshold = BlockType::kLeq50Threshold;
  };

  using BlockBitmapT = typename BlockHelper<
      typename TupleHelper<Types...>::NonAbstractType>::BlockType::BitmapT;

  template<typename T>
  struct TypeId {
    static const uint8_t value =
        TupleHelper<Types...>::template TupleIndex<T, 0>::value;
  };

  __DEV__ static uint8_t get_type(const void* ptr) {
    auto ptr_base = reinterpret_cast<uintptr_t>(ptr);
    uint8_t type_id = ptr_base >> 56;  // Truncated.
    assert(type_id < kNumTypes);
    return type_id;
  }

  __DEV__ SoaAllocator(char* data_buffer) : data_(data_buffer) {
    // Check alignment of data storage buffer.
    assert(reinterpret_cast<uintptr_t>(data_) % 64 == 0);

    global_free_.initialize(true);
    for (int i = 0; i < kNumTypes; ++i) {
      allocated_[i].initialize(false);
      active_[i].initialize(false);
      leq_50_[i].initialize(false);
      num_leq_50_[i] = 0;
    }
  }

  __DEV__ SoaAllocator(const ThisAllocator&) = delete;

  // Try to allocate everything in the same block.
  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    T* result = nullptr;

    do {
      const unsigned active = __activemask();
      // Leader thread is the first thread whose mask bit is set to 1.
      const int leader = __ffs(active) - 1;
      assert(leader >= 0 && leader < 32);
      // Use lane mask to empty all bits higher than the current thread.
      // The rank of this thread is the number of bits set to 1 in the result.
      const unsigned int rank = lane_id();
      assert(rank < 32);

      // Values to be calculated by the leader.
      uint32_t block_idx;
      BlockBitmapT allocation_bitmap;
      if (rank == leader) {
        assert(__popc(__activemask()) == 1);    // Only one thread executing.

        block_idx = find_active_block<T>();
        auto* block = get_block<T>(block_idx);
        BlockBitmapT* free_bitmap = &block->free_bitmap;
        allocation_bitmap = allocate_in_block<T>(
            free_bitmap, __popc(active), block_idx);
        int num_allocated = __popcll(allocation_bitmap);

        uint8_t actual_type_id = block->type_id;
        if (actual_type_id != BlockHelper<T>::kIndex) {
          // Block deallocated and initialized for a new type between lookup
          // from active bitmap and here. This is extremely unlikely!
          // But possible.
          // Undo allocation and update bitmaps accordingly.

          // Note: Cannot be in invalidation here, because we are certain that
          // we allocated the bits that we are about to deallocate here.
          auto before_undo = atomicOr(free_bitmap, allocation_bitmap);
          int slots_before_undo = __popcll(before_undo);

          if (N - slots_before_undo > BlockHelper<T>::kLeq50Threshold
              && N - slots_before_undo - num_allocated <= BlockHelper<T>::kLeq50Threshold) {
            ASSERT_SUCCESS(leq_50_[actual_type_id].allocate<true>(block_idx));
            atomicAdd(&num_leq_50_[actual_type_id], 1);
          }

          // Cases to handle: block now active again or block empty now.
          if (slots_before_undo == 0) {
            // Block became active. (Was full.)
            ASSERT_SUCCESS(active_[actual_type_id].allocate<true>(block_idx));
          } else if (slots_before_undo == N - 1) {
            // Block now empty.
            if (invalidate_block<T>(block_idx)) {
              // Block is invalidated and no new allocations can be performed.
              ASSERT_SUCCESS(active_[actual_type_id].deallocate<true>(block_idx));
              ASSERT_SUCCESS(leq_50_[actual_type_id].deallocate<true>(block_idx));
              atomicSub(&num_leq_50_[actual_type_id], 1);
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

    return new(result) T(args...);
  }

  template<class T>
  __DEV__ void free(T* obj) {
    uint8_t type_id = obj->get_type();
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

  // Call a member functions on all objects of a type.
  template<int W_MULT, class T, void(T::*func)()>
  void parallel_do(int num_blocks, int num_threads) {
    ParallelExecutor<ThisAllocator, T, void, T>
        ::template FunctionWrapper<func>
        ::parallel_do(this, num_blocks, num_threads, /*shared_mem_size=*/ 0);
  }

  template<typename T>
  struct SoaObjectCopier {
    // Copies a single field value from one block to another one.
    template<typename SoaFieldHelperT>
    struct ObjectCopyHelper {
      using SoaFieldType = SoaField<typename SoaFieldHelperT::OwnerClass,
                                    SoaFieldHelperT::kIndex>;

      __DEV__ bool operator()(char* source_block_base, char* target_block_base,
                              uint8_t source_slot, uint8_t target_slot) {
        // TODO: Optimize copy routine for single value. Should not use the
        // assignment operator here.
        // TODO: Make block size a template parameter.
        typename SoaFieldHelperT::type* source_ptr =
            SoaFieldType::data_ptr_from_location(
                source_block_base, BlockHelper<T>::kSize, source_slot);
        typename SoaFieldHelperT::type* target_ptr =
            SoaFieldType::data_ptr_from_location(
                target_block_base, BlockHelper<T>::kSize, target_slot);
        *target_ptr = *source_ptr;

        return true;  // Continue processing.
      }
    };
  };

  template<typename T>
  __DEV__ void defrag_move() {
    // Use 32 threads per SOA block, so that we can use warp shuffles instead
    // of shared memory.
    assert(blockDim.x % 32 == 0);
    int slot_id = threadIdx.x % 32;
    int record_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    const unsigned active = __activemask();

    // Defrag record.
    uint32_t source_block_idx, target_block_idx;
    BlockBitmapT source_bitmap, target_bitmap;

    if (slot_id == 0) {
      source_block_idx = leq_50_work_.deallocate_seed(record_id);
      target_block_idx = leq_50_work_.deallocate_seed(record_id+37);

      assert(get_block<T>(source_block_idx)->type_id
          == BlockHelper<T>::kIndex);
      assert(get_block<T>(target_block_idx)->type_id
          == BlockHelper<T>::kIndex);

      // Invert free_bitmap to get a bitmap of allocations.
      source_bitmap = ~get_block<T>(source_block_idx)->free_bitmap
                      & BlockHelper<T>::BlockType::kBitmapInitState;
      assert(__popcll(source_bitmap) == BlockHelper<T>::kSize   // == N - #free
          - __popcll(get_block<T>(source_block_idx)->free_bitmap));
      // Target bitmap contains all free slots in target block.
      target_bitmap = get_block<T>(target_block_idx)->free_bitmap;

      // Copy to global memory for scan step.
      defrag_records_[record_id].source_block_idx = source_block_idx;
      defrag_records_[record_id].target_block_idx = target_block_idx;
      defrag_records_[record_id].source_bitmap = source_bitmap;
      defrag_records_[record_id].target_bitmap = target_bitmap;

      assert(__popcll(target_bitmap) >= __popcll(source_bitmap));
    }

    // Shuffle defrag records from thread 0.
    source_block_idx = __shfl_sync(active, source_block_idx, 0);
    target_block_idx = __shfl_sync(active, target_block_idx, 0);
    source_bitmap = __shfl_sync(active, source_bitmap, 0);
    target_bitmap = __shfl_sync(active, target_bitmap, 0);

    // Find index of bit in bitmaps.
    int num_moves = __popcll(source_bitmap);
    assert(num_moves <= BlockHelper<T>::kSize/2);
    for (int i = 0; i < slot_id; ++i) {
      // Clear least significant bit.
      source_bitmap &= source_bitmap - 1;
      target_bitmap &= target_bitmap - 1;
    }
    int source_object_id = __ffsll(source_bitmap) - 1;
    int target_object_id = __ffsll(target_bitmap) - 1;

    // Move objects.
    if (source_object_id > -1) {
      assert(target_object_id > -1);

      SoaClassHelper<T>::template dev_for_all<
              SoaObjectCopier<T>::ObjectCopyHelper, true>(
          reinterpret_cast<char*>(get_block<T>(source_block_idx)),
          reinterpret_cast<char*>(get_block<T>(target_block_idx)),
          source_object_id, target_object_id);
    }
    
    // Last thread performs all update, because it has access to the new
    // target_bitmap (with all slots occupied).
    if (slot_id == num_moves - 1) {
      // Invalidate source block.
      get_block<T>(source_block_idx)->free_bitmap = 0ULL;
      // Delete source block.
      // Precond.: Block is leq_50_, active, allocated.
      deallocate_block<T>(source_block_idx);

      // Update free_bitmap in target block.
      target_bitmap &= target_bitmap - 1;   // Clear one more bit.
      get_block<T>(target_block_idx)->free_bitmap = target_bitmap;

      // Update state of target block.
      int num_target_after = __popcll(
          ~target_bitmap & BlockHelper<T>::BlockType::kBitmapInitState);

      if (num_target_after == BlockHelper<T>::kSize) {
        // Block is now full.
         ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(
            target_block_idx));
      }

      if (num_target_after > BlockHelper<T>::kLeq50Threshold) {
        // Block is now more than 50% full.
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(
            target_block_idx));
        atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
      }
    }
  }

  // Select fields of type DefragT* and rewrite pointers if necessary.
  // DefragT: Type that was defragmented.
  // ScanClassT: Class which is being scanned for affected fields.
  // SoaFieldHelperT: SoaFieldHelper of potentially affected field.
  template<typename DefragT>
  struct SoaPointerUpdater {
    template<typename ScanClassT>
    struct ClassIterator {
      using ThisClass = ClassIterator<ScanClassT>;

      // Checks if this field should be rewritten.
      template<typename SoaFieldHelperT>
      struct FieldChecker {
        using FieldType = typename SoaFieldHelperT::type;

        bool operator()() {
          // Stop iterating if at least one field must be rewritten.
          return !(std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value);
        }
      };

      // Scan and rewrite field.
      template<typename SoaFieldHelperT>
      struct FieldUpdater {
        using FieldType = typename SoaFieldHelperT::type;

        using SoaFieldType = SoaField<typename SoaFieldHelperT::OwnerClass,
                                      SoaFieldHelperT::kIndex>;

        // Scan this field.
        template<bool Check, int Dummy> 
        struct FieldSelector {
          template<typename... Args>
          __DEV__ static void call(ThisAllocator* allocator,
                                   ScanClassT* object, int num_records,
                                   DefragRecord<BlockBitmapT>* records) {
            //extern __shared__ DefragRecord<BlockBitmapT> records[];

            // Location of field value to be scanned/rewritten.
            // TODO: This is inefficient. We first build a pointer and then
            // disect it again. (Inside *_from_obj_ptr().)
            FieldType* scan_location =
                SoaFieldType::data_ptr_from_obj_ptr(object);
            assert(reinterpret_cast<char*>(scan_location) >= allocator->data_);
            FieldType scan_value = *scan_location;

            if (scan_value == nullptr) return;
            // Check if value points to an object of type DefragT.
            if (scan_value->get_type() != BlockHelper<DefragT>::kIndex) return;

            // Calculate block index of scan_value.
            // TODO: Find a better way to do this.
            // TODO: There could be some random garbage in the field if it was
            // not initialized. Replace asserts with if-check and return stmt.
            char* block_base =
                PointerHelper::block_base_from_obj_ptr(scan_value);
            assert(block_base >= allocator->data_
                   && block_base < allocator->data_ + kDataBufferSize);
            assert((block_base - allocator->data_) % kBlockSizeBytes == 0);
            uint32_t scan_block_idx = (block_base - allocator->data_)
                / kBlockSizeBytes;
            assert(scan_block_idx < N);

            // Scan shared memory for matching defrag record.
            // TODO: Replace with binary search or hash-based approach.
            for (int i = 0; i < num_records; ++i) {
              if (records[i].source_block_idx == scan_block_idx) {
                // This pointer must be rewritten.
                int src_obj_id = PointerHelper::obj_id_from_obj_ptr(scan_value);
                assert(src_obj_id < BlockHelper<DefragT>::kSize);
                assert((records[i].source_bitmap & (1ULL << src_obj_id)) != 0);

                // First src_obj_id bits are set to 1.
                BlockBitmapT cnt_mask = src_obj_id ==
                   63 ? (~0ULL) : ((1ULL << (src_obj_id + 1)) - 1);
                assert(__popcll(cnt_mask) == src_obj_id + 1);
                int src_bit_cnt =
                    __popcll(cnt_mask & records[i].source_bitmap) - 1;
                assert(src_bit_cnt >= 0 && src_bit_cnt < 64);

                // Find src_bit_cnt-th bit in target bitmap.
                BlockBitmapT target_bitmap = records[i].target_bitmap;
                for (int j = 0; j < src_bit_cnt; ++j) {
                  target_bitmap &= target_bitmap - 1;
                }
                int target_obj_id = __ffsll(target_bitmap) - 1;
                assert(target_obj_id < BlockHelper<DefragT>::kSize);
                assert(target_obj_id >= 0);
                assert((allocator->template get_block<DefragT>(
                        records[i].target_block_idx)->free_bitmap
                    & (1ULL << target_obj_id)) == 0);

                // Rewrite pointer.
                assert(records[i].target_block_idx < N);
                auto* target_block = allocator->template get_block<
                    typename std::remove_pointer<FieldType>::type>(
                        records[i].target_block_idx);
                *scan_location = PointerHelper::rewrite_pointer(
                        scan_value, target_block, target_obj_id);
                assert(PointerHelper::block_base_from_obj_ptr(*scan_location)
                    == reinterpret_cast<char*>(target_block));
                assert((*scan_location)->get_type()
                    == BlockHelper<DefragT>::kIndex);
                break;
              }
            }
          }
        };

        // Do not scan this field.
        template<int Dummy>
        struct FieldSelector<false, Dummy> {
          __DEV__ static void call(ThisAllocator* allocator,
                                   ScanClassT* object,
                                   int num_records,
                                   DefragRecord<BlockBitmapT>* records) {}
        };

        __DEV__ bool operator()(ThisAllocator* allocator, ScanClassT* object,
                                int num_records, DefragRecord<BlockBitmapT>* records) {
          // Rewrite field if field type is a super class (or exact class)
          // of DefragT.
          FieldSelector<std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value, 0>::call(
              allocator, object, num_records, records);
          return true;
        }
      };

      bool operator()(ThisAllocator* allocator, int num_records) {
        bool process_class = SoaClassHelper<ScanClassT>::template for_all<
            FieldChecker, /*IterateBase=*/ true>();
        if (process_class) {
          // TODO: Optimize. No need to do another scan here.
          kernel_init_iteration<ThisAllocator, ScanClassT><<<128, 128>>>(allocator);
          gpuErrchk(cudaDeviceSynchronize());

          ParallelExecutor<ThisAllocator, ScanClassT, void,
                           SoaBase<ThisAllocator>, /*Args...=*/ ThisAllocator*, int>
              ::template FunctionWrapper<&SoaBase<ThisAllocator>
                  ::template rewrite_object<ThisClass, ScanClassT>>
              ::parallel_do(allocator, 128, 128,
                            num_records*sizeof(DefragRecord<BlockBitmapT>),
                            allocator, num_records);
        }

        return true;  // Continue processing.
      }
    };
  };

  // Should be invoked from host side.
  template<typename T>
  void parallel_defrag(int max_records) {
    // Create working copy of leq_50_ bitmap.
    kernel_initialize_leq<T><<<256, 256>>>(this);
    gpuErrchk(cudaDeviceSynchronize());

    // Determine number of records.
    auto num_leq_blocks =
        copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);
    int num_records = min(max_records, num_leq_blocks/2);
    num_records = max(0, num_records);

    // Move objects. 4 SOA block per CUDA block. 32 threads per block.
    // (Because blocks are at most 50% full.)
    kernel_defrag_move<T><<<
        (num_records + 4 - 1) / 4, 128>>>(this, num_records);
    gpuErrchk(cudaDeviceSynchronize());

    // Scan and rewrite pointers.
    TupleHelper<Types...>
        ::template for_all<SoaPointerUpdater<T>::template ClassIterator>(
            this, num_records);
  }

  template<typename T>
  __DEV__ void initialize_iteration() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
      if (allocated_[BlockHelper<T>::kIndex][i]) {
        // Initialize block.
        get_block<T>(i)->initialize_iteration();
      }
    }
  }

  template<typename T>
  __DEV__ void initialize_leq_work_bitmap() {
    leq_50_work_.initialize(leq_50_[BlockHelper<T>::kIndex]);
  }

  // The number of allocated slots of a type. (#blocks * blocksize)
  template<class T>
  __DEV__ uint32_t DBG_allocated_slots() {
    uint32_t counter = 0;
    for (int i = 0; i < N; ++i) {
      if (allocated_[BlockHelper<T>::kIndex][i]) {
        counter += get_block<T>(i)->DBG_num_bits();
      }
    }
    return counter;
  }

  // The number of actually used slots of a type. (#blocks * blocksize)
  template<class T>
  __DEV__ uint32_t DBG_used_slots() {
    uint32_t counter = 0;
    for (int i = 0; i < N; ++i) {
      if (allocated_[BlockHelper<T>::kIndex][i]) {
        counter += get_block<T>(i)->DBG_allocated_bits();
      }
    }
    return counter;
  }

  template<typename T>
  struct SoaTypeDbgPrinter {
    bool operator()() {
      printf("sizeof(%s) = %lu\n", typeid(T).name(), sizeof(T));
      printf("block size(%s) = %i\n", typeid(T).name(), BlockHelper<T>::kSize);
      printf("data segment bytes(%s) = %i\n", typeid(T).name(),
             BlockHelper<T>::kBytes);
      printf("block bytes(%s) = %lu\n", typeid(T).name(),
             sizeof(typename BlockHelper<T>::BlockType));
      SoaClassHelper<T>::DBG_print_stats();
      return true;  // true means "continue processing".
    }
  };

  static void DBG_print_stats() {
    printf("----------------------------------------------------------\n");
    TupleHelper<Types...>::template for_all<SoaTypeDbgPrinter>();
    printf("Smallest block type: %s at %i bytes.\n",
           typeid(typename TupleHelper<Types...>::Type64BlockSizeMin).name(),
           TupleHelper<Types...>::kPadded64BlockMinSize);
    printf("Block size bytes: %i\n", kBlockSizeBytes);
    printf("Data buffer size (MB): %f\n", kDataBufferSize/1024.0/1024.0);
    printf("----------------------------------------------------------\n");
  }

  __DEV__ void DBG_print_state_stats() {

  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  template<typename T>
  __DEV__ BlockBitmapT allocate_in_block(BlockBitmapT* free_bitmap_ptr,
                                         int alloc_size, uint32_t block_idx) {
    // Allocation bits.
    BlockBitmapT selected_bits = 0;
    // Set to true if block is full.
    bool block_full;

    BlockBitmapT free_bitmap = *free_bitmap_ptr;

    do {
      // Bit set to 1 if slot is free.
      unsigned int rotation_len = warp_id() % 64;
      // TODO: Can we use return value from atomic update in second iteration?
      BlockBitmapT updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      int free_slots = __popcll(updated_mask);
      int allocation_size = min(free_slots, alloc_size);
      BlockBitmapT newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        int next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        int next_bit_pos_unrot = (next_bit_pos - rotation_len) % 64;
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
        int num_successful_alloc = __popcll(successful_alloc);
        alloc_size -= num_successful_alloc;
        selected_bits |= successful_alloc;

        if (block_full) {
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
        }

        // Check if more than 50% full now.
        int prev_full = BlockHelper<T>::kSize - __popcll(before_update);
        if (prev_full <= BlockHelper<T>::kLeq50Threshold
            && prev_full + num_successful_alloc > BlockHelper<T>::kLeq50Threshold) {
          ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
          atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        }
      }

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (alloc_size > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return selected_bits;
  }

  // Note: Assuming that the block is leq_50_!
  template<class T>
  __DEV__ void deallocate_block(uint32_t block_idx) {
    ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
    ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
    atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
    ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
    ASSERT_SUCCESS(global_free_.allocate<true>(block_idx));
  }

  template<class T>
  __DEV__ void free_typed(T* obj) {
    obj->~T();
    const uint32_t block_idx = get_block_idx<T>(obj);
    const uint32_t obj_id = get_object_id<T>(obj);
    const auto dealloc_state = get_block<T>(block_idx)->deallocate(obj_id);

    if (dealloc_state == kBlockNowActive) {
      ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
    } else if (dealloc_state == kBlockNowEmpty) {
      // Assume that block is empty.
      // TODO: Special case if N == 2 or N == 1 for leq_50_.
      if (invalidate_block<T>(block_idx)) {
        // Block is invalidated and no new allocations can be performed.
        deallocate_block<T>(block_idx);
      }
    } else if (dealloc_state == kBlockNowLeq50Full) {
      ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
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
  __DEV__ bool is_block_allocated(uint32_t index) {
    return allocated_[BlockHelper<T>::kIndex][index];
  }

  template<class T>
  __DEV__ uint32_t get_block_idx(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t data_as_int = reinterpret_cast<uintptr_t>(data_);

    assert(((ptr_as_int & kBlockAddrBitmask) - data_as_int) % kBlockSizeBytes == 0);
    return ((ptr_as_int & kBlockAddrBitmask) - data_as_int) / kBlockSizeBytes;
  }

  template<class T>
  __DEV__ uint32_t get_object_id(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    return ptr_as_int & kObjectAddrBitmask; 
  }

  template<class T>
  __DEV__ T* get_object(typename BlockHelper<T>::BlockType* block, uint32_t obj_id) {
    assert(obj_id < 64);
    return block->make_pointer(obj_id);
  }

  template<class T>
  __DEV__ typename BlockHelper<T>::BlockType* get_block(uint32_t block_idx) {
    assert(block_idx < N);
    return reinterpret_cast<typename BlockHelper<T>::BlockType*>(
        data_ + block_idx*kBlockSizeBytes);
  }

  template<class T>
  __DEV__ uint32_t find_active_block() {
    uint32_t block_idx;

    do {
      // Retry a couple of times. May reduce fragmentation.
      // TODO: Tune number of retries.
      int retries = 5;   // retries=2 before
      do {
        block_idx = active_[BlockHelper<T>::kIndex]
            .template find_allocated<false>(retries);
      } while (block_idx == Bitmap<uint32_t, N>::kIndexError
               && --retries > 0);

      if (block_idx == Bitmap<uint32_t, N>::kIndexError) {
        // TODO: May be out of memory here.
        block_idx = global_free_.deallocate();
        assert(block_idx != (Bitmap<uint32_t, N>::kIndexError));  // OOM
        initialize_block<T>(block_idx);
        ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
        atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      }
    } while (block_idx == Bitmap<uint32_t, N>::kIndexError);

    assert(block_idx < N);
    return block_idx;
  }

  template<class T>
  __DEV__ void initialize_block(uint32_t block_idx) {
    static_assert(sizeof(typename BlockHelper<T>::BlockType)
          % kNumBlockElements == 0,
        "Internal error: SOA block not aligned to 64 bytes.");
    new(get_block<T>(block_idx)) typename BlockHelper<T>::BlockType();
  }

  template<class T>
  __DEV__ T* get_ptr_from_allocation(uint32_t block_idx, int rank,
                                     BlockBitmapT allocation) {
    assert(block_idx < N);
    assert(rank < 32);

    // Get index of rank-th first bit set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      allocation &= allocation - 1;
    }

    int position = __ffsll(allocation);

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
  __DEV__ bool invalidate_block(uint32_t block_idx) {
    auto* block = get_block<T>(block_idx);

    while (true) {
      auto old_free_bitmap = atomicExch(&block->free_bitmap, 0ULL);
      if (old_free_bitmap == BlockHelper<T>::BlockType::kBitmapInitState) {
        return true;
      } else if (old_free_bitmap != 0ULL) {
        // block->free_bitmap = old_free_bitmap;

        // free_bitmap now 0. We should deactivate the block here, but since
        // the block deallocation procedure expects the invalid block to be
        // active, we omit this and only deactivate it in the specical case
        // described below.

        // At least one bit was modified. Rollback invalidation.
        auto before_rollback = atomicOr(&block->free_bitmap, old_free_bitmap);
        if (before_rollback > 0ULL) {
          if (N - __popcll(old_free_bitmap) > BlockHelper<T>::kLeq50Threshold
              && N - __popcll(before_rollback) <= BlockHelper<T>::kLeq50Threshold) {
            // Some thread is trying to set the bit in the leq_50 bitmap.
            ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
            atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
          }

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

  static const int kBlockSizeBytes = sizeof(typename BlockHelper<
      typename TupleHelper<Types...>::NonAbstractType>::BlockType);

  Bitmap<uint32_t, N> global_free_;

  Bitmap<uint32_t, N> allocated_[kNumTypes];

  Bitmap<uint32_t, N> active_[kNumTypes];

  // Bit set if block is <= 50% full and active.
  Bitmap<uint32_t, N> leq_50_[kNumTypes];
  unsigned int num_leq_50_[kNumTypes];

  // Copy of leq_50_ used during defragmentation.
  Bitmap<uint32_t, N> leq_50_work_;

  // Temporary storage for defragmentation records.
  DefragRecord<BlockBitmapT> defrag_records_[8192];

  char* data_;

  static const uint32_t kN = N;

  static const size_t kDataBufferSize = static_cast<size_t>(N)*kBlockSizeBytes;
};

#endif  // ALLOCATOR_SOA_ALLOCATOR_H
