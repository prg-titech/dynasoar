#ifndef ALLOCATOR_SOA_ALLOCATOR_H
#define ALLOCATOR_SOA_ALLOCATOR_H

#include <assert.h>
#include <tuple>

#include "bitmap/bitmap.h"

#define __DEV__ __device__

struct BlockAllocationResult {
  __device__ BlockAllocationResult(uint64_t allocation_mask_p,
                                   bool block_full_p)
      : allocation_mask(allocation_mask_p), block_full(block_full_p) {}

  uint64_t allocation_mask;

  // Set to true if this allocation request filled up the block entirely.
  bool block_full;
};

template<class T, int N>
class SoaBlock {
 public:
  __DEV__ SoaBlock() {
    free_bitmap = ~0ULL;
  }

  __DEV__ uint64_t invalidate() {
    return static_cast<uint64_t>(atomicExch(&free_bitmap, 0ULL));
  }

  __DEV__ void uninvalidate(uint64_t previous_val) {
    free_bitmap = previous_val;
    // TODO: Thread fence?
  }

  __DEV__ bool deallocate(int position) {
    unsigned long long int before = atomicOr(&free_bitmap, 1ULL << position);
    // TODO: Choose more efficient operation (possible?).
    return __popcll(before) == 63;
  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  __DEV__ BlockAllocationResult allocate(int bits_to_allocate) {
    // Allocation bits.
    unsigned long long int selected_bits = 0;
    // Set to true if this allocation filled up the block.
    bool filled_block = false;
    // Helper variables used inside the loop and in the loop condition.
    unsigned long long int before_update, successful_alloc;

    do {
      // Bit set to 1 if slot is free.
      unsigned long long int updated_mask = free_bitmap;
      // If there are not enough free slots, allocate as many as possible.
      int free_slots = __popcll(updated_mask);
      int allocation_size = min(free_slots, bits_to_allocate);
      unsigned long long int newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        // TODO: To reduce collisions attempt to start allocation at
        // different positions (rotating shift).
        int next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        newly_selected_bits |= 1ULL << next_bit_pos;
      }

      assert(__popcll(newly_selected_bits) == allocation_size);
      // Count the number of bits that were selected but already set to false
      // by another thread.
      before_update = atomicAnd(&free_bitmap, ~newly_selected_bits);
      successful_alloc = newly_selected_bits & before_update;
      bits_to_allocate -= __popcll(successful_alloc);
      selected_bits |= successful_alloc;

      // Block full if at least one slot was allocated and "before update"
      // bit-and "now allocated" indicates that block is full.
      bool block_full = (before_update & ~successful_alloc) == 0;
      filled_block = successful_alloc > 0 && block_full;

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (bits_to_allocate > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return BlockAllocationResult(selected_bits, filled_block);
  }

 private:
  // Bitmap of free slots.
  unsigned long long int free_bitmap;
};

template<uint32_t N_Objects, class... Types>
class SoaAllocator {
 public:
  __DEV__ void initialize() {
    global_free_.initialize(true);
    for (int i = 0; i < kNumTypes; ++i) {
      allocated_[i].initialize(false);
      active_[i].initialize(false);
    }
  }

  // Try to allocate everything in the same block.
  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    T* result = nullptr;

    do {
      const unsigned int active = __activemask();
      // Leader thread is the first thread whose mask bit is set to 1.
      const int leader = __ffs(active) - 1;
      // Use lane mask to empty all bits higher than the current thread.
      // The rank of this thread is the number of bits set to 1 in the result.
      const unsigned int rank = __popc(active & __lanemask_lt());

      // Values to be calculated by the leader.
      uint32_t block_idx;
      uint64_t allocation_bitmap;
      if (rank == leader) {
        block_idx = find_active_block<T>();
        const auto allocation = allocate_in_block(block_idx,
                                                  /*num=*/ __popc(active));
        allocation_bitmap = allocation.allocation_mask;

        if (allocation.block_full) {
          // This request filled up the block entirely.
          bool success = active_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
          assert(success);
        }
      }

      // Get pointer from allocation (nullptr if no allocation).
      allocation_bitmap = __shfl_sync(active, allocation_bitmap, leader);
      result = get_ptr_from_allocation<T>(block_idx, rank, allocation_bitmap);
    } while (result == nullptr);

    return new(result) T(args...);
  }

  template<class T>
  __DEV__ void free(T* obj) {
    obj->~T();
    const uint32_t block_idx = get_block_idx<T>(obj);
    const uint32_t obj_id = get_object_id<T>(obj);
    const bool last_dealloc = deallocate_in_block(block_idx, obj_id);

    if (last_dealloc) {
      // Block is now empty.
      uint64_t before_invalidate = invalidate_block<T>(block_idx);
      if (before_invalidate == 0) {
        // Block is invalidated and no new allocations can be performed.
        bool success = active_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
        assert(success);
        success = allocated_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
        assert(success);
        success = global_free_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
        assert(success);
      } else {
        uninvalidate_block<T>(block_idx, before_invalidate);
      }
    }
  }

 private:
  template<class T>
  __DEV__ uint32_t find_active_block() {
    uint32_t block_idx;

    do {
      // TODO: Retry a couple of times. May reduce fragmentation.
      block_idx = active_[TupleIndex<T, TupleType>::value].find_allocated();

      if (block_idx == Bitmap<uint32_t, N>::kIndexError) {
        block_idx = global_free_.deallocate();
        assert(block_idx != Bitmap<uint32_t, N>::kIndexError);
        initialize_block<T>(block_idx);
        bool success = allocated_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
        assert(success);
        success = active_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
      }
    } while (block_idx == Bitmap<uint32_t, N>::kIndexError);

    return block_idx;
  }

  template<class T>
  __DEV__ void initialize_block(uint32_t block_idx) {
    new(get_block(block_idx)) Block<T, kNumBlockElements>();
  }

  template<class T>
  __DEV__ uint32_t get_block_idx(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    return ptr_as_int & kBlockAddrBitmask;
  }

  template<class T>
  __DEV__ uint32_t get_object_id(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    return ptr_as_int & kObjectAddrBitmask; 
  }

  template<class T>
  __DEV__ SoaBlock<T, kNumBlockElements> get_block(uint32_t block_idx) {
    return reinterpret_cast<SoaBlock<T, kNumBlockElements>*>(
        data_ + block_idx*kBlockMaxSize);
  }

  template<class T>
  __DEV__ BlockAllocationResult allocate_in_block(uint32_t block_idx,
                                                  int num_objects) {
    auto* block = get_block<T>(block_idx);
    // Only executed by one thread per warp.
    return block->allocate(num_objects);
  }

  // Return value indicates if block was emptied by this this request.
  template<class T>
  __DEV__ bool deallocate_in_block(uint32_t block_idx, uint32_t obj_id) {
    auto* block = get_block<T>(block_idx);
    return block->deallocate(obj_id);
  }

  template<class T>
  __DEV__ T* get_ptr_from_allocation(uint32_t block_idx, int rank,
                                     uint64_t allocation) {
    // Get index of rank-th first bit set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      allocation &= allocation - 1; 
    }

    int position = __ffsll(allocation);

    if (position > 0) {
      // Allocation successful.
      uintptr_t block_base = reinterpret_cast<uintptr_t>(get_block<T>(block_idx));
      return reinterpret_cast<T*>(block_base + position - 1);
    } else {
      return nullptr;
    }
  }

  template<class T>
  __DEV__ uint64_t invalidate_block(uint32_t block_idx) {
    return get_block<T>(block_idx)->invalidate();
  }

  template<class T>
  __DEV__ void uninvalidate_block(uint32_t block_idx, uint64_t previous_val) {
    get_block<T>(block_idx)->uninvalidate(previous_val);
  }

  static const uint8_t kObjectAddrBits = 6;
  static const uint32_t kNumBlockElements = 1ULL << kObjectAddrBits;
  static const uint64_t kObjectAddrBitmask = kNumBlockElements - 1;
  static const uint64_t kBlockAddrBitmask = ~kObjectAddrBitmask;
  static_assert(kNumBlockElements == 64,
                "Not implemented: Block size != 64.");
  static const int N = N_Objects / kNumBlockElements;

  static_assert(N_Objects % BlockSize == 0,
                "N_Objects Must be divisible by BlockSize.");

  using TupleType = std::tuple<Types...>;

  static const int kNumTypes = ...;

  static const int kBlockMaxSize = ...;

  char data_[N*kBlockMaxSize];

  Bitmap<uint32_t, N> global_free_;

  Bitmap<uint32_t, N> allocated_[kNumTypes];

  Bitmap<uint32_t, N> active_[kNumTypes];
};

#endif  // ALLOCATOR_SOA_ALLOCATOR_H
