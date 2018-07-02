#ifndef ALLOCATOR_BLOCK_MANAGER_H
#define ALLOCATOR_BLOCK_MANAGER_H

#include "allocator/block.h"
#include "allocator/storage.h"

template<uint32_t Size, bool HasNested = (Size > 64)>
class BlockBitmap {
  // If Size > 64: Nest another BlockBitmap.
};

// Specialization without nested bitmap.  
template<uint32_t Size>
class BlockBitmap<Size, false>
{
 private:
  uint64_t bitmap_;

  constexpr int nesting_level() {
    return nesting_level(64, 0);
  }

  constexpr int nesting_level(uint32_t size, int accumulator) {
    return size >= Size
        ? accumulator : nesting_level(64*size, accumulator + 1);
  }

 public:

}

template<typename T>
class BlockManager {
 private:
  // Maximum number of blocks that can be allocated in the storage.
  static const int kMaxNumBlocks = (kStorageSize + T::kBlockSize - 1)
      / T::kBlockSize;

  // New objects will always be allocated in an active block. If an active
  // block becomes full, it will be added to the "full blocks" list and a
  // block from the "free blocks" list will take its spot.
  BlockHeader* active_blocks[kNumActiveBlocks];

  __device__ void remove_full_block(BlockHeader* ptr) {
    uint32_t l1 = acquire_block(ptr);
    uint32_t r_index = atomicSub(&num_full_block, 1) - 1;

    if (r_index > 0) {
      // This was not the last block that we removed. We can use the preceding
      // block as a replacement block.
      BlockHeader* replacement = full_list[r_index - 1];
      uint32_t l2 = acquire_block(replacement);

      // PROBLEM: Other threads may add and remove at the same time, so
      // multiple threads may try to write into position l1.
      release_block(replacement, l1);
    }
  }

 public:
  uintptr_t allocate() {
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int active_index = warp_id % kNumActiveBlocks;

    // Number of allocations.
    unsigned int alloc_left = __popc(__activemask());

    while (true) {
      // Let loop run until memory was allocated.

      // Rank within this warp (counting only active threads).
      unsigned int rank = __popc(active & __lanemask_lt());

      // Find active block to allocate in.
      // TODO: Maybe only the leader thread within this warp should do this?
      BlockHeader* block = nullptr;
      while (block == nullptr) {
        block = active_blocks[active_index];

        if (block == nullptr) {
          // Block is currently being replaced with a new free block.
          active_index = (active_index + 1) % warp_id;
        }
      }

      // TODO: Replace with new free block if all slots in use now.
      auto allocation = T::try_allocate_in_block(block);
      assert(__popcll(allocation.allocation_mask) <= __popc(__activemask));

      if (allocation.block_full) {
        // try_allocate_in_block gurantees that flag is only set in one thread.
        // Mark as "currently under update". Does not have to be done
        // atomically, because only thread can signal "block full".
        active_blocks[active_index] = 0;
        // Set new block with empty slots.
        active_blocks[active_blocks] = request_free_block();
      }

      if (__popcll(allocation.allocation_mask) > rank) {
        // Memory was allocated for this thread. Find the rank-th bit index
        // that is set to 1.
        for (int i = 0; i < rank; ++i) {
          // Clear last bit.
          allocation.allocation_mask &= allocation.allocation_mask - 1; 
        }

        int position = __ffsll(allocation.allocation_mask) - 1;
        assert(position >= 0);
        return reinterpret_cast<uintptr_t>(block) + position;
      }
    }
  }
};

#endif  // ALLOCATOR_BLOCK_MANAGER_H