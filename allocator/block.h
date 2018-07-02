#ifndef ALLOCATOR_BLOCK_H
#define ALLOCATOR_BLOCK_H

#include <stdio.h>
#include <tuple>
#include <assert.h>
#include <limits>

static const uint8_t kIdBits = 6; // 8
static const uint32_t kMaxId = 1ULL << kIdBits;
static const uint64_t kIdBitmask = kMaxId - 1;
static const uint64_t kBaseBitmask = ~kIdBitmask;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__forceinline__ __device__ unsigned int __lanemask_lt() {
  unsigned int mask;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

// Helper structure used by try_allocate_in_block.
struct BlockAllocationResult {
  __device__ BlockAllocationResult(uint64_t allocation_mask_p,
                                   bool block_full_p)
      : allocation_mask(allocation_mask_p), block_full(block_full_p) {}

  uint64_t allocation_mask;

  // Set to true if this allocation request filled up the block entirely.
  bool block_full;
};

struct BlockHeader {
  __device__ static BlockHeader& from_block(uint64_t ptr) {
    assert((ptr & kIdBitmask) == 0);
    return *reinterpret_cast<BlockHeader*>(ptr);
  }

  __device__ BlockHeader(uint32_t num_free)
      : free_counter(num_free),
        free_bitmap(num_free == 64 ? 0xFFFFFFFFFFFFFFFF
                                   : (1ULL << num_free) - 1) {
    assert(__popcll(free_bitmap) == free_counter);
  }

  __device__ void print_debug() {
    printf("BlockHeader [%p]: %i free (%llx)\n",
           this, free_counter, free_bitmap);
  }

  // Number of free slots.
  uint32_t free_counter;

  // Position of this block if it is a free list.
  uint32_t list_position;

  // Bitmap of free slots.
  uint64_t free_bitmap;
};

template<class Self>
struct AosoaLayoutBase {
  // Type alias for atomic operations.
  using uint64_a_t = unsigned long long int;
  static_assert(sizeof(uint64_t) == sizeof(uint64_a_t), "Type size mismatch.");
  //static_assert(Self::kSoaSize <= (1 << kIdBits), "SOA size too large.");

  template<int FieldIndex, int _T = 0>
  struct FieldType {
    // Extract type from tuple.
    using type =
        typename std::tuple_element<FieldIndex,
                                    typename Self::FieldsTuple>::type;

    // Caluculate offset of SOA array.
    static const uint32_t kArrayOffset =
        FieldType<FieldIndex - 1>::kArraysSize;

    // Calculate size of object (this and all previous field types).
    static const uint32_t kObjectSize =
        FieldType<FieldIndex - 1>::kObjectSize + sizeof(type);

    // Calculate size of all SOA arrays, including this one.
    // Note: This value is intended for internal calculations only.
    static const uint32_t kArraysSizeUnaligned =
        kArrayOffset + sizeof(type) * Self::kSoaSize;

    // Calculate size of all SOA arrays, including proper alignment.
    static const uint32_t kArraysSize =
        ((kArraysSizeUnaligned + 8 - 1) / 8) * 8;
  };

  template<int _T>
  struct FieldType<0, _T> {
    using type =
        typename std::tuple_element<0, typename Self::FieldsTuple>::type;
    static const uint32_t kArrayOffset = 0;
    static const uint32_t kObjectSize = sizeof(type);
    static const uint32_t kArraysSizeUnaligned = sizeof(type) * Self::kSoaSize;
    static const uint32_t kArraysSize =
        ((kArraysSizeUnaligned + 8 - 1) / 8) * 8;
  };

  template<int FieldIndex>
  __device__ static typename FieldType<FieldIndex>::type& get(uintptr_t ptr) {
    uintptr_t base_i = ptr & kBaseBitmask;
    uintptr_t id_i = ptr & kIdBitmask;
    return *reinterpret_cast<typename FieldType<FieldIndex>::type*>(
        base_i
        + kHeaderSize
        + sizeof(FieldType<FieldIndex>::type)*id_i
        + FieldType<FieldIndex>::kArrayOffset);
  }

  // Header is used for "free" bitmap and counter.
  static const uint32_t kHeaderSize = sizeof(BlockHeader);
  // Size of one object (sum of all field type sizes).
  static const uint32_t kObjectSize =
      FieldType<std::tuple_size<typename Self::FieldsTuple>::value>
          ::kObjectSize;
  // Size of all SOA arrays of one block (taking into account alignment).
  static const uint32_t kArraysSize =
      FieldType<std::tuple_size<typename Self::FieldsTuple>::value>
          ::kArraysSize;

  // Calculate block size: Size of SOA arrays and header, aligned to `kMaxId`.
  // Note: This assumes that blocks are aligned properly within a superblock.
  static const uint32_t kBlockSize =
      ((kArraysSize + kHeaderSize + kMaxId - 1) / kMaxId) * kMaxId;

  __device__ static void initialize_block(uintptr_t ptr) {
    assert((ptr & kIdBitmask) == 0);
    new(reinterpret_cast<void*>(ptr)) BlockHeader(Self::kSoaSize);
  }

  // Returns a bit mask, indicating a set of allocations for the current warp.
  // This function updates the bit mask but not the counter.
  // Partly adapted from: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
  // Partly adapted from: https://devtalk.nvidia.com/default/topic/799429/cuda-programming-and-performance/possible-to-use-the-cuda-math-api-integer-intrinsics-to-find-the-nth-unset-bit-in-a-32-bit-int/
  // TODO: Maybe better to put this function in `BlockHeader`?
  __device__ static BlockAllocationResult try_allocate_in_block(uintptr_t ptr) {
    assert((ptr & kIdBitmask) == 0);

    // TODO: Maybe we should check here if there are still free slots?

    unsigned int active = __activemask();
    // Leader thread is the first thread whose mask bit is set to 1.
    int leader = __ffs(active) - 1;
    // Use lane mask to empty all bits higher than the current thread.
    // The rank of this thread is the number of bits set to 1 in the result.
    unsigned int rank = __popc(active & __lanemask_lt());
    // Allocation bits.
    uint64_t selected_bits = 0;
    // Set to true if this allocation filled up the block.
    bool block_full = false;

    if (leader == rank) {
      // This thread updates the bitmask.
      BlockHeader& header = BlockHeader::from_block(ptr);
      // Number of bits to allocate.
      int bits_left = __popc(active);
      // Helper variables used inside the loop and in the loop condition.
      uint64_t before_update, successful_alloc;

      do {
        // Bit set to 1 if slot is free.
        uint64_t updated_mask = header.free_bitmap;
        // If there are not enough free slots, allocate as many as possible and
        // the remaining threads return 0, indicated unsuccessful allocation.
        // In this case, these threads attempt allocation in a different block.
        int free_slots = __popcll(updated_mask);
        int allocation_size = min(free_slots, bits_left);
        uint64_t newly_selected_bits = 0;

        for (int i = 0; i < allocation_size; ++i) {
          // TODO: To reduce collisions attempt to start allocation at
          // different positions (rotating shift).
          int next_bit_pos = __ffsll(updated_mask) - 1;
          assert(next_bit_pos >= 0);
          assert(((1ULL << next_bit_pos) & updated_mask) > 0);
          // Clear bit in updated mask.
          updated_mask &= updated_mask - 1;
          // Save location of selected bit.
          newly_selected_bits |= 1ULL << next_bit_pos;
        }

        assert(__popcll(newly_selected_bits) == allocation_size);
        // Count the number of bits that were selected but already set to false
        // by another thread.
        before_update =
            atomicAnd(reinterpret_cast<uint64_a_t*>(&header.free_bitmap),
                      static_cast<uint64_a_t>(~newly_selected_bits));
        successful_alloc = newly_selected_bits & before_update;
        bits_left -= __popcll(successful_alloc);

        selected_bits |= successful_alloc;

        // Block full if at least one slot was allocated and "before update"
        // bit-and "now allocated" indicates that block is full.
        block_full = successful_alloc > 0
                     && (before_update & ~successful_alloc) == 0;

        // Stop loop if no more free bits available in this block or all
        // requested allocations completed successfully.
      } while (bits_left > 0 && __popcll(before_update
                                         & ~successful_alloc) > 0);
    }

    selected_bits = __shfl_sync(active, selected_bits, leader);

    // At most one thread should indicate that the block filled up.
    return BlockAllocationResult(selected_bits, block_full && rank == leader);
  }

  // TODO: This could also be aggregated per warp.
  __device__ static void free(uintptr_t ptr) {
    uintptr_t base_i = ptr & kBaseBitmask;
    uintptr_t id_i = ptr & kIdBitmask;

    BlockHeader& header = BlockHeader::from_block(base_i);
    atomicXor(reinterpret_cast<uint64_a_t*>(&header.free_bitmap),
              1ULL << id_i);
    atomicAdd(&header.free_counter, 1);
  }
};

#endif  // ALLOCATOR_BLOCK_H
