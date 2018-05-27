// #define NDEBUG

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <assert.h>
#include <limits>

static const uint64_t kIdBits = 0x3F; // 0xFF
static const uint64_t kBaseBits = ~kIdBits;

// Size chosen such that 56 blocks of 8-byte objects can be stored.
// Maximum object size is around 512.
// A 4GB address space has 131072 superblocks; bitmap size 16KB.
static const uint32_t kSuperblockSize = 32768;

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


struct BlockHeader {
  __device__ static BlockHeader& from_block(uint64_t ptr) {
    assert(ptr & kIdBits == 0);
    return *reinterpret_cast<BlockHeader*>(ptr);
  }

  __device__ BlockHeader(uint32_t num_free)
      : free_counter(num_free),
        free_bitmap(num_free == 64 ? 0xFFFFFFFFFFFFFFFF
                                   : (1ULL << num_free) - 1) {
    assert(__popc(free_bitmap) == free_counter);
  }

  uint32_t free_counter;
  uint64_t free_bitmap;
};

template<class Self>
struct AosoaLayoutBase {
  // Type alias for atomic operations.
  using uint64_a_t = unsigned long long int;
  static_assert(sizeof(uint64_t) == sizeof(uint64_a_t), "Type size mismatch.");

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
    uintptr_t base_i = ptr & kBaseBits;
    uintptr_t id_i = ptr & kIdBits;
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

  // Calculate block size: Size of SOA arrays and header, aligned to `kIdBits`.
  // Note: This assumes that blocks are aligned properly within a superblock.
  static const uint32_t kBlockSize =
      ((kArraysSize + kHeaderSize + kIdBits - 1) / kIdBits) * kIdBits;

  __device__ static void initialize_block(uintptr_t ptr) {
    assert(ptr & kIdBits == 0);
    new(reinterpret_cast<void*>(ptr)) BlockHeader(Self::kSoaSize);
  }

  // Every thread must find a free slot in the block.
  // This function assumes that there is enough space in this block for every
  // allocating thread inthe warp.
  // This function updates the bit mask but not the counter.
  // Partly adapted from: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
  // Partly adapted from: https://devtalk.nvidia.com/default/topic/799429/cuda-programming-and-performance/possible-to-use-the-cuda-math-api-integer-intrinsics-to-find-the-nth-unset-bit-in-a-32-bit-int/
  __device__ static uintptr_t allocate_in_block(uintptr_t ptr,
                                                unsigned int leader) {
    assert(ptr & kIdBits == 0);
    unsigned int active = __activemask();
    // Use lane mask to empty all bits higher than the current thread.
    unsigned int rank = __popc(active & __lanemask_lt());
    // Allocation bits.
    uint64_t selected_bits = 0;

    if (leader == rank) {
      // This thread updates the bitmask.
      BlockHeader& header = BlockHeader::from_block(ptr);
      // Number of bits to allocate.
      int bits_left = __popc(active);

      do {
        uint64_t updated_mask = header.free_bitmap;
        uint64_t newly_selected_bits = 0;

        for (int i = 0; i < bits_left; ++i) {
          // TODO: To reduce collisions attempt to start allocation at
          // different positions (rotating shift).
          int next_bit_pos = __ffsll(updated_mask) - 1;
          assert(next_bit_pos >= 0);
          assert((1ULL << next_bit_pos) & updated_mask > 0);
          // Clear bit in updated mask.
          updated_mask &= updated_mask - 1;
          // Save location of selected bit.
          newly_selected_bits |= 1ULL << next_bit_pos;
        }

        assert(__popc(newly_selected_bits) == bits_left);
        // Count the number of bits that were selected but already set to false
        // by another thread.
        uint64_t successful_alloc = newly_selected_bits
            & atomicAnd(reinterpret_cast<uint64_a_t*>(&header.free_bitmap),
                        static_cast<uint64_a_t>(~newly_selected_bits));
        bits_left -= __popc(successful_alloc);
        selected_bits |= newly_selected_bits;
      } while (bits_left > 0);
    }

    __shfl_sync(active, selected_bits, leader);

    // Find the rank-th bit index that is set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      selected_bits &= selected_bits - 1; 
    }

    int position = __ffsll(selected_bits) - 1;
    assert(position >= 0);
    return ptr + position;
  }

  __device__ static void free(uintptr_t ptr) {
    uintptr_t base_i = ptr & kBaseBits;
    uintptr_t id_i = ptr & kIdBits;

    BlockHeader& header = BlockHeader::from_block(base_i);
    atomicXor(reinterpret_cast<uint64_a_t*>(&header.free_bitmap),
              1ULL << id_i);
    atomicAdd(&header.free_counter, 1);
  }
};

struct DummyClass : public AosoaLayoutBase<DummyClass> {
  using FieldsTuple = std::tuple<int, double, char, char>;
  static const uint32_t kSoaSize = 64;
};

__device__ char storage[10000000];

__global__ void kernel(uintptr_t* l) {
  uintptr_t block_loc = reinterpret_cast<uintptr_t>(storage);
  block_loc = ((block_loc + kIdBits - 1) / kIdBits) * kIdBits;

  auto x = DummyClass::allocate_in_block(block_loc, 0);
  *l = x;
}

int main() {
  kernel<<<1,1>>>(nullptr);
  gpuErrchk(cudaDeviceSynchronize());
}
