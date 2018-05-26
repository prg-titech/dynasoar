#include <iostream>
#include <stdio.h>
#include <tuple>
#include <assert.h>
#include <limits>

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
  static BlockHeader& from_block(uint64_t ptr) {
    assert(ptr & 0xFF == 0);
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
  // Header is used for "free" bitmap and counter.
  static const uint32_t kHeaderSize = 16;
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
    static const uint32_t kArrayOffsetUnaligned =
        FieldType<FieldIndex - 1>::kArrayOffset
        + sizeof(FieldType<FieldIndex - 1>::type) * Self::kSoaSize;

    // Round to nearest multiple of 8.
    static const uint32_t kArrayOffset =
        ((kArrayOffsetUnaligned + 8 - 1) / 8) * 8;
  };

  template<int _T>
  struct FieldType<0, _T> {
    using type =
        typename std::tuple_element<0, typename Self::FieldsTuple>::type;
    static const uint32_t kArrayOffsetUnaligned = 0;
    static const uint32_t kArrayOffset = 0;
  };

  template<int FieldIndex>
  __device__ static typename FieldType<FieldIndex>::type& get(uintptr_t ptr) {
    uintptr_t base_i = ptr & 0xFFFFFFFFFFFFFF00;
    uintptr_t id_i = ptr & 0xFF;
    return *reinterpret_cast<typename FieldType<FieldIndex>::type*>(
        base_i
        + kHeaderSize
        + sizeof(FieldType<FieldIndex>::type)*id_i
        + FieldType<FieldIndex>::kArrayOffset);
  }

  __device__ static void initialize_block(uintptr_t ptr) {
    assert(reinterpret_cast<uintptr_t>(ptr) & 0xFF == 0);
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
          updated_mask ^= 1ULL << next_bit_pos;
          newly_selected_bits |= 1ULL << next_bit_pos;
        }

        assert(__popc(newly_selected_bits) == bits_left);
        // Count the number of bits that were selected but already set to true
        // by another thread.
        uint64_t collisions = newly_selected_bits
            & atomicOr(reinterpret_cast<uint64_a_t*>(&header.free_bitmap),
            static_cast<uint64_a_t>(newly_selected_bits));
        bits_left = __popc(collisions);
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
};

struct DummyClass : public AosoaLayoutBase<DummyClass> {
  using FieldsTuple = std::tuple<int, double, char, char>;
  static const uint32_t kSoaSize = 64;
};

__device__ char storage[1000000000];

int main() {
  //kernel<<<1,1>>>(nullptr, nullptr);  
  gpuErrchk(cudaDeviceSynchronize());
}
