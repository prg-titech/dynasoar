#ifndef BITMAP_BITMAP_H
#define BITMAP_BITMAP_H

#include <assert.h>
#include <limits>
#include <stdint.h>

//#include <thrust/scan.h>
#include <cub/cub.cuh>

#define __DEV__ __device__

#ifndef NDEBUG
static const int kMaxRetry = 10000000;
#define CONTINUE_RETRY (_retry_counter++ < kMaxRetry)
#define INIT_RETRY int _retry_counter = 0;
#else
#define CONTINUE_RETRY (true)
#define INIT_RETRY
#endif  // NDEBUG

// Shift left, rotating.
// Copied from: https://gist.github.com/pabigot/7550454
template <typename T>
__DEV__ T rotl (T v, unsigned int b)
{
  static_assert(std::is_integral<T>::value, "rotate of non-integral type");
  static_assert(! std::is_signed<T>::value, "rotate of signed type");
  constexpr unsigned int num_bits {std::numeric_limits<T>::digits};
  static_assert(0 == (num_bits & (num_bits - 1)), "rotate value bit length not power of two");
  constexpr unsigned int count_mask {num_bits - 1};
  const unsigned int mb {b & count_mask};
  using promoted_type = typename std::common_type<int, T>::type;
  using unsigned_promoted_type = typename std::make_unsigned<promoted_type>::type;
  return ((unsigned_promoted_type{v} << mb)
          | (unsigned_promoted_type{v} >> (-mb & count_mask)));
}

// Seems like this is a scheduler warp ID and may change.
__forceinline__ __device__ unsigned warp_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// Compute scan result for a L0 bitmap with a single container.
template<typename T>
__global__ void kernel_trivial_scan(T* ptr) {
  assert(blockDim.x*gridDim.x == 1);
  ptr->trivial_scan();
}

template<typename T>
__global__ void kernel_pre_scan(T* ptr) {
  ptr->pre_scan();
}

template<typename T>
__global__ void kernel_post_scan(T* ptr) {
  ptr->post_scan();
}

template<typename T>
__global__ void kernel_set_result_size(T* ptr) {
  assert(blockDim.x*gridDim.x == 1);
  ptr->set_result_size();
}

template<typename T>
__global__ void kernel_atomic_add_scan_init(T* ptr) {
  assert(blockDim.x*gridDim.x == 1);
  ptr->atomic_add_scan_init();
}

template<typename T>
__global__ void kernel_atomic_add_scan(T* ptr) {
  ptr->atomic_add_scan();
}

template<typename T>
T read_from_device(T* ptr) {
  T host_storage;
  cudaMemcpy(&host_storage, ptr, sizeof(T), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
  return host_storage;
}


// TODO: Only works with unsigned ContainerT types.
template<typename SizeT, SizeT N, typename ContainerT = unsigned long long int>
class Bitmap {
 public:
  static const SizeT kIndexError = std::numeric_limits<SizeT>::max();

  __DEV__ Bitmap() {}

  // Delete copy constructor.
  __DEV__ Bitmap(const Bitmap&) = delete;

  // Allocate specific position, i.e., set bit to 1. Return value indicates
  // success. If Retry, then continue retrying until successful update.
  template<bool Retry = false>
  __DEV__ bool allocate(SizeT pos) {
    assert(pos != kIndexError);
    assert(pos < N);
    SizeT container = pos / kBitsize;
    SizeT offset = pos % kBitsize;

    // Set bit to one.
    ContainerT pos_mask = static_cast<ContainerT>(1) << offset;
    ContainerT previous;
    bool success;

    INIT_RETRY;

    do {
      previous = atomicOr(data_.containers + container, pos_mask);
      success = (previous & pos_mask) == 0;
    } while (Retry && !success && CONTINUE_RETRY);    // Retry until success

    if (kHasNested && success && previous == 0) {
      // Allocated first bit, propagate to nested.
      bool success2 = data_.nested_allocate<true>(container);
      assert(success2);
    }

    return success;
  }

  // Return the index of an allocated bit, utilizing the hierarchical bitmap
  // structure.
  template<bool Retry = false>
  __DEV__ SizeT find_allocated(int seed) const {
    SizeT index;

    INIT_RETRY;

    do {
      index = find_allocated_private(seed);
    } while (Retry && index == kIndexError && CONTINUE_RETRY);

    assert(!Retry || index != kIndexError);
    assert(index == kIndexError || index < N);
    return index;
  }

  // Deallocate arbitrary bit and return its index. Assuming that there is
  // at least one remaining allocation. Retries until success.
  __DEV__ SizeT deallocate() {
    SizeT index;
    bool success;

    INIT_RETRY;

    int retries = 0;
    do {
      index = find_allocated<false>(retries++);
      if (index == kIndexError) {
        success = false;
      } else {
        success = deallocate<false>(index); // if false: other thread was faster
      }
    } while (!success && CONTINUE_RETRY);

    assert(success);

    return index;
  }

  // Deallocate specific position, i.e., set bit to 0. Return value indicates
  // success. If Retry, then continue retrying until successful update.
  template<bool Retry = false>
  __DEV__ bool deallocate(SizeT pos) {
    assert(pos != kIndexError);
    assert(pos < N);
    SizeT container = pos / kBitsize;
    SizeT offset = pos % kBitsize;

    // Set bit to one.
    ContainerT pos_mask = static_cast<ContainerT>(1) << offset;
    ContainerT previous;
    bool success;

    INIT_RETRY;

    do {
      previous = atomicAnd(data_.containers + container, ~pos_mask);
      success = (previous & pos_mask) != 0;
    } while (Retry && !success && CONTINUE_RETRY);    // Retry until success

    if (kHasNested && success && count_bits(previous) == 1) {
      // Deallocated only bit, propagate to nested.
      bool success2 = data_.nested_deallocate<true>(container);
      assert(success2);
    }

    return success;
  }

  // Return the index of an arbitrary, allocated bit. This algorithm forwards
  // the request to the top-most (deepest nested) bitmap and returns the index
  // of the container in which to search. On the lowest level, the container ID
  // equals the bit index.
  // TODO: Sould be private.
  __DEV__ SizeT find_allocated_private(int seed) const {
    SizeT container;

    if (kHasNested) {
      container = data_.nested_find_allocated_private(seed);

      if (container == kIndexError) {
        return kIndexError;
      }
    } else {
      container = 0;
    }

    return find_allocated_in_container(container, seed);
  }

  __DEV__ void initialize(bool allocated = false) {
    for (SizeT i = blockIdx.x*blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x*gridDim.x) {
      if (allocated) {
        if (i == kNumContainers - 1 && N % kBitsize > 0) {
          // Last container is only partially occupied.
          data_.containers[i] =
              (static_cast<ContainerT>(1) << (N % kBitsize)) - 1;
        } else if (i < kNumContainers) {
          data_.containers[i] = ~static_cast<ContainerT>(0);
        }
      } else {
        data_.containers[i] = 0;
      }
    }

    if (kHasNested) {
      data_.nested_initialize(allocated);
    }
  }

  // Return true if index is allocated.
  __DEV__ bool operator[](SizeT index) const {
    return data_.containers[index/kBitsize]
        & (static_cast<ContainerT>(1) << (index % kBitsize));
  }

  // Initiate scan operation (from the host side). This request is forwarded
  // to the next-level bitmap. Afterwards, scan continues here.
  void scan() {
    data_.scan();
  }

  // May only be called after scan.
  __DEV__ SizeT scan_num_bits() const {
    return data_.enumeration_result_size;
  }

  __DEV__ SizeT scan_get_index(SizeT pos) const {
    return data_.enumeration_result_buffer[pos];
  }

  // TODO: Should be private.
 public:
  template<SizeT NumContainers, bool HasNested>
  struct BitmapData;

  template<SizeT NumContainers>
  struct BitmapData<NumContainers, true> {
    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    ContainerT containers[NumContainers];

    // Buffer for parallel enumeration (prefix sum).
    // TODO: These buffers can be shared among all types.
    // TODO: We probably do not need all these buffers.
    SizeT enumeration_base_buffer[NumContainers*kBitsize];
    SizeT enumeration_id_buffer[NumContainers*kBitsize];
    SizeT enumeration_result_buffer[NumContainers*kBitsize];
    SizeT enumeration_cub_temp[3*NumContainers*kBitsize];
    SizeT enumeration_cub_output[NumContainers*kBitsize];
    SizeT enumeration_result_size;

    Bitmap<SizeT, NumContainers, ContainerT> nested;

    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) {
      return nested.allocate<Retry>(pos);
    }

    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) {
      return nested.deallocate<Retry>(pos);
    }

    __DEV__ SizeT nested_find_allocated_private(int seed) const {
      return nested.find_allocated_private(seed);
    }

    __DEV__ void nested_initialize(bool allocated) {
      nested.initialize(allocated);
    }

    __DEV__ void atomic_add_scan_init() {
      enumeration_result_size = 0;
    }

    __DEV__ void atomic_add_scan() {
      SizeT* selected = nested.data_.enumeration_result_buffer;
      SizeT num_selected = nested.data_.enumeration_result_size;
      //printf("num_selected=%i\n", (int) num_selected);

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        int num_bits = __popcll(value);

        auto before = atomicAdd(reinterpret_cast<unsigned int*>(&enumeration_result_size),
                                num_bits);

        for (int i = 0; i < num_bits; ++i) {
          int next_bit = __ffsll(value) - 1;
          assert(next_bit >= 0);
          enumeration_result_buffer[before+i] = container_id*kBitsize + next_bit;
          //Advance to next bit.
          value &= value - 1;
        }
      }      
    }

    // TODO: Run with num_selected threads, then we can remove the loop.
    __DEV__ void pre_scan() {
      SizeT* selected = nested.data_.enumeration_result_buffer;
      SizeT num_selected = nested.data_.enumeration_result_size;

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        for (int i = 0; i < kBitsize; ++i) {
          // Write "1" if allocated, "0" otherwise.
          bool bit_selected = value & (static_cast<ContainerT>(1) << i);
          enumeration_base_buffer[sid*kBitsize + i] = bit_selected;
          if (bit_selected) {
            enumeration_id_buffer[sid*kBitsize + i] =
                kBitsize*container_id + i;
          }
        }
      }
    }

    // Run with num_selected threads.
    // Assumption: enumeration_base_buffer contains exclusive prefix sum.
    __DEV__ void post_scan() {
      SizeT* selected = nested.data_.enumeration_result_buffer;
      SizeT num_selected = nested.data_.enumeration_result_size;

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        for (int i = 0; i < kBitsize; ++i) {
          // Write "1" if allocated, "0" otherwise.
          bool bit_selected = value & (static_cast<ContainerT>(1) << i);
          if (bit_selected) {
            // Minus one because scan operation is inclusive.
            enumeration_result_buffer[enumeration_cub_output[sid*kBitsize + i] - 1] =
                enumeration_id_buffer[sid*kBitsize + i];
          }
        }
      }
    }

    __DEV__ void set_result_size() {
      SizeT num_selected = nested.data_.enumeration_result_size;
      // Base buffer contains prefix sum.
      SizeT result_size = enumeration_cub_output[num_selected*kBitsize - 1];
      enumeration_result_size = result_size;
    }

    void scan() {
      run_atomic_add_scan();
      // Performance evaluation...
      //run_cub_scan();
    }

    void run_atomic_add_scan() {
      nested.scan();

      SizeT num_selected = read_from_device<SizeT>(&nested.data_.enumeration_result_size);
      kernel_atomic_add_scan_init<<<1, 1>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
      kernel_atomic_add_scan<<<num_selected/256+1, 256>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
    }

    void run_cub_scan() {  
      nested.scan();

      SizeT num_selected = read_from_device<SizeT>(&nested.data_.enumeration_result_size);
      kernel_pre_scan<<<num_selected/256+1, 256>>>(this);
      gpuErrchk(cudaDeviceSynchronize());

      size_t temp_size = 3*NumContainers*kBitsize;
      cub::DeviceScan::InclusiveSum(enumeration_cub_temp,
                                    temp_size,
                                    enumeration_base_buffer,
                                    enumeration_cub_output,
                                    num_selected*kBitsize);
      kernel_post_scan<<<num_selected/256+1, 256>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
      kernel_set_result_size<<<1, 1>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
    }
  };

  template<SizeT NumContainers>
  struct BitmapData<NumContainers, false> {
    static_assert(NumContainers == 1,
                  "L0 bitmap should have only one container.");

    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    ContainerT containers[NumContainers];

    SizeT enumeration_result_buffer[NumContainers*kBitsize];
    SizeT enumeration_result_size;

    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) { assert(false); return false; }

    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) { assert(false); return false; }

    __DEV__ SizeT nested_find_allocated_private(int seed) const {
      assert(false);
      return kIndexError;
    }

    __DEV__ void nested_initialize(bool allocated) { assert(false); }

    __DEV__ void trivial_scan() {
      SizeT current_size = 0;
      for (int i = 0; i < kBitsize; ++i) {
        if (containers[0] & (static_cast<ContainerT>(1) << i)) {
          enumeration_result_buffer[current_size++] = i;
        }
      }

      enumeration_result_size = current_size;
      assert(__popcll(containers[0]) == current_size);
    }

    void scan() {
      // Does not perform a prefix scan but computes the result directly.
      kernel_trivial_scan<<<1, 1>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
    }
  };

  // Returns the index of an allocated bit inside a container. Returns
  // kIndexError if not allocated bit was found.
  __DEV__ SizeT find_allocated_in_container(SizeT container, int seed) const {
    // TODO: For better performance, choose random one.
    int selected = find_allocated_bit(data_.containers[container], seed);
    if (selected == -1) {
      // No space in here.
      return kIndexError;
    } else {
      return selected + container*kBitsize;
    }
  }

  __DEV__ int find_first_bit(ContainerT val) const {
    // TODO: Adapt for other data types.
    return __ffsll(val);
  }

  __DEV__ int find_allocated_bit(ContainerT val, int seed) const {
    return find_allocated_bit_fast(val, seed);
  }

  // Find index of *some* bit that is set to 1.
  // TODO: Make this more efficient!
  __DEV__ int find_allocated_bit_fast(ContainerT val, int seed) const {
    unsigned int rotation_len = (seed+warp_id()) % (sizeof(val)*8);
    const ContainerT rotated_val = rotl(val, rotation_len);

    int first_bit_pos = __ffsll(rotated_val) - 1;
    if (first_bit_pos == -1) {
      return -1;
    } else {
      return (first_bit_pos - rotation_len) % (sizeof(val)*8);
    }
  }

  __DEV__ int find_allocated_bit_compact(ContainerT val) const {
    return __ffsll(val) - 1;
  }

  __DEV__ int count_bits(ContainerT val) const {
    // TODO: Adapt for other data types.
    return __popcll(val);
  }

  static const uint8_t kBitsize = 8*sizeof(ContainerT);

  static const SizeT kNumContainers = N <= 64 ? 1 : N / kBitsize;

  static const bool kHasNested = kNumContainers > 1;

  static_assert(!kHasNested || N % kBitsize == 0,
                "N must be of size (sizeof(ContainerT)*8)**D * x.");

  BitmapData<kNumContainers, kHasNested> data_;
};

#endif  // BITMAP_BITMAP_H
