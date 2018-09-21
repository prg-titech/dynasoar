#ifndef BITMAP_BITMAP_H
#define BITMAP_BITMAP_H

#include <assert.h>
#include <limits>
#include <stdint.h>

#include "bitmap/util.h"

#ifdef SCAN_CUB
#include <cub/cub.cuh>
#endif  // SCAN_CUB

#ifndef NDEBUG
static const int kMaxRetry = 10000000;
#define CONTINUE_RETRY (_retry_counter++ < kMaxRetry)
#define INIT_RETRY int _retry_counter = 0;
#else
#define CONTINUE_RETRY (true)
#define INIT_RETRY
#endif  // NDEBUG


// TODO: Only works with ContainerT = unsigned long long int.
template<typename SizeT, SizeT N, typename ContainerT = unsigned long long int>
class Bitmap {
 public:
  static const SizeT kIndexError = std::numeric_limits<SizeT>::max();
  static const ContainerT kZero = 0;
  static const ContainerT kOne = 1;

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
    ContainerT pos_mask = kOne << offset;
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
    ContainerT pos_mask = kOne << offset;
    ContainerT previous;
    bool success;

    INIT_RETRY;

    do {
      previous = atomicAnd(data_.containers + container, ~pos_mask);
      success = (previous & pos_mask) != 0;
    } while (Retry && !success && CONTINUE_RETRY);    // Retry until success

    if (kHasNested && success && __popcll(previous) == 1) {
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

  // Initialize bitmap to all 0 or all 1.
  __DEV__ void initialize(bool allocated = false) {
    for (SizeT i = blockIdx.x*blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x*gridDim.x) {
      if (allocated) {
        if (i == kNumContainers - 1 && N % kBitsize > 0) {
          // Last container is only partially occupied.
          data_.containers[i] = (kOne << (N % kBitsize)) - 1;
        } else if (i < kNumContainers) {
          data_.containers[i] = ~kZero;
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
    return data_.containers[index/kBitsize] & (kOne << (index % kBitsize));
  }

  // Initiate scan operation (from the host side). This request is forwarded
  // to the next-level bitmap. Afterwards, scan continues here.
  void scan() { data_.scan(); }

  // May only be called after scan.
  __DEV__ SizeT scan_num_bits() const {
    return data_.enumeration_result_size;
  }

  __DEV__ SizeT scan_get_index(SizeT pos) const {
    return data_.enumeration_result_buffer[pos];
  }

  // Nested bitmap data structure.
  template<SizeT NumContainers, bool HasNested>
  struct BitmapData;

  // Bitmap data structure with a nested bitmap.
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, true> {
    using ThisClass = BitmapData<NumContainers, true>;

    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    ContainerT containers[NumContainers];

    // Buffers for parallel enumeration (prefix sum).
    // TODO: These buffers can be shared among all types.
#ifdef CUB_SCAN
    // TODO: We probably do not need all these buffers.
    SizeT enumeration_base_buffer[NumContainers*kBitsize];
    SizeT enumeration_id_buffer[NumContainers*kBitsize];
    SizeT enumeration_cub_temp[3*NumContainers*kBitsize];
    SizeT enumeration_cub_output[NumContainers*kBitsize];
#endif  // CUB_SCAN

    SizeT enumeration_result_buffer[NumContainers*kBitsize];
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

    void scan() {
#ifdef SCAN_CUB
      run_cub_scan();
#else
      run_atomic_add_scan();
#endif  // CUB_SCAN
    }

#ifdef SCAN_CUB
#include "bitmap/scan_cub.inc"
#else
#include "bitmap/scan_atomic.inc"
#endif  // CUB_SCAN
  };

  // Bitmap data structure without a nested bitmap.
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, false> {
    static_assert(NumContainers == 1,
                  "L0 bitmap should have only one container.");

    using ThisClass = BitmapData<NumContainers, false>;

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
        if (containers[0] & (kOne << i)) {
          enumeration_result_buffer[current_size++] = i;
        }
      }

      enumeration_result_size = current_size;
      assert(__popcll(containers[0]) == current_size);
    }

    void scan() {
      // Does not perform a prefix scan but computes the result directly.
      member_func_kernel<ThisClass, &ThisClass::trivial_scan><<<1, 1>>>(this);
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

  // The number of bits per container.
  static const uint8_t kBitsize = 8*sizeof(ContainerT);

  // The number of containers on this level.
  static const SizeT kNumContainers = N <= kBitsize ? 1 : N / kBitsize;

  // Indicates if this bitmap has a higher-level (nested) bitmap.
  static const bool kHasNested = kNumContainers > 1;

  static_assert(!kHasNested || N % kBitsize == 0,
                "N must be of size (sizeof(ContainerT)*8)**D * x.");

  // Nested bitmap structure.
  BitmapData<kNumContainers, kHasNested> data_;
};

#endif  // BITMAP_BITMAP_H
