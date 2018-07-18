#ifndef BITMAP_BITMAP_H
#define BITMAP_BITMAP_H

#include <assert.h>
#include <limits>
#include <stdint.h>

#define __DEV__ __device__

// TODO: Remove this once code work properly.
static const int kMaxRetry = 10000000;
#define CONTINUE_RETRY (_retry_counter++ < kMaxRetry)
#define INIT_RETRY int _retry_counter = 0;

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

// Problem: Deadlock if two threads in the same warp want to update the same
// value. E.g., t0 wants to write "1" and waits for "0" to appear. But t1 cannot
// write "0" because of thread divergence.

// Sol. 1: Only one thread per warp is allowed to update values.

// Bitmap mode: Set nested (parent) bitmap bit to 1 if there is at least one
// bit set to 1 in the current bitmap. Allows for efficient deallocate(), but
// not allocate().
// TODO: ContainerT must be unsigned (?). Or maybe signed is also OK.
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
  __DEV__ SizeT find_allocated() const {
    SizeT index;

    INIT_RETRY;

    do {
      index = find_allocated_private();
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

    do {
      index = find_allocated<true>();
      success = deallocate<false>(index); // if false: other thread was faster
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
  __DEV__ SizeT find_allocated_private() const {
    SizeT container;

    if (kHasNested) {
      container = data_.nested_find_allocated_private();

      if (container == kIndexError) {
        return kIndexError;
      }
    } else {
      container = 0;
    }

    return find_allocated_in_container(container);
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

 private:
  template<SizeT NumContainers, bool HasNested>
  struct BitmapData;

  template<SizeT NumContainers>
  struct BitmapData<NumContainers, true> {
    ContainerT containers[NumContainers];

    Bitmap<SizeT, NumContainers, ContainerT> nested;

    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) {
      return nested.allocate<Retry>(pos);
    }

    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) {
      return nested.deallocate<Retry>(pos);
    }

    __DEV__ SizeT nested_find_allocated_private() const {
      return nested.find_allocated_private();
    }

    __DEV__ void nested_initialize(bool allocated) {
      nested.initialize(allocated);
    }
  };

  template<SizeT NumContainers>
  struct BitmapData<NumContainers, false> {
    ContainerT containers[NumContainers];

    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) { assert(false); return false; }

    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) { assert(false); return false; }

    __DEV__ SizeT nested_find_allocated_private() const {
      assert(false);
      return kIndexError;
    }

    __DEV__ void nested_initialize(bool allocated) { assert(false); }
  };

  // Returns the index of an allocated bit inside a container. Returns
  // kIndexError if not allocated bit was found.
  __DEV__ SizeT find_allocated_in_container(SizeT container) const {
    // TODO: For better performance, choose random one.
    int selected = find_allocated_bit(data_.containers[container]);
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

  __DEV__ int find_allocated_bit(ContainerT val) const {
    return find_allocated_bit_fast(val);
  }

  // Find index of *some* bit that is set to 1.
  // TODO: Make this more efficient!
  __DEV__ int find_allocated_bit_fast(ContainerT val) const {
    unsigned int rotation_len = threadIdx.x % (sizeof(val)*8);
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
