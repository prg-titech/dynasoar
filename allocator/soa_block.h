#ifndef ALLOCATOR_SOA_BLOCK_H
#define ALLOCATOR_SOA_BLOCK_H

#include "allocator/configuration.h"
#include "allocator/soa_helper.h"
#include "allocator/util.h"


enum DeallocationState : int8_t {
  kBlockNowEmpty,      // Deallocate block.
  kBlockNowLeq50Full,  // Less/equal to 50% full.
  kBlockNowActive,     // Activate block.
  kRegularDealloc      // Nothing to do.
};


// A SOA block containing objects.
// T: Base type of the block.
// N_Max: Maximum number of objects per block (regardless of type). Currently
//        fixed at 64.
// N: Maximum number of objects in a block of type T.
template<class T, int TypeId, int N, int N_Max>
class SoaBlock {
 public:
  using BitmapT = unsigned long long int;

  // TODO: Should measure free level instead of fill level.
  static const int kLeq50Threshold = N / 2;

  static_assert(N_Max == 64, "Not implemented: Custom N_Max.");

  // Masks bits that are in use in this block. N_T bits set to 1.
  static const BitmapT kUsedBitsMask =
      N == N_Max ? (~0ULL) : ((1ULL << N) - 1);

  // Initializes a new block.
  __DEV__ SoaBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % N_Max == 0);   // Alignment.
    type_id = TypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    allocation_bitmap = static_cast<Bitmap>(0);
  }

  // Constructs an object identifier.
  __DEV__ T* make_pointer(uint8_t index) {
    uintptr_t ptr_as_int = index;
    uintptr_t block_size = N;
    ptr_as_int |= block_size << 48;
    uintptr_t type_id = TypeId;
    ptr_as_int |= type_id << 56;
    uintptr_t block_ptr = reinterpret_cast<uintptr_t>(this);
    assert(block_ptr < (1ULL << 49));   // Only 48 bits used in address space.
    assert((block_ptr & 0x3F) == 0);    // Block is aligned.
    ptr_as_int |= block_ptr;
    return reinterpret_cast<T*>(ptr_as_int);
  }

  // Initializes object iteration bitmap.
  __DEV__ void initialize_iteration() {
    iteration_bitmap = allocation_bitmap;
  }

  __DEV__ DeallocationState deallocate(int position) {
    BitmapT before;
    BitmapT selected_bit_mask = 1ULL << position;
    BitmapT mask = ~selected_bit_mask;

    do {
      // successful if: bit was "1" (allocated). Needed because we could be in
      // invalidation check. Retry otherwise.
      before = atomicAnd(&allocation_bitmap, mask);
    } while ((before & selected_bit_mask) == 0);

    int slots_allocated_before = __popcll(before);
    if (slots_allocated_before == N) {
      return kBlockNowActive;
    } else if (slots_allocated_before == 1) {
      return kBlockNowEmpty;
    } else if (slots_allocated_before == kLeq50Threshold + 1) {
      return kBlockNowLeq50Full;
    } else {
      return kRegularDealloc;
    }
  }

  __DEV__ int DBG_num_bits() {
    return N;
  }

  __DEV__ int DBG_allocated_bits() {
    return __popcll(allocation_bitmap);
  }

  __DEV__ bool is_slot_allocated(int index) {
    return allocation_bitmap & (1ULL << index);
  }

 private:
  template<uint32_t, class...> friend class SoaAllocator;

  // Dummy area that may be overridden by zero initialization.
  // Data section begins after kBlockDataSectionOffset bytes.
  // TODO: Do we need this on GPU?
  // TODO: Can this be replaced when using ROSE?
  char initialization_header_[kBlockDataSectionOffset - 3*sizeof(BitmapT)];

  // Bitmap of allocated slots.
  BitmapT allocation_bitmap;

  // A copy of allocation_bitmap. Set before the beginning of an iteration.
  BitmapT iteration_bitmap;

  // Padding to 8 bytes.
  uint8_t type_id;

  // Size of data segment.
  static const int kRawStorageBytes =
      SoaClassHelper<T>::template BlockConfig<N>::kDataSegmentSize;

  // Object size must be multiple of 64 bytes.
  static const int kStorageBytes = ((kRawStorageBytes + N_Max - 1) / N_Max) * N_Max;

  static_assert(N <= N_Max, "Assertion failed: N <= N_Max");

  // Data storage.
  char data_[kStorageBytes];
};

#endif  // ALLOCATOR_SOA_BLOCK_H
