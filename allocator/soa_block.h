#ifndef ALLOCATOR_SOA_BLOCK_H
#define ALLOCATOR_SOA_BLOCK_H

#include "allocator/configuration.h"
#include "allocator/soa_helper.h"
#include "allocator/util.h"


enum DeallocationState : int8_t {
  kBlockNowEmpty,      // Deallocate block.
#ifdef OPTION_DEFRAG
  kBlockNowLeq50Full,  // Less/equal to 50% full.
#endif  // OPTION_DEFRAG
  kBlockNowActive,     // Activate block.
  kRegularDealloc      // Nothing to do.
};


// TODO: Fix visibility.
// A SOA block containing objects.
// T: Base type of the block.
// N: Maximum number of objects in a block of type T.
template<class T, TypeIndexT TypeId, int N>
class SoaBlock {
 public:
  using BitmapT = unsigned long long int;

#ifdef OPTION_DEFRAG
  // This is the number of allocated objects.
  static const int kLeq50Threshold =
      1.0f*kDefragFactor / (kDefragFactor + 1) * N;
#endif  // OPTION_DEFRAG

  // Bitmap initializer: N_T bits set to 1.
  static const BitmapT kBitmapInitState =
      N == 64 ? (~0ULL) : ((1ULL << N) - 1);

  // Initializes a new block.
  __DEV__ SoaBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % 64 == 0);   // Alignment.
    type_id = TypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
  }

  // Constructs an object identifier.
  // TODO: Signed to unsigned cast here.
  __DEV__ T* make_pointer(int index) {
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

  __DEV__ BitmapT allocation_bitmap() const {
    return ~free_bitmap & kBitmapInitState;
  }

  // Initializes object iteration bitmap.
  __DEV__ void initialize_iteration() {
    iteration_bitmap = allocation_bitmap();
  }

  __DEV__ DeallocationState deallocate(int position) {
    BitmapT before;
    BitmapT mask = 1ULL << position;

    do {
      // successful if: bit was "0" (allocated). Needed because we could be in
      // invalidation check.
      before = atomicOr(&free_bitmap, mask);
    } while ((before & mask) != 0);

    auto slots_free_before = __popcll(before);
    if (slots_free_before == 0) {
      return kBlockNowActive;
    } else if (slots_free_before == N - 1) {
      return kBlockNowEmpty;
#ifdef OPTION_DEFRAG
    } else if (slots_free_before == N - kLeq50Threshold - 1) {
      return kBlockNowLeq50Full;
#endif  // OPTION_DEFRAG
    } else {
      return kRegularDealloc;
    }
  }

  __DEV__ int DBG_num_bits() {
    return N;
  }

  __DEV__ int DBG_allocated_bits() {
    return N - __popcll(free_bitmap);
  }

  __DEV__ bool is_slot_allocated(int index) {
    return (free_bitmap & (1ULL << index)) == 0;
  }

  template<BlockIndexT, class...> friend class SoaAllocator;

  // Dummy area that may be overridden by zero initialization.
  // Data section begins after kBlockDataSectionOffset bytes.
  // TODO: Do we need this on GPU?
  // TODO: Can this be replaced when using ROSE?
  char initialization_header_[kBlockDataSectionOffset - 3*sizeof(BitmapT)];

  // Bitmap of free slots.
  BitmapT free_bitmap;

  // A copy of ~free_bitmap. Set before the beginning of an iteration. Does
  // not contain dirty objects.
  BitmapT iteration_bitmap;

  // Padding to 8 bytes.
  volatile TypeIndexT type_id;

  // Size of data segment.
  static const int kStorageBytes =
      SoaClassHelper<T>::template BlockConfig<N>::kDataSegmentSize;

  static_assert(N <= 64, "Assertion failed: N <= 64");

  // Data storage.
  char data_[kStorageBytes];
};

#endif  // ALLOCATOR_SOA_BLOCK_H
