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


class AbstractBlock {
 public:
  using BitmapT = unsigned long long int;

  __DEV__ AbstractBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % 64 == 0);   // Alignment.
  }

  __DEV__ bool is_slot_allocated(ObjectIndexT index) {
    return (free_bitmap & (1ULL << index)) == 0;
  }

  // Dummy area that may be overwritten by zero initialization.
  // Data section begins after kBlockDataSectionOffset bytes.
  // TODO: Do we need this on GPU? Can this be replaced when using ROSE?
  char initialization_header_[kBlockDataSectionOffset - 3*sizeof(BitmapT)];

  // Bitmap of free slots.
  BitmapT free_bitmap;

  // A copy of ~free_bitmap. Set before the beginning of an iteration. Does
  // not contain dirty objects.
  BitmapT iteration_bitmap;

  // Padding to 8 bytes.
  volatile TypeIndexT type_id;

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  __DEV__ void** forwarding_pointer_address(ObjectIndexT pos) const {
    char* block_base = const_cast<char*>(reinterpret_cast<const char*>(this));
    // Address of SOA array.
    auto* soa_array = reinterpret_cast<void**>(
        block_base + kBlockDataSectionOffset);
    return soa_array + pos;
  }

  __DEV__ void set_forwarding_pointer(ObjectIndexT pos, void* ptr) {
    *forwarding_pointer_address(pos) = ptr;
  }

  __DEV__ void* get_forwarding_pointer(ObjectIndexT pos) const {
    return *forwarding_pointer_address(pos);
  }
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
};


// A SOA block containing objects.
// T: Base type of the block.
// TypeId: Type ID of T.
// N: Maximum number of objects in a block of type T.
template<class T, TypeIndexT TypeId, ObjectIndexT N>
class SoaBlock : public AbstractBlock {
 public:
  static const int kN = N;
  static_assert(N <= 64, "Assertion failed: N <= 64");

#ifdef OPTION_DEFRAG
  // This is the number of allocated objects.
  static const ObjectIndexT kLeq50Threshold =
      1.0f*kDefragFactor / (kDefragFactor + 1) * N;
#endif  // OPTION_DEFRAG

  // Bitmap initializer: N_T bits set to 1.
  static const BitmapT kBitmapInitState =
      N == 64 ? (~0ULL) : ((1ULL << N) - 1);

  // Initializes a new block.
  __DEV__ SoaBlock() : AbstractBlock() {
    type_id = TypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
  }

  // Constructs an object identifier.
  __DEV__ T* make_pointer(ObjectIndexT index) {
    uint8_t obj_idx = reinterpret_cast<uint8_t&>(index);
    uintptr_t ptr_as_int = obj_idx;

    ObjectIndexT block_size = N;
    uintptr_t u_block_size = reinterpret_cast<uint8_t&>(block_size);
    ptr_as_int |= u_block_size << 48;

    TypeIndexT s_type_index = TypeId;
    uintptr_t type_id = reinterpret_cast<uint8_t&>(s_type_index);
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

  __DEV__ DeallocationState deallocate(ObjectIndexT position) {
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

  __DEV__ ObjectIndexT DBG_num_bits() {
    return N;
  }

  __DEV__ ObjectIndexT DBG_allocated_bits() {
    return N - __popcll(free_bitmap);
  }

  __DEV__ TypeIndexT get_static_type() const {
    return TypeId;
  }

  // Size of data segment.
  static const int kStorageBytes =
      SoaClassHelper<T>::template BlockConfig<N>::kDataSegmentSize;

  // Data storage.
  char data_[kStorageBytes];
};

#endif  // ALLOCATOR_SOA_BLOCK_H
