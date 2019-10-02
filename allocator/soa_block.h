#ifndef ALLOCATOR_SOA_BLOCK_H
#define ALLOCATOR_SOA_BLOCK_H

#include "allocator/configuration.h"
#include "allocator/soa_helper.h"
#include "allocator/util.h"

/**
 * Resulting state of an atomic deallocation in a block. Based on this state,
 * block state bitmap may have to be updated.
 */
enum DeallocationState : int8_t {
  /**
   * The block is now empty and should be deallocated.
   */
  kBlockNowEmpty,

#ifdef OPTION_DEFRAG
  /**
   * The block is now less/equal than n/(n+1) full, where n is the
   * defragmentation factor. By default, n = 1, so the default threshold is
   * 50%.
   */
  kBlockNowLeq50Full,
#endif  // OPTION_DEFRAG

  /**
   * The block is now active (previously inactive, i.e., full).
   */
  kBlockNowActive,

  /**
   * No state change. Nothing to do.
   */
  kRegularDealloc
};


/**
 * An untyped DynaSOAr block.
 */
class AbstractBlock {
 public:
  /**
   * 64-bit data type for object allocation/iteration bitmaps.
   */
  using BitmapT = unsigned long long int;

  /**
   * Constructor of this block.
   */
  __device__ AbstractBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % 64 == 0);   // Alignment.
  }

  /**
   * Checks if object slot \p index is allocated (in use).
   * @param index Object slot index
   */
  __device__ __host__ bool is_slot_allocated(ObjectIndexT index) const {
    return (free_bitmap & (1ULL << index)) == 0;
  }

  /**
   * Dummy area that may be overwritten by zero initialization.
   * Data section begins after kBlockDataSectionOffset bytes from the beginning
   * of this object.
   * TODO: Do we need this on GPU? Can this be replaced when using ROSE?
   */
  char initialization_header_[kBlockDataSectionOffset - 3*sizeof(BitmapT)];

  /**
   * Bitmap of free slots. Note: In contrast to the paper, we maintain a
   * bitmap of free object slots instead of an object allocation bitmap.
   */
  BitmapT free_bitmap;

  /*
   * Object iteration bitmap: A copy of ~free_bitmap. Initialized before the
   * beginning of an iteration.
   */
  BitmapT iteration_bitmap;

  /**
   * Type ID of this block. Must be volatile (alternative: only access with
   * atomic operations) to ensure that we always read the most recent type of
   * the block. TypeIndexT is an 8-bit integer, but the next field (data
   * segment in subclass) starts at an 8-byte offset.
   */
  volatile TypeIndexT type_id;

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  __device__ void** forwarding_pointer_address(ObjectIndexT pos) const {
    char* block_base = const_cast<char*>(reinterpret_cast<const char*>(this));
    // Address of SOA array.
    auto* soa_array = reinterpret_cast<void**>(
        block_base + kBlockDataSectionOffset);
    return soa_array + pos;
  }

  __device__ void set_forwarding_pointer(ObjectIndexT pos, void* ptr) {
    *forwarding_pointer_address(pos) = ptr;
  }

  __device__ void* get_forwarding_pointer(ObjectIndexT pos) const {
    return *forwarding_pointer_address(pos);
  }
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
};


/**
 * A DynaSOAr block containing up to \p N objects of type \p T in SOA data
 * layout.
 * @tparam T Type of objects in this block ("block type")
 * @tparam TypeId Type ID of T
 * @tparam N Maximum number of objects in this block (block capacity)
 */
template<class T, TypeIndexT TypeId, ObjectIndexT N>
class SoaBlock : public AbstractBlock {
 public:
  static const int kN = N;

  /**
   * Maximum block capacity is 64.
   */
  static_assert((N <= 64), "Assertion failed: N <= 64");

#ifdef OPTION_DEFRAG
  // This is the number of allocated objects.
  static const ObjectIndexT kLeq50Threshold =
      1.0f*kDefragFactor / (kDefragFactor + 1) * N;
#endif  // OPTION_DEFRAG

  /**
   * Initial free_bitmap state (if the entire block is empty).
   */
  static const BitmapT kBitmapInitState =
      N == 64 ? (~0ULL) : ((1ULL << N) - 1);

  /**
   * Initializes a new block. See paper for explanation of threadfence.
   */
  __device__ SoaBlock() : AbstractBlock() {
    type_id = TypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
  }

  /**
   * Constructs a fake pointer to the object at object slot \p index.
   * @param index Object slot index
   */
  __device__ __host__ T* make_pointer(ObjectIndexT index) const {
    uint8_t obj_idx = reinterpret_cast<uint8_t&>(index);
    assert(obj_idx < N);
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

  /**
   * Returns a copy of the object allocation bitmap.
   */
  __device__ __host__ BitmapT allocation_bitmap() const {
    return ~free_bitmap & kBitmapInitState;
  }

  /**
   * Initializes the object iteration bitmap.
   */
  __device__ void initialize_iteration() {
    iteration_bitmap = allocation_bitmap();
  }

  /**
   * Deallocates the object at a given position.
   * @param position Object slot index
   */
  __device__ DeallocationState deallocate(ObjectIndexT position) {
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

  /**
   * Returns the capacity of this block.
   */
  __device__ __host__ ObjectIndexT DBG_num_bits() const {
    return N;
  }

  /**
   * Returns the number of alloocated objects in this block.
   */
  __device__ __host__ ObjectIndexT DBG_allocated_bits() const {
    return N - bit_popcll(free_bitmap);
  }

  /**
   * Returns the type of this block.
   */
  __device__ __host__ TypeIndexT get_type() const {
    return TypeId;
  }

  /**
   * Size of data segment. The data segment must be large enough to hold all
   * SOA arrays. SOA arrays may also have to be padded, which must be taken
   * into account here.
   */
  static const int kStorageBytes =
      SoaClassHelper<T>::template BlockConfig<N>::kDataSegmentSize;

  /**
   * Data storage/data segment.
   */
  char data_[kStorageBytes];
};

#endif  // ALLOCATOR_SOA_BLOCK_H
