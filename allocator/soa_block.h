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

enum AllocationState : int8_t {
  kNoOp,                 // Nothing changed.
  kRegularAlloc,
  kBlockNowGt50,         // More than 50% full.
  kBlockNowFull,         // Deactivate block.
  kBlockNowFullAndGt50   // Full annd more than 50%.
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

  // Bitmap initializer: N_T bits set to 1.
  static const BitmapT kBitmapInitState =
      N == N_Max ? (~0ULL) : ((1ULL << N) - 1);

  // Result of block allocation.
  struct BlockAllocationResult {
    __device__ BlockAllocationResult(BitmapT allocation_mask_p,
                                     AllocationState state_p)
        : allocation_mask(allocation_mask_p), state(state_p) {}

    BitmapT allocation_mask;

    AllocationState state;
  };

  // Initializes a new block.
  __DEV__ SoaBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % N_Max == 0);   // Alignment.
    type_id = TypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
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
    iteration_bitmap = (~free_bitmap) & kBitmapInitState;
  }

  __DEV__ DeallocationState deallocate(int position) {
    BitmapT before;
    BitmapT mask = 1ULL << position;

    do {
      // successful if: bit was "0" (allocated). Needed because we could be in
      // invalidation check.
      before = atomicOr(&free_bitmap, mask);
    } while ((before & mask) != 0);

    int slots_free_before = __popcll(before);
    if (slots_free_before == 0) {
      return kBlockNowActive;
    } else if (slots_free_before == N - 1) {
      return kBlockNowEmpty;
    } else if (slots_free_before == N - kLeq50Threshold - 1) {
      return kBlockNowLeq50Full;
    } else {
      return kRegularDealloc;
    }
  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  __DEV__ BlockAllocationResult allocate(int bits_to_allocate) {
    // Allocation bits.
    BitmapT selected_bits = 0;
    // Set to true if block is full.
    bool block_full;
    // State of allocation result.
    AllocationState state = kNoOp;

    do {
      // Bit set to 1 if slot is free.
      unsigned int rotation_len = warp_id() % 64;
      BitmapT updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      int free_slots = __popcll(updated_mask);
      int allocation_size = min(free_slots, bits_to_allocate);
      BitmapT newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        int next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        int next_bit_pos_unrot = (next_bit_pos - rotation_len) % 64;
        newly_selected_bits |= 1ULL << next_bit_pos_unrot;
      }

      assert(__popcll(newly_selected_bits) == allocation_size);
      // Count the number of bits that were selected but already set to false
      // by another thread.
      BitmapT before_update = atomicAnd(&free_bitmap, ~newly_selected_bits);
      BitmapT successful_alloc = newly_selected_bits & before_update;
      block_full = (before_update & ~successful_alloc) == 0;

      if (successful_alloc > 0ULL) {
        // At least one slot allocated.
        int num_successful_alloc = __popcll(successful_alloc);
        bits_to_allocate -= num_successful_alloc;
        selected_bits |= successful_alloc;

        // First allocation.
        if (state == kNoOp) state = kRegularAlloc;

        if (state != kBlockNowGt50) {
          assert(state != kBlockNowFull);
          assert(state != kBlockNowFullAndGt50);

          // Check if more than 50% full now.
          int prev_full = N - __popcll(before_update);
          if (prev_full <= kLeq50Threshold
              && prev_full + num_successful_alloc > kLeq50Threshold) {
            state = kBlockNowGt50;
          }
        }

        if (block_full) {
          if (state == kRegularAlloc) state = kBlockNowFull;
#ifdef NDEBUG
          else /*if (state == kBlockNowGt50)*/ state = kBlockNowFullAndGt50;
#else
          else if (state == kBlockNowGt50) {
            state = kBlockNowFullAndGt50;
          }
          else assert(false);
#endif  // NDEBUG
        }
      }

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (bits_to_allocate > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return BlockAllocationResult(selected_bits, state);
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

 private:
  template<uint32_t, class...> friend class SoaAllocator;

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
