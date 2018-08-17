#ifndef ALLOCATOR_SOA_ALLOCATOR_H
#define ALLOCATOR_SOA_ALLOCATOR_H

#include <assert.h>
#include <tuple>
#include <type_traits>

#include "bitmap/bitmap.h"

#define __DEV__ __device__

#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif  // gpuErrchk

// Only for debug purposes.
__device__ char* DBG_data_storage;
__device__ char* DBG_data_storage_end;

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned int __lanemask_lt() {
  unsigned int mask;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

template<int W_MULT, class T, void(T::*func)(), typename AllocatorT>
__global__ void kernel_parallel_do(AllocatorT* allocator) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  allocator->template parallel_do_cuda<W_MULT, T, func>();
}

template<typename T, int Field, int Offset>
class SoaField {
 private:
  // Offset of data section within SOA buffer.
  // TODO: Do not hard-code.
  static const int kSoaBufferOffset = 128;

  __DEV__ T* data_ptr() {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(this) - Field;
    uint8_t block_size = ptr_base >> 48;    // Should be truncated.
    uint8_t obj_id = static_cast<uint8_t>(ptr_base) & static_cast<uint8_t>(0x3F);   // Truncated.
    uintptr_t block_base = ptr_base & static_cast<uintptr_t>(0xFFFFFFFFFFC0);
    assert(obj_id < block_size);
    T* soa_array = reinterpret_cast<T*>(
        block_base + kSoaBufferOffset + block_size*Offset);

    assert(reinterpret_cast<char*>(soa_array + obj_id) > DBG_data_storage);
    assert(reinterpret_cast<char*>(soa_array + obj_id) < DBG_data_storage_end);
    return soa_array + obj_id;
  }

  __DEV__ T* data_ptr() const {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(this) - Field;
    uint8_t block_size = ptr_base >> 48;    // Should be truncated.
    uint8_t obj_id = static_cast<uint8_t>(ptr_base) & static_cast<uint8_t>(0x3F);   // Truncated.
    uintptr_t block_base = ptr_base & static_cast<uintptr_t>(0xFFFFFFFFFFC0);
    assert(obj_id < block_size);
    T* soa_array = reinterpret_cast<T*>(
        block_base + kSoaBufferOffset + block_size*Offset);

    assert(reinterpret_cast<char*>(soa_array + obj_id) > DBG_data_storage);
    assert(reinterpret_cast<char*>(soa_array + obj_id) < DBG_data_storage_end);
    return soa_array + obj_id;
  }

 public:
  __DEV__ SoaField() {}

  __DEV__ explicit SoaField(const T& value) {
    *data_ptr() = value;
  }

  __DEV__ operator T&() {
    return *data_ptr();
  }

  __DEV__ operator const T&() const {
    return *data_ptr();
  }

  __DEV__ T* operator&() {
    return data_ptr();
  }

  __DEV__ T operator->() {
    return *data_ptr();
  }

  // TODO: This may not be the correct implementation. Need a const reference?
  __DEV__ T& operator=(T value) {
    *data_ptr() = value;
    return *data_ptr();
  }
};

// Get index of type within tuple.
// Taken from:
// https://stackoverflow.com/questions/18063451/get-index-of-a-tuple-elements-type
template<class T, class Tuple>
struct TupleIndex;

template<class T, class... Types>
struct TupleIndex<T, std::tuple<T, Types...>> {
  static const std::size_t value = 0;
};

template<class T, class U, class... Types>
struct TupleIndex<T, std::tuple<U, Types...>> {
  static const std::size_t value =
      1 + TupleIndex<T, std::tuple<Types...>>::value;
};

struct BlockAllocationResult {
  __device__ BlockAllocationResult(uint64_t allocation_mask_p,
                                   bool block_full_p)
      : allocation_mask(allocation_mask_p), block_full(block_full_p) {}

  uint64_t allocation_mask;

  // Set to true if this allocation request filled up the block entirely.
  bool block_full;
};

enum DeallocationState : int8_t {
  kBlockNowEmpty,
  kBlockNowActive,
  kRegularDealloc
};

template<class T, int N_Max>
class SoaBlock {
 public:
  static const int N = T::kBlockSize;

  static const unsigned long long int kBitmapInitState =
      N == 64 ? (~0ULL) : ((1ULL << N) - 1);

  __DEV__ SoaBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % N_Max == 0);   // Alignment.
    type_id = T::kTypeId;
    __threadfence();
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
  }

  __DEV__ T* make_pointer(uint8_t index) {
    uintptr_t ptr_as_int = index;
    uintptr_t block_size = N;
    ptr_as_int |= block_size << 48;
    uintptr_t type_id = T::kTypeId;
    ptr_as_int |= type_id << 56;
    uintptr_t block_ptr = reinterpret_cast<uintptr_t>(this);
    assert(block_ptr < (1ULL << 49));   // Only 48 bits used in address space.
    assert((block_ptr & 0x3F) == 0);    // Block is aligned.
    ptr_as_int |= block_ptr;
    return reinterpret_cast<T*>(ptr_as_int);
  }

  __DEV__ void initialize_iteration() {
    iteration_bitmap = (~free_bitmap) & kBitmapInitState;
  }

  __DEV__ uint64_t invalidate() {
    auto old_free_bitmap = atomicExch(&free_bitmap, 0ULL);
    if (old_free_bitmap == kBitmapInitState) {
      return true;
    } else {
      free_bitmap = old_free_bitmap;
      return false;
    }
  }

  __DEV__ void uninvalidate(uint64_t previous_val) {
    free_bitmap = previous_val;
    // TODO: Thread fence?
  }

  __DEV__ DeallocationState deallocate(int position) {
    unsigned long long int before;
    unsigned long long int mask = 1ULL << position;

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
    } else {
      return kRegularDealloc;
    }
  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  __DEV__ BlockAllocationResult allocate(int bits_to_allocate) {
    // Allocation bits.
    unsigned long long int selected_bits = 0;
    // Set to true if this allocation filled up the block.
    bool filled_block = false, block_full;
    // Helper variables used inside the loop and in the loop condition.
    unsigned long long int before_update, successful_alloc;

    do {
      // Bit set to 1 if slot is free.
      unsigned int rotation_len = warp_id() % 64;
      unsigned long long int updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      int free_slots = __popcll(updated_mask);
      int allocation_size = min(free_slots, bits_to_allocate);
      unsigned long long int newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        // TODO: To reduce collisions attempt to start allocation at
        // different positions (rotating shift).
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
      before_update = atomicAnd(&free_bitmap, ~newly_selected_bits);
      successful_alloc = newly_selected_bits & before_update;
      bits_to_allocate -= __popcll(successful_alloc);
      selected_bits |= successful_alloc;

      // Block full if at least one slot was allocated and "before update"
      // bit-and "now allocated" indicates that block is full.
      block_full = (before_update & ~successful_alloc) == 0;
      filled_block = successful_alloc > 0 && block_full;

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (bits_to_allocate > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return BlockAllocationResult(selected_bits, filled_block);
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

  // TODO: Should be private.

 public:
  // Dummy area that may be overridden by zero initialization.
  // Data section begins after 128 bytes.
  // TODO: Do we need this on GPU?
  // TODO: Can this be replaced when using ROSE?
  char initialization_header_[128 - 3*sizeof(unsigned long long int)];

  // Bitmap of free slots.
  unsigned long long int free_bitmap;

  // A copy of ~free_bitmap. Set before the beginning of an iteration. Does
  // not contain dirty objects.
  unsigned long long int iteration_bitmap;

  // Padding to 8 bytes.
  uint8_t type_id;

  static const int kRawStorageBytes = N*T::kObjectSize;

  // Object size must be multiple of 64 bytes.
  static const int kStorageBytes = ((kRawStorageBytes + N_Max - 1) / N_Max) * N_Max;

  static_assert(N <= N_Max, "Assertion failed: N <= N_Max");

  // Data storage.
  char data_[kStorageBytes];
};

// Get largest SOA block size among all tuple elements.
// TODO: Assuming max. block size of 64.
template<class Tuple>
struct TupleMaxBlockSize;

template<class T, class... Types>
struct TupleMaxBlockSize<std::tuple<T, Types...>> {
  static const size_t value =
      sizeof(T) > TupleMaxBlockSize<std::tuple<Types...>>::value
          ? sizeof(SoaBlock<T, /*N_Max=*/ 64>)
          : TupleMaxBlockSize<std::tuple<Types...>>::value;
};

template<class T>
struct TupleMaxBlockSize<std::tuple<T>> {
  static const size_t value = sizeof(SoaBlock<T, /*N_Max=*/ 64>);
};

template<typename A>
__global__ void kernel_init_heap(A* allocator, void* ptr) {
  //printf("ALLO: %p\n", allocator);
  allocator->data_ = reinterpret_cast<char*>(ptr);
}

template<typename A>
void init_allocator_heap(A* dev_allocator, uint64_t bytes) {
  void* heap_addr;
  cudaMalloc(&heap_addr, bytes);
  assert(heap_addr != nullptr);
  assert(dev_allocator != nullptr);
  //printf("DEV ALLOCATOR: %p\nHEAP: %p\n", dev_allocator, heap_addr);
  kernel_init_heap<<<1,1>>>(dev_allocator, heap_addr);
  gpuErrchk(cudaDeviceSynchronize());
}

template<uint32_t N_Objects, class... Types>
class SoaAllocator {
 private:
  static const uint8_t kObjectAddrBits = 6;
  static const uint32_t kNumBlockElements = 64;
  static const uint64_t kObjectAddrBitmask = kNumBlockElements - 1;
  static const uint64_t kBlockAddrBitmask = 0xFFFFFFFFFFC0;
  static_assert(kNumBlockElements == 64,
                "Not implemented: Block size != 64.");
  static const int N = N_Objects / kNumBlockElements;

  static_assert(N_Objects % kNumBlockElements == 0,
                "N_Objects Must be divisible by BlockSize.");

 public:
  template<typename T>
  __DEV__ void remove_dirty() {
    // TODO: There may be a more efficient way to do this.
    // dirty_[TupleIndex<T, TupleType>::value].initialize(false);
    // Have to do this also with SoaBlocks.
  }

  __DEV__ void initialize() {
    // Check alignment of data storage buffer.
    assert(reinterpret_cast<uintptr_t>(data_) % 64 == 0);

    global_free_.initialize(true);
    for (int i = 0; i < kNumTypes; ++i) {
      allocated_[i].initialize(false);
      active_[i].initialize(false);
      dirty_[i].initialize(false);
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      DBG_data_storage = data_;
      DBG_data_storage_end = data_ + N*kBlockMaxSize;
    }
  }

  // Try to allocate everything in the same block.
  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    T* result = nullptr;

    do {
      const unsigned active = __activemask();
      // Leader thread is the first thread whose mask bit is set to 1.
      const int leader = __ffs(active) - 1;
      assert(leader >= 0 && leader < 32);
      // Use lane mask to empty all bits higher than the current thread.
      // The rank of this thread is the number of bits set to 1 in the result.
      const unsigned int rank = lane_id();
      assert(rank < 32);

      // Values to be calculated by the leader.
      uint32_t block_idx;
      uint64_t allocation_bitmap;
      if (rank == leader) {
        assert(__popc(__activemask()) == 1);    // Only one thread executing.

        block_idx = find_active_block<T>();
        auto* block = get_block<T>(block_idx);
        const auto allocation = allocate_in_block<T>(block,
                                                     /*num=*/ __popc(active));
        allocation_bitmap = allocation.allocation_mask;

        if (allocation.block_full) {
          // This request filled up the block entirely.
          bool success = active_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
          assert(success);
        }

        uint8_t actual_type_id = block->type_id;
        if (actual_type_id != T::kTypeId) {
          // Block deallocated and initialized for a new type between lookup
          // from active bitmap and here. This is extremely unlikely!
          // But possible.
          // Undo allocation and update bitmaps accordingly.
          unsigned long long int* free_bitmap = &block->free_bitmap;

          // Note: Cannot be in invalidation here, because we are certain that
          // we allocated the bits that we are about to deallocate here.
          auto before_undo = atomicOr(
              free_bitmap,
              static_cast<unsigned long long int>(allocation_bitmap));
          int slots_before_undo = __popcll(before_undo);

          // Cases to handle: block now active again or block empty now.
          if (slots_before_undo == 0) {
            // Block became active. (Was full.)
            bool success = active_[actual_type_id].allocate<true>(block_idx);
            assert(success);
          } else if (slots_before_undo == N - 1) {
            // Block now empty.
            if (block->invalidate()) {
              // Block is invalidated and no new allocations can be performed.
              bool success = active_[actual_type_id].deallocate<true>(block_idx);
              assert(success);
              success = allocated_[actual_type_id].deallocate<true>(block_idx);
              assert(success);
              success = global_free_.allocate<true>(block_idx);
              assert(success);
            }
          }
        }
      }

      assert(__activemask() == active);
      // Get pointer from allocation (nullptr if no allocation).
      allocation_bitmap = __shfl_sync(active, allocation_bitmap, leader);
      block_idx = __shfl_sync(active, block_idx, leader);
      assert(block_idx < N);
      result = get_ptr_from_allocation<T>(
          block_idx, __popc(__lanemask_lt() & active), allocation_bitmap);

      //printf("ALLOCTED: %i / %i, rank=%i, lt=%i, result=%p,\n",(int) __popc(allocation_bitmap), (int)__popc(active), (int)rank, (int)__popc(__lanemask_lt()&active), result);

    } while (result == nullptr);

    //assert(reinterpret_cast<char*>(result) >= data_);
    //assert(reinterpret_cast<char*>(result) < data_ + N*kBlockMaxSize);
    return new(result) T(args...);
    //return (T*) result;
  }

  template<class T>
  __DEV__ void free(T* obj) {
    //assert(reinterpret_cast<char*>(obj) >= data_);
    //assert(reinterpret_cast<char*>(obj) < data_ + N*kBlockMaxSize);

    obj->~T();
    const uint32_t block_idx = get_block_idx<T>(obj);
    const uint32_t obj_id = get_object_id<T>(obj);
    const DeallocationState dealloc_state = deallocate_in_block<T>(block_idx,
                                                                   obj_id);

    if (dealloc_state == kBlockNowActive) {
      bool success = active_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
      assert(success);
    } else if (dealloc_state == kBlockNowEmpty) {
      // Assume that block is empty.
      if (invalidate_block<T>(block_idx)) {
        // Block is invalidated and no new allocations can be performed.
        bool success = active_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
        assert(success);
        success = allocated_[TupleIndex<T, TupleType>::value].deallocate<true>(block_idx);
        assert(success);
        success = global_free_.allocate<true>(block_idx);
        assert(success);
      }
    }
  }

  template<int TypeIndex>
  __DEV__ void free_untyped(void* obj) {
    auto* typed = static_cast<
        typename std::tuple_element<TypeIndex, TupleType>::type*>(obj);
    free(typed);
  }

  // W_SZ: Allocated threads per block.
  // Problem: Small W_SZ means less memory coalescing.
  /*
  template<int W_SZ, class T, void(T::*func)()>
  __DEV__ void parallel_do() {
    // TODO: This may not be doing the right thing. What if we have more
    // threads than blocks?

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
      // Process element (block) i with W_SZ many threads.
      int warp_id = i / W_SZ;
      int warp_offset = i % W_SZ;

      for (int j = 0; j < W_SZ; ++j) {
        uint32_t block_idx = warp_id*W_SZ + j;

        if (block_idx < N
            && allocated_[TupleIndex<T, TupleType>::value][block_idx]) {
          auto* block = get_block<T>(block_idx);
          assert(reinterpret_cast<uintptr_t>(block)
              >= reinterpret_cast<uintptr_t>(DBG_data_storage));
          assert(reinterpret_cast<uintptr_t>(block)+64
              < reinterpret_cast<uintptr_t>(DBG_data_storage_end));

          auto iteration_bitmap = block->iteration_bitmap;
          int block_size = __popcll(iteration_bitmap);

          // Advance bitmap to return warp_offset-th bit index.
          for (int i = 0; i < warp_offset; ++i) {
            // Clear last bit.
            iteration_bitmap &= iteration_bitmap - 1;
          }

          for (int object_idx = warp_offset; object_idx < block_size;
               object_idx += W_SZ) {
            int obj_bit = __ffsll(iteration_bitmap);
            assert(obj_bit > 0);
            T* obj = get_object<T>(block, obj_bit - 1);
            // Call function.
            (obj->*func)();

            if (object_idx + W_SZ < block_size) {
              // There will be another iteration. Advance bitmap.
              for (int i = 0; i < W_SZ; ++i) {
                // Clear last bit.
                iteration_bitmap &= iteration_bitmap - 1;
              }
            }
          }
        }
      }
    }
  }
  */

  // Should be invoked from host side.
  template<int W_MULT, class T, void(T::*func)()>
  void parallel_do(int num_blocks, int num_threads) {
    allocated_[TupleIndex<T, TupleType>::value].scan();
    kernel_parallel_do<W_MULT, T, func><<<num_blocks, num_threads>>>(this);
    gpuErrchk(cudaDeviceSynchronize());
  }

  // Device-side version, invoked from kernel.
  // This version does a #threads-stride loop, which is probably not optimal.
  template<int W_MULT, class T, void(T::*func)()>
  __DEV__ void parallel_do_cuda2() {
    const uint32_t N_alloc = allocated_[TupleIndex<T, TupleType>::value].scan_num_bits();

    int num_threads = blockDim.x * gridDim.x;
    assert(num_threads > N_alloc);
    int threads_per_block = num_threads/N_alloc;
    assert(threads_per_block > 0);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid/threads_per_block < N_alloc) {
      int block_idx = allocated_[TupleIndex<T, TupleType>::value].scan_get_index(
          tid/threads_per_block);

      // TODO: Consider doing a scan over "allocated" bitmap.
      auto* block = get_block<T>(block_idx);
      auto iteration_bitmap = block->iteration_bitmap;
      int block_size = __popcll(iteration_bitmap);
      //int objs_per_thread = block_size/threads_per_block + 1;

      // Offset of this thread when processing this block.
      int thread_offset = tid - (tid/threads_per_block)*threads_per_block;
      // Advance bitmap to return thread_offset-th bit index.
      for (int i = 0; i < thread_offset; ++i) {
        // Clear last bit.
        iteration_bitmap &= iteration_bitmap - 1;
      }

      // Now process objects within block.
      for (int pos = thread_offset; pos < block_size; pos += threads_per_block) {
        int obj_bit = __ffsll(iteration_bitmap);
        assert(obj_bit > 0);
        T* obj = get_object<T>(block, obj_bit - 1);
        // Call function.
        (obj->*func)();

        if (pos + threads_per_block < block_size) {
          // There will be another iteration. Advance bitmap.
          for (int i = 0; i < threads_per_block; ++i) {
            // Clear last bit.
            iteration_bitmap &= iteration_bitmap - 1;
          }
        }
      }
    }
  }

  // This version assigns 64 threads to every block.
  template<int W_MULT, class T, void(T::*func)()>
  __DEV__ void parallel_do_cuda() {
    const uint32_t N_alloc = allocated_[TupleIndex<T, TupleType>::value].scan_num_bits();
    const int num_objs = T::kBlockSize;

    // Round to multiple of 64.
    int num_threads = ((blockDim.x * gridDim.x)/num_objs)*num_objs;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
      for (int j = tid/num_objs; j < N_alloc; j += num_threads/num_objs) {
        // i is the index of in the scan array.
        int block_idx = allocated_[TupleIndex<T, TupleType>::value].scan_get_index(j);

        // TODO: Consider doing a scan over "allocated" bitmap.
        auto* block = get_block<T>(block_idx);
        auto iteration_bitmap = block->iteration_bitmap;
        int block_size = __popcll(iteration_bitmap);
        //int objs_per_thread = block_size/threads_per_block + 1;

        int thread_offset = tid % num_objs;
        // Advance bitmap to return thread_offset-th bit index.
        for (int i = 0; i < thread_offset; ++i) {
          // Clear last bit.
          iteration_bitmap &= iteration_bitmap - 1;
        }
        int obj_bit = __ffsll(iteration_bitmap);
        if (obj_bit > 0) {
          T* obj = get_object<T>(block, obj_bit - 1);
          // call the function.
          (obj->*func)();
        }
      }
    }
  }

  template<typename T>
  __DEV__ void initialize_iteration() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
      // TODO: Check dirty bitmap instead?
      if (allocated_[TupleIndex<T, TupleType>::value][i]) {
        // Initialize block.
        get_block<T>(i)->initialize_iteration();
      }
    }
  }

  // TODO: Should be private.
 public:
  template<class T>
  __DEV__ uint32_t find_active_block() {
    uint32_t block_idx;

    do {
      // Retry a couple of times. May reduce fragmentation.
      // TODO: Tune number of retries.
      int retries = 5;   // retries=2 before
      do {
        block_idx = active_[TupleIndex<T, TupleType>::value].template find_allocated<false>(retries);
      } while (block_idx == Bitmap<uint32_t, N>::kIndexError
               && --retries > 0);

      if (block_idx == Bitmap<uint32_t, N>::kIndexError) {
        // TODO: May be out of memory here.
        block_idx = global_free_.deallocate();
        assert(block_idx != (Bitmap<uint32_t, N>::kIndexError));  // OOM
        initialize_block<T>(block_idx);
        bool success = allocated_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
        assert(success);
        success = active_[TupleIndex<T, TupleType>::value].allocate<true>(block_idx);
      }
    } while (block_idx == Bitmap<uint32_t, N>::kIndexError);

    assert(block_idx < N);
    return block_idx;
  }

  template<class T>
  __DEV__ void initialize_block(uint32_t block_idx) {
    static_assert(sizeof(SoaBlock<T, kNumBlockElements>)
          % kNumBlockElements == 0,
        "Internal error: SOA block not aligned to 64 bytes.");
    new(get_block<T>(block_idx)) SoaBlock<T, kNumBlockElements>();
  }

  template<class T>
  __DEV__ uint32_t get_block_idx(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t data_as_int = reinterpret_cast<uintptr_t>(data_);

    assert(((ptr_as_int & kBlockAddrBitmask) - data_as_int) % kBlockMaxSize == 0);
    return ((ptr_as_int & kBlockAddrBitmask) - data_as_int) / kBlockMaxSize;
  }

  template<class T>
  __DEV__ uint32_t get_object_id(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    return ptr_as_int & kObjectAddrBitmask; 
  }

  template<class T>
  __DEV__ T* get_object(SoaBlock<T, kNumBlockElements>* block, uint32_t obj_id) {
    assert(obj_id < 64);
    return block->make_pointer(obj_id);
  }

  template<class T>
  __DEV__ SoaBlock<T, kNumBlockElements>* get_block(uint32_t block_idx) {
    assert(block_idx < N);
    return reinterpret_cast<SoaBlock<T, kNumBlockElements>*>(
        data_ + block_idx*kBlockMaxSize);
  }

  template<class T>
  __DEV__ BlockAllocationResult allocate_in_block(SoaBlock<T, kNumBlockElements>* block,
                                                  int num_objects) {
    // Only executed by one thread per warp.
    return block->allocate(num_objects);
  }

  // Return value indicates if block was emptied or activated.
  template<class T>
  __DEV__ DeallocationState deallocate_in_block(uint32_t block_idx,
                                                uint32_t obj_id) {
    assert(block_idx < N);
    auto* block = get_block<T>(block_idx);
    return block->deallocate(obj_id);
  }

  template<class T>
  __DEV__ T* get_ptr_from_allocation(uint32_t block_idx, int rank,
                                     uint64_t allocation) {
    assert(block_idx < N);
    assert(rank < 32);

    // Get index of rank-th first bit set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      allocation &= allocation - 1;
    }

    int position = __ffsll(allocation);

    if (position > 0) {
      // Allocation successful.
      return get_block<T>(block_idx)->make_pointer(position - 1);
    } else {
      return nullptr;
    }
  }

  template<class T>
  __DEV__ bool invalidate_block(uint32_t block_idx) {
    return get_block<T>(block_idx)->invalidate();
  }

  // The number of allocated slots of a type. (#blocks * blocksize)
  template<class T>
  __DEV__ uint32_t DBG_allocated_slots() {
    uint32_t counter = 0;
    for (int i = 0; i < N; ++i) {
      if (allocated_[TupleIndex<T, TupleType>::value][i]) {
        counter += get_block<T>(i)->DBG_num_bits();
      }
    }
    return counter;
  }

  // The number of actually used slots of a type. (#blocks * blocksize)
  template<class T>
  __DEV__ uint32_t DBG_used_slots() {
    uint32_t counter = 0;
    for (int i = 0; i < N; ++i) {
      if (allocated_[TupleIndex<T, TupleType>::value][i]) {
        counter += get_block<T>(i)->DBG_allocated_bits();
      }
    }
    return counter;
  }

  template<typename T>
  __DEV__ bool is_block_allocated(uint32_t index) {
    return allocated_[TupleIndex<T, TupleType>::value][index];
  }

  using TupleType = std::tuple<Types...>;

  static const int kNumTypes = std::tuple_size<TupleType>::value;

  static const int kBlockMaxSize = TupleMaxBlockSize<TupleType>::value;

  static const uint32_t kN = N;

  char* data_;

  Bitmap<uint32_t, N> global_free_;

  Bitmap<uint32_t, N> allocated_[kNumTypes];

  Bitmap<uint32_t, N> active_[kNumTypes];

  // TODO: No hierarchy needed in this bitmap.
  // TODO: Not needed anymore.
  Bitmap<uint32_t, N> dirty_[kNumTypes];
};

#endif  // ALLOCATOR_SOA_ALLOCATOR_H
