#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED


#include "bitmap/bitmap.h"
#include "../allocator_interface_adapter.h"


template<typename BitmapT>
__global__ void kernel_bitmap_alloc_init_bitmap(BitmapT* bitmap) {
  bitmap->initialize(true);
}


template<typename AllocatorT>
struct AllocatorState {
  static const int kObjectSize = AllocatorT::kLargestTypeSize;

  char* data_storage;
  Bitmap<uint32_t, AllocatorT::kMaxObjects> global_free;

  void initialize() {
    char* host_data_storage;
    cudaMalloc(&host_data_storage, 3ULL*kMallocHeapSize/4);
    assert(host_data_storage != nullptr);
    cudaMemcpy(&host_data_storage, &data_storage, sizeof(char*),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_bitmap_alloc_init_bitmap<<<128, 128>>>(&global_free);
    gpuErrchk(cudaDeviceSynchronize());
  }

  template<class T, typename... Args>
  __device__ T* make_new(Args... args) {
    // Use malloc and placement-new so that we can catch OOM errors.
    auto slot = global_free.deallocate();
    void* ptr = data_storage + slot*kObjectSize;
    assert(ptr != nullptr);
    return new(ptr) T(args...); 
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    assert(reinterpret_cast<uint64_t>(obj)
        >= reinterpret_cast<uint64_t>(data_storage));
    obj->~T();
    uint32_t slot = (reinterpret_cast<uint64_t>(obj)
        - reinterpret_cast<uint64_t>(data_storage)) / kObjectSize;
    assert((reinterpret_cast<uint64_t>(obj)
        - reinterpret_cast<uint64_t>(data_storage)) % kObjectSize == 0);
    global_free.allocate<true>(slot);
  }
};


template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
