#ifndef EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED

#include "../allocator_interface_adapter.h"
#include "mallocmc_config.h"

template<typename AllocatorT>
struct AllocatorState {
  static const bool kHasParallelDo = false;

  ScatterAllocator::DevAllocator* allocator_handle;

  void initialize() {
    auto* host_handle = new ScatterAllocator(3ULL*kMallocHeapSize/4);
    auto* device_handle = host_handle->getAllocatorHandle().devAllocator;

    cudaMemcpy(&allocator_handle, &device_handle, sizeof(void*),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
    std::cout << ScatterAllocator::info("\n") << std::endl;
#endif  // NDEBUG
  }

  template<class T, typename... Args>
  __device__ T* make_new(Args... args) {
    // Use malloc and placement-new so that we can catch OOM errors.
    void* ptr = allocator_handle->malloc(sizeof(T));
    assert(ptr != nullptr);
    return new(ptr) T(args...); 
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    obj->~T();
    void* ptr = obj;
    allocator_handle->free(ptr);
  }
};


template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
