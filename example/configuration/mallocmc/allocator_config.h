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

  void initialize(size_t allocator_heap_size) {
    auto* host_handle = new ScatterAllocator(allocator_heap_size);
    auto* device_handle = host_handle->getAllocatorHandle().devAllocator;

    cudaMemcpy(&allocator_handle, &device_handle, sizeof(void*),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
    std::cout << ScatterAllocator::info("\n") << std::endl;
#endif  // NDEBUG
  }

  template<class T>
  __device__ T* allocate_new() {
    // Use malloc and placement-new so that we can catch OOM errors.
    T* ptr = (T*) allocator_handle->malloc(sizeof(T));
    assert(ptr != nullptr);
    return ptr;
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    void* ptr = obj;
    allocator_handle->free(ptr);
  }
};


template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
