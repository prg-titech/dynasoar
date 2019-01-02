#ifndef EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H

#include "../allocator_interface_adapter.h"
#include "mallocmc_config.h"


ScatterAllocator* external_host_handle;
__device__ ScatterAllocator::DevAllocator* external_device_handle;


void initialize_custom_allocator() {
  external_host_handle = new ScatterAllocator(kMallocHeapSize);
  auto* dev_allocator = external_host_handle->getAllocatorHandle().devAllocator;

  cudaMemcpyToSymbol(external_device_handle, &dev_allocator, 
                     sizeof(ScatterAllocator::DevAllocator*), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
  std::cout << ScatterAllocator::info("\n") << std::endl;
#endif  // NDEBUG
}


template<uint32_t N_Objects, class... Types>
template<class T, typename... Args>
__device__ T* SoaAllocator<N_Objects, Types...>
    ::external_allocator_make_new(Args... args) {
  // Use malloc and placement-new so that we can catch OOM errors.
  void* ptr = external_device_handle->malloc(sizeof(T));
  assert(ptr != nullptr);
  return new(ptr) T(args...);
}


template<uint32_t N_Objects, class... Types>
template<class T>
__device__ void SoaAllocator<N_Objects, Types...>
    ::external_allocator_free(T* obj) {
  assert(obj != nullptr);
  obj->~T();
  void* ptr = obj;
  external_device_handle->free(ptr);
}

#endif  // EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
