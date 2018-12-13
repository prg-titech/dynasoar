#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H

#include "../allocator_interface_adapter.h"

template<uint32_t N_Objects, class... Types>
template<class T, typename... Args>
__device__ T* SoaAllocator<N_Objects, Types...>
    ::external_allocator_make_new(Args... args) {
  return new T(args...);
}

template<uint32_t N_Objects, class... Types>
template<class T>
__device__ void SoaAllocator<N_Objects, Types...>
    ::external_allocator_free(T* obj) {
  delete obj;
}

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
