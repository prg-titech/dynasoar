#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H

#include "../allocator_interface_adapter.h"


static const size_t kMallocHeapSize = 3*1024U*1024*1024;


void initialize_custom_allocator() {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, kMallocHeapSize);

#ifndef NDEBUG
  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  assert(heap_size == kMallocHeapSize);
  printf("CUDA malloc heap size: %f MB\n", heap_size / 1024.0f / 1024.0f);
#endif  // NDEBUG
}


template<uint32_t N_Objects, class... Types>
template<class T, typename... Args>
__device__ T* SoaAllocator<N_Objects, Types...>
    ::external_allocator_make_new(Args... args) {
  // Use malloc and placement-new so that we can catch OOM errors.
  void* ptr = malloc(sizeof(T));
  assert(ptr != nullptr);
  return new(ptr) T(args...);
}


// SoaAllocator::free shadows CUDA free.
__device__ void cuda_free(void* ptr) { free(ptr); }


template<uint32_t N_Objects, class... Types>
template<class T>
__device__ void SoaAllocator<N_Objects, Types...>
    ::external_allocator_free(T* obj) {
  assert(obj != nullptr);
  obj->~T();
  void* ptr = obj;
  cuda_free(ptr);
}

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
