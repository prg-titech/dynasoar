#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED

#include "../allocator_interface_adapter.h"


// SoaAllocator::free shadows CUDA free.
__device__ void cuda_free(void* ptr) { free(ptr); }


template<typename AllocatorT>
struct AllocatorState {
  static const bool kHasParallelDo = false;

  void initialize() {}

  template<class T, typename... Args>
  __device__ T* make_new(Args... args) {
    // Use malloc and placement-new so that we can catch OOM errors.
    void* ptr = malloc(sizeof(T));
    assert(ptr != nullptr);
    return new(ptr) T(args...);
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    obj->~T();
    void* ptr = obj;
    cuda_free(ptr);
  }
};

template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOC_ALLOCATOR_CONFIG_H
