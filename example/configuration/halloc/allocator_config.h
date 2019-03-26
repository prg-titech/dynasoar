#ifndef EXAMPLE_CONFIGURATION_HALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_HALLOC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED

#include "../allocator_interface_adapter.h"
#include "halloc.cu"
#include "utils.cu"


template<typename AllocatorT>
struct AllocatorState {
  static const bool kHasParallelDo = false;

  void initialize(size_t allocator_heap_size) {
    ha_init(halloc_opts_t(allocator_heap_size));
  }

  template<class T>
  __device__ T* allocate_new() {
    // Use malloc and placement-new so that we can catch OOM errors.
    void* ptr = hamalloc(sizeof(T));
    assert(ptr != nullptr);
    return (T*) ptr;
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    void* ptr = obj;
    hafree(ptr);
  }
};

template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_HALLOC_ALLOCATOR_CONFIG_H
