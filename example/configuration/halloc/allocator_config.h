#ifndef EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED

template<typename AllocatorT>
struct AllocatorState {};

#include "../allocator_interface_adapter.h"
#include "halloc.cu"
#include "utils.cu"


void initialize_custom_allocator() {
  ha_init(halloc_opts_t(3ULL*kMallocHeapSize/4));
}


template<uint32_t N_Objects, class... Types>
template<class T, typename... Args>
__device__ T* SoaAllocator<N_Objects, Types...>
    ::external_allocator_make_new(Args... args) {
  // Use malloc and placement-new so that we can catch OOM errors.
  void* ptr = hamalloc(sizeof(T));
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
  hafree(ptr);
}

#endif  // EXAMPLE_CONFIGURATION_MALLOCMC_ALLOCATOR_CONFIG_H
