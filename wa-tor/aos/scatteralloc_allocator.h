#ifndef WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H
#define WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

#include <cuda.h>
#define uint32 uint32_t
typedef unsigned int uint;

#define HEAPARGS 4096, 8, 16, 8, true, false

#include "scatteralloc/heap_impl.cuh"

namespace wa_tor {
  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    void* data = theHeap.alloc(sizeof(T));
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    theHeap.dealloc(ptr);
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    theHeap.dealloc(ptr);
  }

  __device__ void initialize_allocator() {}
}  // namespace wa_tor


#endif  // WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H
