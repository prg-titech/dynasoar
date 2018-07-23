#ifndef WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H
#define WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

#include <cuda.h>
#define uint32 uint32_t
typedef unsigned int uint;


#include "halloc.h"

namespace wa_tor {
  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    void* data = hamalloc(sizeof(T));
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    hafree(ptr);
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    hafree(ptr);
  }

  __device__ void initialize_allocator() {}

  __global__ void test_kernel() {
    void* x = hamalloc(128);
    printf("PTR: %p\n", x);
  }
  void initHeap(int bytes) {
    ha_init();
  }
}  // namespace wa_tor


#endif  // WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

