#ifndef WA_TOR_AOS_AOS_ALLOCATOR_H
#define WA_TOR_AOS_AOS_ALLOCATOR_H

#include "allocator/aos_allocator.h"

#include "wa-tor/aos/wator.h"

namespace wa_tor {
  __device__ AosAllocator<64*64*64, Agent, Fish, Shark> memory_allocator;

  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    return memory_allocator.make_new<T>(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    memory_allocator.free<T>(ptr);
  }

  template<int TypeIndex>
  __device__ void deallocate_untyped(void* ptr) {
    memory_allocator.free_untyped<TypeIndex>(ptr);
  }

  __global__ void init_alloc() {
    memory_allocator.initialize();
  }

  void initialize_allocator() {
    init_alloc<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());
  }
}  // namespace wa_tor

#endif  // WA_TOR_AOS_AOS_ALLOCATOR_H