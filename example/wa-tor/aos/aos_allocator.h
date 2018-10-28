#ifndef WA_TOR_AOS_AOS_ALLOCATOR_H
#define WA_TOR_AOS_AOS_ALLOCATOR_H

#include "allocator/aos_allocator.h"

#include "wa-tor/aos/wator.h"

namespace wa_tor {
  __device__ AosAllocator<64*64*64*64, Agent, Fish, Shark, Cell> memory_allocator;

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

  __device__ void initialize_allocator() {
    memory_allocator.initialize();
  }

  void initHeap(int size) {}
}  // namespace wa_tor

#endif  // WA_TOR_AOS_AOS_ALLOCATOR_H