#ifndef WA_TOR_AOS_CUDA_ALLOCATOR_H
#define WA_TOR_AOS_CUDA_ALLOCATOR_H

namespace wa_tor {
  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    return new T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    delete ptr;
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    delete ptr;
  }

  __device__ void initialize_allocator() {}
}  // namespace wa_tor

#endif  // WA_TOR_AOS_CUDA_ALLOCATOR_H