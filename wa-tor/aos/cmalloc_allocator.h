#ifndef WA_TOR_AOS_CUDA_ALLOCATOR_H
#define WA_TOR_AOS_CUDA_ALLOCATOR_H


#define _M_X64


#include "AllocTest.hpp"
#include "gpualloc.cu"

namespace wa_tor {
  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    void* data = mallocCircularMalloc(sizeof(T));
    assert(data != nullptr);
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    freeCircularMalloc(ptr);
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    freeCircularMalloc(ptr);
  }

  __device__ void initialize_allocator() {

  }

  void initHeap(int bytes) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, bytes);
    CircularMallocPrepare1<<<256, 32>>>(100);
    gpuErrchk(cudaDeviceSynchronize());
  }
}  // namespace wa_tor

#endif  // WA_TOR_AOS_CUDA_ALLOCATOR_H
