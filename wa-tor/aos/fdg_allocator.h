#ifndef WA_TOR_AOS_FDG_ALLOCATOR_H
#define WA_TOR_AOS_FDG_ALLOCATOR_H

#include <cuda.h>
#define uint32 uint32_t
typedef unsigned int uint;


// include FDGMalloc Header
#include "fdg/FDGMalloc.cuh"

// include FDGMalloc Implementation
#include "fdg/FDGMalloc.cu"

using namespace FDG;

namespace wa_tor {
  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    Warp* warp = Warp::start();
    assert(warp != nullptr);

    void* data = warp->alloc(sizeof(T));
    assert(data != nullptr);
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    // Not implemented in FDG
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    
  }

  __device__ void initialize_allocator() {}

  void initHeap(int bytes) {
    // select device
    //selectDevice();
    //setHeapLimit(512 * 1024 * 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  }
}  // namespace wa_tor


#endif  // WA_TOR_AOS_FDG_ALLOCATOR_H


