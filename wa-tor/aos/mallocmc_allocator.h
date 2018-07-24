#ifndef WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H
#define WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

#include <cuda.h>
#include "mallocMC/mallocMC.hpp"

#define uint32 uint32_t
typedef unsigned int uint;

using namespace mallocMC;

typedef mallocMC::Allocator<
  CreationPolicies::Scatter<>,
  DistributionPolicies::XMallocSIMD<>,
  OOMPolicies::ReturnNull,
  ReservePoolPolicies::SimpleCudaMalloc,
  AlignmentPolicies::Noop
> ScatterAllocator;


namespace wa_tor {
  //__device__ char alloc_handle [sizeof(ScatterAllocator::AllocatorHandle)];
  __device__ ScatterAllocator::AllocatorHandle alloc_handle;

  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    void* data = alloc_handle.malloc(sizeof(T));
    assert(data != nullptr);
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    alloc_handle.free(ptr);
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    alloc_handle.free(ptr);
  }

  __device__ void initialize_allocator() {}

  __global__ void copy_handle(ScatterAllocator::AllocatorHandle handle) {
    alloc_handle = handle;
  }

  void initHeap(int bytes) {
    auto* sa = new ScatterAllocator( 1U * 512U * 1024U * 1024U ); // heap size of 512MiB
    copy_handle<<<1,1>>>(*sa);
    gpuErrchk(cudaDeviceSynchronize());
  }
}  // namespace wa_tor


#endif  // WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

