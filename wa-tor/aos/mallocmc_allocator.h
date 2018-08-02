#ifndef WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H
#define WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

#include <cuda.h>
#include "mallocMC/mallocMC.hpp"

#define uint32 uint32_t
typedef unsigned int uint;

using namespace mallocMC;

struct ScatterHeapConfig : mallocMC::CreationPolicies::Scatter<>::HeapProperties{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<8>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashConfig : mallocMC::CreationPolicies::Scatter<>::HashingProperties{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterHeapConfig::pagesize pagesize;
};

typedef mallocMC::Allocator<
  CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
  DistributionPolicies::XMallocSIMD<DistributionConfig>,
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
    auto* sa = new ScatterAllocator( 2U * 512U * 1024U * 1024U ); // heap size of 512MiB
    copy_handle<<<1,1>>>(*sa);
    gpuErrchk(cudaDeviceSynchronize());
  }
}  // namespace wa_tor


#endif  // WA_TOR_AOS_SCATTERALLOC_ALLOCATOR_H

