#ifndef WA_TOR_AOS_MALLOCMC_ALLOCATOR_H
#define WA_TOR_AOS_MALLOCMC_ALLOCATOR_H

#pragma once

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

// basic files for mallocMC
#include "src/include/mallocMC/mallocMC_hostclass.hpp"

// Load all available policies for mallocMC
#include "src/include/mallocMC/CreationPolicies.hpp"
#include "src/include/mallocMC/DistributionPolicies.hpp"
#include "src/include/mallocMC/OOMPolicies.hpp"
#include "src/include/mallocMC/ReservePoolPolicies.hpp"
#include "src/include/mallocMC/AlignmentPolicies.hpp"
    


// configurate the CreationPolicy "Scatter" to modify the default behaviour
struct ScatterHeapConfig : mallocMC::CreationPolicies::Scatter<>::HeapProperties{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashConfig : mallocMC::CreationPolicies::Scatter<>::HashingProperties{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct XMallocConfig : mallocMC::DistributionPolicies::XMallocSIMD<>::Properties {
  typedef ScatterHeapConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct ShrinkConfig : mallocMC::AlignmentPolicies::Shrink<>::Properties {
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator< 
  mallocMC::CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
  mallocMC::DistributionPolicies::XMallocSIMD<XMallocConfig>,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>
  > ScatterAllocator;

namespace wa_tor {
  //__device__ ScatterAllocator::AllocatorHandle m_allocator;
  __device__ char m_allocator[sizeof(ScatterAllocator::AllocatorHandle)];

  __device__ ScatterAllocator::AllocatorHandle& get_allocator() {
    return *reinterpret_cast<ScatterAllocator::AllocatorHandle*>(
        reinterpret_cast<void*>(m_allocator));
  }

  template<typename T, typename... Args>
  __device__ T* allocate(Args... args) {
    assert(get_allocator().malloc(10) != nullptr);
    return nullptr;

    void* data = get_allocator().malloc(sizeof(T));
    assert(data != nullptr);
    return new(data) T(args...);
  }

  template<typename T>
  __device__ void deallocate(T* ptr) {
    get_allocator().free(ptr);
  }

  template<int TypeIndex, typename T>
  __device__ void deallocate_untyped(T* ptr) {
    get_allocator().free(ptr);
  }

  __global__ void set_allocator(ScatterAllocator::AllocatorHandle mmc) {
    //m_allocator = mmc;
    memcpy(m_allocator, &mmc, sizeof(ScatterAllocator::AllocatorHandle));

    printf("avail: %u\n", get_allocator().getAvailableSlots(10));

    for (int i = 0; i< 100; ++i) {
      assert(get_allocator().malloc(sizeof(Fish)) != nullptr);
    //assert(mmc.malloc(10) != nullptr);
    }
    printf("avail: %u\n", get_allocator().getAvailableSlots(10));
  }

  void initialize_allocator() {
    ScatterAllocator mmc(64U*1024U*1024U); //64MB for device-side malloc
    std::cout << ScatterAllocator::info("\n") << std::endl;


    set_allocator<<<5,5>>>(mmc);
    gpuErrchk(cudaDeviceSynchronize());
  }
}  // namespace wa_tor

#endif  // WA_TOR_AOS_MALLOCMC_ALLOCATOR_H
