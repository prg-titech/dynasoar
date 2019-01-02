#ifndef EXAMPLE_CONFIGURATION_MALLOCMC_MALLOCMC_CONFIG_H
#define EXAMPLE_CONFIGURATION_MALLOCMC_MALLOCMC_CONFIG_H

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

#endif  // EXAMPLE_CONFIGURATION_MALLOCMC_MALLOCMC_CONFIG_H
