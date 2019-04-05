#ifndef EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H
#define EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H

#include "allocator_config.h"
#include "configuration.h"

// Pre-declare all classes.
class DummyClass;

using AllocatorT = SoaAllocator<kNumObjects, DummyClass>;


class DummyClass : public AllocatorT::Base {
 public:
  declare_field_types(DummyClass, double, double, double, double,
                                  double, double, double, double)

  SoaField<DummyClass, 0> f0_;
  SoaField<DummyClass, 1> f1_;
  SoaField<DummyClass, 2> f2_;
  SoaField<DummyClass, 3> f3_;
  SoaField<DummyClass, 4> f4_;
  SoaField<DummyClass, 5> f5_;
  SoaField<DummyClass, 6> f6_;
  SoaField<DummyClass, 7> f7_;

  __device__ DummyClass() {}

};

#endif  // EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H
