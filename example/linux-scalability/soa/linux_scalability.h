#ifndef EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H
#define EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H

#include "allocator_config.h"
#include "configuration.h"

// Pre-declare all classes.
class DummyClass;

using AllocatorT = SoaAllocator<kTotalNumObjects, DummyClass>;


class DummyClass : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<int, int, int, int>;

  SoaField<DummyClass, 0> field1_;
  SoaField<DummyClass, 1> field2_;
  SoaField<DummyClass, 2> field3_;
  SoaField<DummyClass, 3> field4_;
};

#endif  // EXAMPLE_LINUX_SCALABILITY_SOA_LINUX_SCALABILITY_H
