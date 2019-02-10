#ifndef ALLOCATOR_SOA_ALLOCATOR_H
#define ALLOCATOR_SOA_ALLOCATOR_H

#include "allocator/soa_allocator.h"

namespace _compiler_check {
  class DummyClass;
  using DummyAllocator = SoaAllocator<64*64*64, DummyClass>;

  class DummyClass {
   public:
    declare_field_types(DummyClass, int)
    SoaField<DummyClass, 0> field1_;
  };

  static_assert(sizeof(SoaField<DummyClass, 0>) == 0,
                "Compiler not supported. Need zero-size alternative impl.");

  static_assert(sizeof(DummyClass) == 0, "Internal error");
}  // _compiler_check

#endif