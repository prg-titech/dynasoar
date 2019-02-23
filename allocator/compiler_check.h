#ifndef ALLOCATOR_COMPILER_CHECK_H
#define ALLOCATOR_COMPILER_CHECK_H

#include "allocator/soa_allocator.h"

namespace _compiler_check {
  class DummyClass;
  class DummyClass2;
  class DummyClass3;
  using DummyAllocator = SoaAllocator<
      64*64*64, DummyClass, DummyClass2, DummyClass3>;

  class DummyClass : public SoaBase<DummyAllocator> {
   public:
    declare_field_types(DummyClass, int, int)
    SoaField<DummyClass, 0> field1_;
    SoaField<DummyClass, 1> field1b_;
  };

  class DummyClass2 : public DummyClass {
   public:
    declare_field_types(DummyClass2, int)
    SoaField<DummyClass2, 0> field2_;
  };

  class DummyClass3 : public DummyClass {
   public:
    declare_field_types(DummyClass3, int)
    SoaField<DummyClass3, 0> field3_;
  };

  static_assert(sizeof(SoaField<DummyClass, 0>) == 0,
                "Compiler not supported. Need zero-size alternative impl.");

  static_assert(sizeof(SoaField<DummyClass2, 0>) == 0,
                "Compiler not supported. Need zero-size alternative impl.");

  static_assert(sizeof(SoaField<DummyClass3, 0>) == 0,
                "Compiler not supported. Need zero-size alternative impl.");

  static_assert(sizeof(DummyClass) == 2, "Internal error");

  static_assert(sizeof(DummyClass2) == 3, "Internal error");

  static_assert(sizeof(DummyClass3) == 3, "Internal error");
}  // _compiler_check

#endif  // ALLOCATOR_COMPILER_CHECK_H
