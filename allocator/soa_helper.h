#ifndef ALLOCATOR_SOA_HELPER_H
#define ALLOCATOR_SOA_HELPER_H

#include "allocator/util.h"

// Helper functions and fields that are used in other classes in this file.
template<class C>
struct SoaClassUtil {
  static const int kNumFieldThisClass =
      std::tuple_size<typename C::FieldTypes>::value;
};

template<>
struct SoaClassUtil<void> {
  static const int kNumFieldThisClass = 0;
};

// Helpers for SOA field "Index" in class C.
template<class C, int Index>
struct SoaFieldHelper {
  using type = typename std::tuple_element<Index, typename C::FieldTypes>::type;
  using PrevHelper = SoaFieldHelper<C, Index - 1>;

  // Offset of this field.
  static const int kOffset = PrevHelper::kOffsetWithField;
  // End-offset of this field.
  static const int kOffsetWithField = kOffset + sizeof(type);

  static void DBG_print_stats() {
    printf("%s[%i]: type = %s, offset = %i, size = %lu\n",
           typeid(C).name(), Index, typeid(type).name(),
           kOffset, sizeof(type));
    SoaFieldHelper<C, Index - 1>::DBG_print_stats();
  }
};

template<class C>
struct SoaFieldHelper<C, -1> {
  using BaseLastFieldHelper = SoaFieldHelper<
      typename C::BaseClass,
      SoaClassUtil<typename C::BaseClass>::kNumFieldThisClass - 1>;

  // Align to multiple of 8 bytes when starting new subclass.
  static const int kOffset =
      ((BaseLastFieldHelper::kOffsetWithField + 8 - 1) / 8) * 8;
  static const int kOffsetWithField = kOffset;

  static void DBG_print_stats() {
    BaseLastFieldHelper::DBG_print_stats();
  }
};

template<>
struct SoaFieldHelper<void, -1> {
  static const int kOffset = 0;
  static const int kOffsetWithField = 0;

  static void DBG_print_stats() {}
};

// Helpers for SOA class C.
template<class C>
struct SoaClassHelper {
  static const int kNumFieldThisClass = SoaClassUtil<C>::kNumFieldThisClass;

  // The number of SOA fields in C, including fields of the superclass.
  static const int kNumFields =
      kNumFieldThisClass + SoaClassHelper<typename C::BaseClass>::kNumFields;

  template<int BlockSize>
  struct BlockConfig {
    using LastFieldHelper = SoaFieldHelper<C, kNumFieldThisClass - 1>;

    static const int kDataSegmentSize =
        LastFieldHelper::kOffsetWithField * BlockSize;
  };

  static void DBG_print_stats() {
    printf("----------------------------------------------------------\n");
    printf("Class %s: data_segment_size(1) = %i\n",
           typeid(C).name(), BlockConfig<1>::kDataSegmentSize);
    SoaFieldHelper<C, kNumFieldThisClass - 1>::DBG_print_stats();
    printf("----------------------------------------------------------\n");
  }
};

template<>
struct SoaClassHelper<void> {
  static const int kNumFieldThisClass = 0;
  static const int kNumFields = 0;

  static void DBG_print_stats() {}
};

#endif  // ALLOCATOR_SOA_HELPER_H
