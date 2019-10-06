#ifndef ALLOCATOR_SOA_HELPER_H
#define ALLOCATOR_SOA_HELPER_H

#include <tuple>

#include "allocator/util.h"

// Determine to which byte boundary objects should be aligned.
template<typename T>
struct ObjectAlignment {
  static const int value = sizeof(T);
};

template<typename T, size_t N>
struct ObjectAlignment<DeviceArray<T, N>> {
  static const int value = sizeof(T);
};

// Helper functions and fields that are used in other classes in this file.
template<class C>
struct SoaClassUtil {
  static const int kNumFieldThisClass =
      std::tuple_size<typename C::FieldTypes>::value;

  static const int kNumFields =
      SoaClassUtil<typename C::BaseClass>::kNumFields + kNumFieldThisClass;
};

template<>
struct SoaClassUtil<void> {
  static const int kNumFieldThisClass = 0;
  static const int kNumFields = 0;
};

// Helpers for SOA field "Index" in class C.
template<class C, int Index>
struct SoaFieldHelper {
  using OwnerClass = C;
  using type = typename std::tuple_element<Index, typename C::FieldTypes>::type;
  using PrevHelper = SoaFieldHelper<C, Index - 1>;

  using ThisClass = SoaFieldHelper<C, Index>;

  // Required alignment of SOA array.
  static const int kAlignment = ObjectAlignment<type>::value;
  // Index of this field.
  static const int kIndex = Index;
  // Offset of this field.
  static const int kOffset = PrevHelper::kOffsetWithField;
  // End-offset of this field.
  static const int kOffsetWithField = kOffset + sizeof(type);
  // Count bytes of all fields without padding and alignment.
  static const int kSimpleSize = PrevHelper::kSimpleSize + sizeof(type);

  static_assert(SoaFieldHelper<C, Index - 1>::kAlignment % kAlignment == 0,
                "Fields in class must be sorted by size.");

  // Runs a functor for all fields in the tuple.
  // Returns true if F returned false for a field.
  template<template<class> typename F, bool IterateBase, typename... Args>
  static bool for_all(Args... args) {
    F<ThisClass> func;
    if (func(std::forward<Args>(args)...)) {
      return PrevHelper::template for_all<F, IterateBase>(
          std::forward<Args>(args)...);
    } else {
      // If F returns false, stop enumerating.
      return true;
    }
  }

  template<template<class> typename F, bool IterateBase, typename... Args>
  __device__ static bool dev_for_all(Args... args) {
    F<ThisClass> func;
    if (func(std::forward<Args>(args)...)) {
      return PrevHelper::template dev_for_all<F, IterateBase>(
          std::forward<Args>(args)...);
    } else {
      // If F returns false, stop enumerating.
      return true;
    }
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
  static const int kAlignment = SoaFieldHelper<C, 0>::kAlignment;
  static const int kSimpleSize = BaseLastFieldHelper::kSimpleSize;

  template<template<class> typename F, bool IterateBase, typename... Args>
  static bool for_all(Args... args) {
    if (IterateBase) {
      return BaseLastFieldHelper::template for_all<F, IterateBase>(
          std::forward<Args>(args)...);
    } else {
      return false;
    }
  }

  template<template<class> typename F, bool IterateBase, typename... Args>
  __device__ static bool dev_for_all(Args... args) {
    if (IterateBase) {
      return BaseLastFieldHelper::template dev_for_all<F, IterateBase>(
          std::forward<Args>(args)...);
    } else {
      return false;
    }
  }
};

template<>
struct SoaFieldHelper<void, -1> {
  static const int kOffset = 0;
  static const int kOffsetWithField = 0;
  static const int kSimpleSize = 0;

  template<template<class> typename F, bool IterateBase, typename... Args>
  static bool for_all(Args... args) { return false; }

  template<template<class> typename F, bool IterateBase, typename... Args>
  __device__ static bool dev_for_all(Args... args) { return false; }
};

// Helper for printing debug information about field.
template<typename SoaFieldHelperT>
struct SoaFieldDbgPrinter {
  bool operator()() {
    printf("│ %5i │ %15s │ %21s │ %8i │ %8i │\n",
           SoaFieldHelperT::kIndex,
           typeid(typename SoaFieldHelperT::OwnerClass).name(),
           typeid(typename SoaFieldHelperT::type).name(),
           (int) sizeof(typename SoaFieldHelperT::type),
           (int) SoaFieldHelperT::kOffset);
    return true;  // true means "continue processing".
  }
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

    static const int kDataSegmentSizeUnpadded =
        LastFieldHelper::kOffsetWithField * BlockSize;

    static const int kSimpleSize = LastFieldHelper::kSimpleSize * BlockSize;

    // Data segment size must be a multiple of 64.
    static const int kDataSegmentSize =
        ((kDataSegmentSizeUnpadded + 64 - 1) / 64) * 64;
  };

  static void DBG_print_stats() {
    printf("│ data seg. [ 1] sz  │ %8i bytes                                   │\n",
           (int) BlockConfig<1>::kDataSegmentSize);
    printf("│         (unpadded) │ %8i bytes                                   │\n",
           (int) BlockConfig<1>::kDataSegmentSizeUnpadded);
    printf("│ data seg. [64] sz  │ %8i bytes                                   │\n",
           (int) BlockConfig<64>::kDataSegmentSize);
    printf("│         (unpadded) │ %8i bytes                                   │\n",
           (int) BlockConfig<64>::kDataSegmentSizeUnpadded);

    printf("├────────────────────┴──────────────────────────────────────────────────┤\n");
    printf("│ Fields                                                                │\n");
    printf("├───────┬─────────────────┬───────────────────────┬──────────┬──────────┤\n");
    printf("│ Index │ Def. Class      │ Type                  │ Size     │ Offset   │\n");
    printf("├───────┼─────────────────┼───────────────────────┼──────────┼──────────┤\n");

    for_all<SoaFieldDbgPrinter, /*IterateBase=*/ true>();

    //Σ
    printf("├───────┼─────────────────┼───────────────────────┼──────────┼──────────┤\n");
    printf("│     Σ │                 │                       │ %8i │          │\n",
           (int) BlockConfig<1>::kSimpleSize);
    printf("└───────┴─────────────────┴───────────────────────┴──────────┴──────────┘\n");
  }

  template<template<class> typename F, bool IterateBase, typename... Args>
  static bool for_all(Args... args) {
    return SoaFieldHelper<C, kNumFieldThisClass - 1>
        ::template for_all<F, IterateBase>(std::forward<Args>(args)...);
  }

  template<template<class> typename F, bool IterateBase, typename... Args>
  __device__ static bool dev_for_all(Args... args) {
    return SoaFieldHelper<C, kNumFieldThisClass - 1>
        ::template dev_for_all<F, IterateBase>(std::forward<Args>(args)...);
  }
};

template<>
struct SoaClassHelper<void> {
  static const int kNumFieldThisClass = 0;
  static const int kNumFields = 0;

  static void DBG_print_stats() {}

  template<template<class> typename F, bool IterateBase, typename... Args>
  static bool for_all(Args... args) { return false; }

  template<template<class> typename F, bool IterateBase, typename... Args>
  __device__ static bool dev_for_all(Args... args) { return false; }
};

#endif  // ALLOCATOR_SOA_HELPER_H
