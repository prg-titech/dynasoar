#ifndef ALLOCATOR_TUPLE_HELPER_H
#define ALLOCATOR_TUPLE_HELPER_H

#include "allocator/soa_block.h"
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
};

template<>
struct SoaClassUtil<void> {
  static const int kNumFieldThisClass = 0;
};

// Helpers for SOA field "Index" in class C.
template<class C, int Index>
struct SoaFieldHelper {
  using type = typename std::tuple_element<Index, typename C::FieldTypes>::type;
  static const int kAlignment = ObjectAlignment<type>::value;

  static constexpr int offset(int block_size) {
    // constexprs may not contain statements.... ugh.
    return ((SoaFieldHelper<C, Index - 1>::offset(block_size)
             + SoaFieldHelper<C, Index - 1>::size(block_size)
        + kAlignment - 1) / kAlignment) * kAlignment;
  }

  static constexpr int size(int block_size) {
    return sizeof(type) * block_size;
  }

  // It is difficult to call constexpr functions and use their value in the
  // same class, so we provide a second implementation here based on templates.
  template<int BlockSize>
  struct BlockConfig {
    using PrevConfig = typename SoaFieldHelper<C, Index - 1>
        ::template BlockConfig<BlockSize>;

    static const int kOffset =
        ((PrevConfig::kOffset + PrevConfig::kSize
        + kAlignment - 1) / kAlignment) * kAlignment;

    static const int kSize = sizeof(type) * BlockSize;
  };

  static void DBG_print_stats() {
    printf("%s[%i]: type = %s, offset = %i, size = %i\n",
           typeid(C).name(), Index, typeid(type).name(),
           BlockConfig<1>::kOffset, BlockConfig<1>::kSize);
    SoaFieldHelper<C, Index - 1>::DBG_print_stats();
  }
};

template<class C>
struct SoaFieldHelper<C, -1> {
  using BaseLastFieldHelper = SoaFieldHelper<
      typename C::BaseClass,
      SoaClassUtil<typename C::BaseClass>::kNumFieldThisClass - 1>;

  static constexpr int offset(int block_size) {
    // Fields in superclass.
    return BaseLastFieldHelper::offset(block_size)
        + BaseLastFieldHelper::size(block_size);
  }

  static constexpr int size(int block_size) {
    return 0;
  }

  template<int BlockSize>
  struct BlockConfig {
    static const int kOffset =
        BaseLastFieldHelper::template BlockConfig<BlockSize>::kOffset
        + BaseLastFieldHelper::template BlockConfig<BlockSize>::kSize;

    static const int kSize = 0;
  };

  static void DBG_print_stats() {
    BaseLastFieldHelper::DBG_print_stats();
  }
};

template<>
struct SoaFieldHelper<void, -1> {
  static constexpr int offset(int block_size) { return 0; }

  static constexpr int size(int block_size) { return 0; }

  template<int BlockSize>
  struct BlockConfig {
    static const int kOffset = 0;

    static const int kSize = 0;
  };

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
        LastFieldHelper::template BlockConfig<BlockSize>::kOffset
        + LastFieldHelper::template BlockConfig<BlockSize>::kSize;
  };

  static void DBG_print_stats() {
    printf("----------------------------------------------------------\n");
    printf("Class %s: data_segment_size(1) = %i, data_segment_size(64) = %i\n",
           typeid(C).name(), BlockConfig<1>::kDataSegmentSize,
           BlockConfig<64>::kDataSegmentSize);
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

// Helpers for multiple SOA classes "Types".
template<class... Types>
struct TupleHelper;

template<class T, class... Types>
struct TupleHelper<T, Types...> {
  // Get largest SOA block size among all tuple elements.
  // The size of a block is chosen such that 64 objects of the smallest type
  // can fit.
  static const size_t kMaxSize =
      sizeof(T) > TupleHelper<Types...>::kMaxSize
          ? sizeof(SoaBlock<T, /*N_Max=*/ 64>)
          : TupleHelper<Types...>::kMaxSize;

  // Runs a functor for all types in the tuple.
  template<template<class> typename F>
  static void for_all() {
    F<T> func;
    func();
    TupleHelper<Types...>::template for_all<F>();
  }

  // Returns the index of U within the tuple.
  template<
      typename U,
      typename std::enable_if<std::is_same<U, T>::value, void*>::type = nullptr>
  static constexpr int tuple_index() {
    return 0;
  }

  template<
      typename U,
      typename std::enable_if<!std::is_same<U, T>::value, void*>::type = nullptr>
  static constexpr int tuple_index() {
    return TupleHelper<Types...>::template tuple_index<U>() + 1;
  }

  // Get type by index.
  template<int Index, int Dummy>
  struct Element {
    using type = typename TupleHelper<Types...>
        ::template Element<Index - 1, Dummy>::type;
  };

  template<int Dummy>
  struct Element<0, Dummy> { using type = T; };

  // Size of a block of size 64 of type T.
  static const int kThisClass64BlockSize =
      SoaClassHelper<T>::template BlockConfig<64>::kDataSegmentSize;

  static const bool kIsThisClassMinBlock =
      kThisClass64BlockSize <= TupleHelper<Types...>::k64BlockMinSize
      && !T::kIsAbstract;

  // Size of smallest block with 64 elements among all types.
  // Ignore abstract classes.
  static const int k64BlockMinSize = kIsThisClassMinBlock
      ? kThisClass64BlockSize : TupleHelper<Types...>::k64BlockMinSize;

  using Type64BlockSizeMin = typename std::conditional<
      kIsThisClassMinBlock,
      /*T=*/ T,
      /*F=*/ typename TupleHelper<Types...>::Type64BlockSizeMin>::type;
};

template<>
struct TupleHelper<> {
  static const size_t kMaxSize = 0;

  template<template<class> typename F>
  static void for_all() {}

  static const int kThisClass64BlockSize = std::numeric_limits<int>::max();
  static const int k64BlockMinSize = kThisClass64BlockSize;
  using Type64BlockSizeMin = void;
};

template<class T, int MaxSize, int BlockBytes>
struct SoaBlockSizeCalculator {
  static const int kThisSizeBlockBytes =
      SoaClassHelper<T>::template BlockConfig<MaxSize>::kDataSegmentSize;

  static const int kBytes =
      kThisSizeBlockBytes <= BlockBytes ? kThisSizeBlockBytes
      : SoaBlockSizeCalculator<T, MaxSize - 1, BlockBytes>::kBytes;

  static const int kSize =
      kThisSizeBlockBytes <= BlockBytes ? MaxSize
      : SoaBlockSizeCalculator<T, MaxSize - 1, BlockBytes>::kSize;
};

template<class T, int BlockBytes>
struct SoaBlockSizeCalculator<T, 0, BlockBytes> {
  static const int kBytes = std::numeric_limits<int>::max();

  static const int kSize = -1;
};

#define TYPE_INDEX(tuple, type) TupleHelper<tuple>::template tuple_index<type>()

#define TYPE_ELEMENT(tuple, index) typename TupleHelper<tuple...> \
    ::Element<index, /*Dummy=*/ 0>::type

#endif  // ALLOCATOR_TUPLE_HELPER_H
