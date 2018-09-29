#ifndef ALLOCATOR_TUPLE_HELPER_H
#define ALLOCATOR_TUPLE_HELPER_H

#include "allocator/soa_block.h"
#include "allocator/soa_helper.h"

// Helpers for multiple SOA classes "Types".
template<class... Types>
struct TupleHelper;

template<class T, class... Types>
struct TupleHelper<T, Types...> {
  // The first non-abstract type.
  using NonAbstractType = typename std::conditional<
      !T::kIsAbstract,
      /*T=*/ T,
      /*F=*/ typename TupleHelper<Types...>::NonAbstractType>::type;

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

  // Smallest block size, padded to multiple of 64 bytes.
  static const int kPadded64BlockMinSize =
      ((k64BlockMinSize + 64 - 1) / 64) * 64;

  using Type64BlockSizeMin = typename std::conditional<
      kIsThisClassMinBlock,
      /*T=*/ T,
      /*F=*/ typename TupleHelper<Types...>::Type64BlockSizeMin>::type;
};

template<>
struct TupleHelper<> {
  using NonAbstractType = void;

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
