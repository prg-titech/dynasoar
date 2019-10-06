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
  // Returns true if F returned false for one type. Returns true if iteration
  // terminated early, without processing all elements.
  template<template<class> typename F, typename... Args>
  static bool for_all(Args... args) {
    F<T> func;
    if (func(std::forward<Args>(args)...)) {
      return TupleHelper<Types...>::template for_all<F>(
          std::forward<Args>(args)...);
    } else {
      // If F returns false, stop enumerating.
      return true;
    }
  }

  template<template<class> typename F, typename... Args>
  __device__ __host__ static bool dev_for_all(Args... args) {
    F<T> func;
    if (func(std::forward<Args>(args)...)) {
      return TupleHelper<Types...>::template dev_for_all<F>(
          std::forward<Args>(args)...);
    } else {
      // If F returns false, stop enumerating.
      return true;
    }
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

  template<typename U, int Dummy>
  struct TupleIndex {
    static const int value =
        TupleHelper<Types...>::template TupleIndex<U, Dummy>::value + 1;
  };

  template<int Dummy>
  struct TupleIndex<T, Dummy>{
    static const int value = 0;
  };

  // Get type by index.
  template<int Index, int Dummy>
  struct Element {
    using type = typename TupleHelper<Types...>
        ::template Element<Index - 1, Dummy>::type;
  };

  template<int Dummy>
  struct Element<0, Dummy> { using type = T; };

  // Size of a block of size 64 of type T.
  // TODO: This is actually the data segment size. Rename.
  static const int kThisClass64BlockSize =
      SoaClassHelper<T>::template BlockConfig<64>::kDataSegmentSize;

  static const bool kIsThisClassMinBlock =
      kThisClass64BlockSize <= TupleHelper<Types...>::k64BlockMinSize
      && !T::kIsAbstract;

  // Size of smallest block with 64 elements among all types.
  // Ignore abstract classes.
  static const int k64BlockMinSize = kIsThisClassMinBlock
      ? kThisClass64BlockSize : TupleHelper<Types...>::k64BlockMinSize;
  static_assert(k64BlockMinSize % 64 == 0, "Invalid data segment size.");

  using Type64BlockSizeMin = typename std::conditional<
      kIsThisClassMinBlock,
      /*T=*/ T,
      /*F=*/ typename TupleHelper<Types...>::Type64BlockSizeMin>::type;

  static const int kLargestTypeSize =
      sizeof(T) > TupleHelper<Types...>::kLargestTypeSize
      ? sizeof(T) : TupleHelper<Types...>::kLargestTypeSize;
};

template<>
struct TupleHelper<> {
  using NonAbstractType = void;

  template<template<class> typename F, typename... Args>
  static bool for_all(Args... /*args*/) { return false; }

  template<template<class> typename F, typename... Args>
  __device__ __host__ static bool dev_for_all(Args... /*args*/) {
    return false;
  }

  static const int kThisClass64BlockSize = std::numeric_limits<int>::max();
  static const int k64BlockMinSize = kThisClass64BlockSize;
  using Type64BlockSizeMin = void;
  static const int kLargestTypeSize = 0;
};

// This helper calculates the size and #objects of a block of type T,
// given the target block size BlockBytes (should be k64BlockMinSize).
template<class T, int MaxSize, int BlockBytes>
struct SoaBlockSizeCalculator {
  static const int kThisSizeBlockBytes =
      SoaClassHelper<T>::template BlockConfig<MaxSize>::kDataSegmentSize;

  static const int kBytes =
      kThisSizeBlockBytes <= BlockBytes ? kThisSizeBlockBytes
      : SoaBlockSizeCalculator<T, MaxSize - 1, BlockBytes>::kBytes;
  static_assert(kBytes <= BlockBytes, "Invalid block size.");

  static const int kSize =
      kThisSizeBlockBytes <= BlockBytes ? MaxSize
      : SoaBlockSizeCalculator<T, MaxSize - 1, BlockBytes>::kSize;
};

template<class T, int BlockBytes>
struct SoaBlockSizeCalculator<T, 0, BlockBytes> {
  static const int kBytes = std::numeric_limits<int>::max();

  static const int kSize = -1;
};

#define TYPE_INDEX_VAR(tuple, type) TupleHelper<tuple>::template tuple_index<type>()

#define TYPE_INDEX(tuple, type) (TupleHelper<tuple>::template TupleIndex<type, 0>::value)

#define TYPE_ID(allocatort, type) allocatort::TypeId<type>::value

#define TYPE_ELEMENT(tuple, index) typename TupleHelper<tuple...> \
    ::Element<index, /*Dummy=*/ 0>::type

#endif  // ALLOCATOR_TUPLE_HELPER_H
