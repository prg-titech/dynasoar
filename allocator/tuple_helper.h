#ifndef ALLOCATOR_TUPLE_HELPER_H
#define ALLOCATOR_TUPLE_HELPER_H

#include "allocator/soa_block.h"

// Get index of type within tuple.
// Taken from:
// https://stackoverflow.com/questions/18063451/get-index-of-a-tuple-elements-type
template<class T, class Tuple>
struct TupleIndex;

template<class T, class... Types>
struct TupleIndex<T, std::tuple<T, Types...>> {
  static const std::size_t value = 0;
};

template<class T, class U, class... Types>
struct TupleIndex<T, std::tuple<U, Types...>> {
  static const std::size_t value =
      1 + TupleIndex<T, std::tuple<Types...>>::value;
};

template<class Tuple>
struct TupleHelper;

template<class T, class... Types>
struct TupleHelper<std::tuple<T, Types...>> {
  // Get largest SOA block size among all tuple elements.
  // The size of a block is chosen such that 64 objects of the smallest type
  // can fit.
  static const size_t kMaxSize =
      sizeof(T) > TupleHelper<std::tuple<Types...>>::kMaxSize
          ? sizeof(SoaBlock<T, /*N_Max=*/ 64>)
          : TupleHelper<std::tuple<Types...>>::kMaxSize;

  // Run a functor for all types in the tuple.
  template<template<class> typename F>
  static void for_all() {
    F<T> func;
    func();
    TupleHelper<std::tuple<Types...>>::template for_all<F>();
  }
};

template<class T>
struct TupleHelper<std::tuple<T>> {
  static const size_t kMaxSize = sizeof(SoaBlock<T, /*N_Max=*/ 64>);

  template<template<class> typename F>
  static void for_all() {
    F<T> func;
    func();
  }
};

#endif  // ALLOCATOR_TUPLE_HELPER_H
