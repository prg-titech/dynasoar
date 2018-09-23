#ifndef ALLOCATOR_TUPLE_HELPER_H
#define ALLOCATOR_TUPLE_HELPER_H

#include "allocator/soa_block.h"

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

  template<int Index, int Dummy>
  struct Element {
    using type = typename TupleHelper<Types...>
        ::template Element<Index - 1, Dummy>::type;
  };

  template<int Dummy>
  struct Element<0, Dummy> { using type = T; };
};

template<>
struct TupleHelper<> {
  static const size_t kMaxSize = 0;

  template<template<class> typename F>
  static void for_all() {}
};

#define TYPE_INDEX(tuple, type) TupleHelper<tuple>::template tuple_index<type>()

#define TYPE_ELEMENT(tuple, index) typename TupleHelper<tuple...> \
    ::Element<index, /*Dummy=*/ 0>::type

#endif  // ALLOCATOR_TUPLE_HELPER_H
