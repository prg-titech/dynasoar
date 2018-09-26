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

  // Size of smallest block among all types checked so far.
  static constexpr int min_block_bytes(int block_size);

  // Types with smallest block size.
  using min_block_type = typename std::conditional<
      (min_block_bytes(64) <
          typename TupleHelper<Types...>::min_block_bytes(64)),
      /*T=*/ T,
      /*F=*/ typename TupleHelper<Types...>::min_block_type(64)>::type;
};

template<>
struct TupleHelper<> {
  static const size_t kMaxSize = 0;

  template<template<class> typename F>
  static void for_all() {}

  static constexpr int min_block_bytes(int block_size) {
    return std::numeric_limits<int>::max();
  }

  using first_type = void;
};

template<class C>
struct SoaClassHelper {
  static const int kNumFieldThisClass =
      std::tuple_size<typename C::FieldTypes>::value;

  // The number of SOA fields in C, including fields of the superclass.
  static const int kNumFields =
      kNumFieldThisClass + SoaClassHelper<typename C::BaseClass>::kNumFields;

  // Determine the size of the data segment.
  static constexpr int data_segment_size(int block_size);

  static void DBG_print_stats();
};

template<>
struct SoaClassHelper<void> {
  static const int kNumFieldThisClass = 0;
  static const int kNumFields = 0;

  static void DBG_print_stats() {}
};

// Get offset of SOA array within block.
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

  static void DBG_print_stats() {
    printf("%s[%i]: type = %s, offset = %i, size = %i\n",
           typeid(C).name(), Index, typeid(type).name(), offset(1), size(1));
    SoaFieldHelper<C, Index - 1>::DBG_print_stats();
  }
};

template<class C>
struct SoaFieldHelper<C, -1> {
  using BaseLastFieldHelper = SoaFieldHelper<
      typename C::BaseClass,
      SoaClassHelper<typename C::BaseClass>::kNumFieldThisClass - 1>;

  static constexpr int offset(int block_size) {
    // Fields in superclass.
    return BaseLastFieldHelper::offset(block_size)
        + BaseLastFieldHelper::size(block_size);
  }

  static constexpr int size(int block_size) {
    return 0;
  }

  static void DBG_print_stats() {
    BaseLastFieldHelper::DBG_print_stats();
  }
};

template<>
struct SoaFieldHelper<void, -1> {
  static constexpr int offset(int block_size) { return 0; }

  static constexpr int size(int block_size) { return 0; }

  static void DBG_print_stats() {}
};

template<class C>
void SoaClassHelper<C>::DBG_print_stats() {
  SoaFieldHelper<C, kNumFieldThisClass - 1>::DBG_print_stats();
}

template<class C>
constexpr int SoaClassHelper<C>::data_segment_size(int block_size) {
  return SoaFieldHelper<C, kNumFieldThisClass>::offset()
      + SoaFieldHelper<C, kNumFieldThisClass>::size();
}

template<typename T, class... Types>
constexpr int TupleHelper<T, Types...>::min_block_bytes(int block_size) {
  return std::min(TupleHelper<Types...>::min_block_bytes(block_size),
                  SoaClassHelper<T>::data_segment_size(block_size));
}

#define TYPE_INDEX(tuple, type) TupleHelper<tuple>::template tuple_index<type>()

#define TYPE_ELEMENT(tuple, index) typename TupleHelper<tuple...> \
    ::Element<index, /*Dummy=*/ 0>::type

#endif  // ALLOCATOR_TUPLE_HELPER_H
