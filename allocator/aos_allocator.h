#ifndef MEMORY_AOS_ALLOCATOR_H
#define MEMORY_AOS_ALLOCATOR_H

#include <assert.h>
#include <tuple>

#include "bitmap/bitmap.h"

#define __DEV__ __device__

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

// Get size of biggest tuple element.
template<class Tuple>
struct TupleMaxSize;

template<class T, class... Types>
struct TupleMaxSize<std::tuple<T, Types...>> {
  static const size_t value =
      sizeof(T) > TupleMaxSize<std::tuple<Types...>>::value
          ? sizeof(T)
          : TupleMaxSize<std::tuple<Types...>>::value;
};

template<class T>
struct TupleMaxSize<std::tuple<T>> {
  static const size_t value = sizeof(T);
};


template<uint32_t N, class... Types>
class AosAllocator {
 public:
  __DEV__ void initialize() {
    global_free_.initialize(true);
    for (int i = 0; i < kNumTypes; ++i) {
      allocated_[i].initialize(false);
    }
  }

  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    uint32_t index = global_free_.deallocate();
    assert(index < N);

    bool success = allocated_[TupleIndex<T, TupleType>::value].allocate(index);
    assert(success);

    return new(data_location(index)) T(args...);
  }

  template<class T>
  __DEV__ void free(T* obj) {
    uint32_t index = bitmap_location(obj);
    // TODO: First crash is here!
    delete obj;

    bool success = allocated_[TupleIndex<T, TupleType>::value].deallocate<false>(index);
    assert(success);
    success = global_free_.allocate<false>(index);
    assert(success);
  }

  template<int TypeIndex>
  __DEV__ void free_untyped(void* obj) {
    auto* typed = static_cast<
        typename std::tuple_element<TypeIndex, TupleType>::type*>(obj);
    free(typed);
  }

 private:
  using TupleType = std::tuple<Types...>;

  static const int kNumTypes = std::tuple_size<TupleType>::value;

  static const int kTypeMaxSize = TupleMaxSize<TupleType>::value;

  char data_[N*kTypeMaxSize];

  Bitmap<uint32_t, N> global_free_;

  Bitmap<uint32_t, N> allocated_[kNumTypes];

  __DEV__ void* data_location(uint32_t index) {
    assert(index < N*kTypeMaxSize);
    return data_ + index*kTypeMaxSize;
  }

  __DEV__ uint32_t bitmap_location(void* ptr) {
    assert(ptr >= data_ && ptr < data_ + N*kTypeMaxSize);
    return (data_ - static_cast<char*>(ptr)) / kTypeMaxSize;
  }
};

#endif  // MEMORY_AOS_ALLOCATOR_H
