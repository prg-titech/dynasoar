#ifndef BITMAP_SEQUENTIAL_ENUMERATE_H
#define BITMAP_SEQUENTIAL_ENUMERATE_H

template<typename BitmapT, bool HasNested, typename OuterBitmapT>
struct SequentialEnumerator {
  using OuterBitmapT = BitmapT::OuterBitmapT;

  template<typename... Args>
  struct HandlerWrapper {
    __DEV__ static void enumerate() {
      // OK
    }

    __DEV__ static BitmapT::OuterBitmapT* outer_bitmap(BitmapT* nested_bitmap) {
      auto nested_offset = offsetof()
    }
  }
};

template<typename BitmapT>
using BitmapEnumerator = SequentialEnumerator<BitmapT, BitmapT::kHasNested>;

#endif  // BITMAP_SEQUENTIAL_ENUMERATE_H
