#ifndef BITMAP_SEQUENTIAL_ENUMERATE_H
#define BITMAP_SEQUENTIAL_ENUMERATE_H

#include <type_traits>

template<typename BitmapT, bool HasNested, int Level>
struct SequentialEnumerator {
  using OuterBitmapT = BitmapT::OuterBitmapT;

  template<typename... Args>
  struct HandlerWrapper {
    template<bool N = HasNested>
    __DEV__ typename std::enable_if<N, void>::type
    static void enumerate(BitmapT* bitmap) {
      // Has nested bitmap. Delegate to next level.
      SequentialEnumerator<BitmapT::BitmapDataT::BitmapT,
                           BitmapT::BitmapDataT::BitmapT::kHasNested,
                           Level + 1>
          ::HandlerWrapper<Args...>::enumerate(&bitmap->data_.nested);
    }

    template<bool N = HasNested>
    __DEV__ typename std::enable_if<!N, void>::type
    static void enumerate(BitmapT* bitmap) {
      // Does not have a nested bitmap. Start top-down traversal.
      enumerate_top_down(bitmap, 0);
    }

    template<int L = Level>
    __DEV__ typename std::enable_if<(L > 0), void>::type
    static void enumerate_top_down(BitmapT* bitmap, BitmapT::SizeTT cid) {
      // Nested bitmap. Bits are container IDs in outer bitmap.
      assert(cid < BitmapT::kNumContainers);
      Bitmap::ContainerTT container = bitmap->data_.containers[cid];
      assert(container != 0);
      OuterBitmapT* outer = outer_bitmap(bitmap);

      // Enumerate all bits.
      while (container != 0) {
        int pos = BitmapT::kBitsize*cid + __ffsll(container) - 1;
        SequentialEnumerator<OuterBitmapT,
                             /*HasNested=*/ false,  /* does not matter */
                             Level - 1>
            ::HandlerWrapper<Args...>::enumerate_top_down(outer, pos);

        // Mask out bit from bitmap.
        container &= container - 1;
      }
    }

    template<int L = Level>
    __DEV__ typename std::enable_if<(L == 0), void>::type
    static void enumerate_top_down(BitmapT* bitmap, Bitmap::SizeTT cid) {
      // L0 bitmap.
      assert(cid < BitmapT::kNumContainers);
      Bitmap::ContainerTT container = bitmap->data_.containers[cid];
      assert(container != 0);

      // Enumerate all bits.
      while (container != 0) {
        int pos = BitmapT::kBitsize*cid + __ffsll(container) - 1;
        // TODO: Handle bit.
        printf("Found bit: %llu\n", pos);

        // Mask out bit from bitmap.
        container &= container - 1;
      }
    }

    __DEV__ static OuterBitmapT* outer_bitmap(BitmapT* nested_bitmap) {
      auto nested_offset = offsetof(OuterBitmapT::BitmapDataT, nested);
      return reinterpret_cast<OuterBitmapT*>(
          reinterpret_cast<uintptr_t>(nested_bitmap) - nested_offset);
    }
  }
};

template<typename BitmapT>
using BitmapEnumerator = SequentialEnumerator<BitmapT, BitmapT::kHasNested, 0>;

#endif  // BITMAP_SEQUENTIAL_ENUMERATE_H
