#ifndef BITMAP_SEQUENTIAL_ENUMERATE_H
#define BITMAP_SEQUENTIAL_ENUMERATE_H

#include <type_traits>

#include "util/util.h"

template<typename BitmapT, bool HasNested, int Level>
struct SequentialEnumerator {
  using OuterBitmapT = typename BitmapT::OuterBitmapT;
  using SizeT = typename BitmapT::SizeTT;
  using ContainerT = typename BitmapT::ContainerTT;

  template<typename... Args>
  struct HandlerWrapper {
    template<typename F, bool N = HasNested>
    __host_or_device__ static typename std::enable_if<N, void>::type
    enumerate(BitmapT* bitmap, F func, Args... args) {
      // Has nested bitmap. Delegate to next level.
      SequentialEnumerator<typename BitmapT::BitmapDataT::BitmapT,
                           BitmapT::BitmapDataT::BitmapT::kHasNested,
                           Level + 1>
          ::template HandlerWrapper<Args...>::enumerate(
              &bitmap->data_.nested, func, std::forward<Args>(args)...);
    }

    template<typename F, bool N = HasNested>
    __host_or_device__ static typename std::enable_if<!N, void>::type
    enumerate(BitmapT* bitmap, F func, Args... args) {
      // Does not have a nested bitmap. Start top-down traversal.
      enumerate_top_down(bitmap, 0, func, std::forward<Args>(args)...);
    }

    template<typename F, int L = Level>
    __host_or_device__ static typename std::enable_if<(L > 0), void>::type
    enumerate_top_down(BitmapT* bitmap, SizeT cid, F func, Args... args) {
      // Nested bitmap. Bits are container IDs in outer bitmap.
      assert(cid < BitmapT::kNumContainers);
      ContainerT container = bitmap->data_.containers[cid];
      assert(container != 0);
      OuterBitmapT* outer = outer_bitmap(bitmap);

      // Enumerate all bits.
      while (container != 0) {
        SizeT pos = BitmapT::kBitsize*cid + bit_ffsll(container) - 1;
        SequentialEnumerator<OuterBitmapT,
                             /*HasNested=*/ false,  /* does not matter */
                             Level - 1>
            ::template HandlerWrapper<Args...>
            ::enumerate_top_down(outer, pos, func,
                                 std::forward<Args>(args)...);

        // Mask out bit from bitmap.
        container &= container - 1;
      }
    }

    template<typename F, int L = Level>
    __host_or_device__ static typename std::enable_if<(L == 0), void>::type
    enumerate_top_down(BitmapT* bitmap, SizeT cid, F func, Args... args) {
      // L0 bitmap.
      assert(cid < BitmapT::kNumContainers);
      ContainerT container = bitmap->data_.containers[cid];
      assert(container != 0);

      // Enumerate all bits.
      while (container != 0) {
        SizeT pos = BitmapT::kBitsize*cid + bit_ffsll(container) - 1;
        func(pos, std::forward<Args>(args)...);

        // Mask out bit from bitmap.
        container &= container - 1;
      }
    }

    __host_or_device__ static OuterBitmapT* outer_bitmap(
        BitmapT* nested_bitmap) {
      auto nested_offset = offsetof(typename OuterBitmapT::BitmapDataT, nested);
      return reinterpret_cast<OuterBitmapT*>(
          reinterpret_cast<uintptr_t>(nested_bitmap) - nested_offset);
    }

  };
};

template<typename BitmapT>
using BitmapEnumerator = SequentialEnumerator<BitmapT, BitmapT::kHasNested, 0>;

template<typename SizeT, SizeT N, typename ContainerT, int ScanType>
template<typename F, typename... Args>
__host_or_device__ void Bitmap<SizeT, N, ContainerT, ScanType>::enumerate(
    F func, Args... args) {
  BitmapEnumerator<Bitmap<SizeT, N, ContainerT>>
      ::template HandlerWrapper<Args...>::enumerate(
          this, func, std::forward<Args>(args)...);
}

#endif  // BITMAP_SEQUENTIAL_ENUMERATE_H
