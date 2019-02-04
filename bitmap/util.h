#ifndef BITMAP_UTIL_H
#define BITMAP_UTIL_H

#include "util/util.h"

// Shift left, rotating.
// Copied from: https://gist.github.com/pabigot/7550454
template <typename T>
__DEV__ T rotl (T v, unsigned int b)
{
  static_assert(std::is_integral<T>::value, "rotate of non-integral type");
  static_assert(! std::is_signed<T>::value, "rotate of signed type");
  constexpr unsigned int num_bits {std::numeric_limits<T>::digits};
  static_assert(0 == (num_bits & (num_bits - 1)), "rotate value bit length not power of two");
  constexpr unsigned int count_mask {num_bits - 1};
  const unsigned int mb {b & count_mask};
  using promoted_type = typename std::common_type<int, T>::type;
  using unsigned_promoted_type = typename std::make_unsigned<promoted_type>::type;
  return ((unsigned_promoted_type{v} << mb)
          | (unsigned_promoted_type{v} >> (-mb & count_mask)));
}

// Seems like this is a scheduler warp ID and may change.
__forceinline__ __device__ unsigned warp_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__device__ __inline__ unsigned long long int ld_gbl_cg(
    const unsigned long long int *addr) {
  unsigned long long int return_value;
  asm("ld.global.cg.s64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

#endif  // BITMAP_UTIL_H
