#ifndef ALLOCATOR_SOA_UTIL_H
#define ALLOCATOR_SOA_UTIL_H

#include "util/util.h"

#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__))


__forceinline__ __device__ unsigned __lane_id()
{
  unsigned ret;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__forceinline__ __device__ unsigned int __lanemask_lt() {
  unsigned int mask;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

template<typename T>
T copy_from_device(T* device_ptr) {
  T result;
  cudaMemcpy(&result, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
  return result;
}

template<typename T, T>
struct FunctionTypeGetter;

template<typename T, typename R, typename... Args, R (T::*func)(Args&&...)>
struct FunctionTypeGetter<R (T::*)(Args&&...), func> {
  static constexpr R (T::*FunctionType)(Args&&...) = func;
};

#endif  // ALLOCATOR_SOA_UTIL_H
