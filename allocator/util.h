#ifndef ALLOCATOR_SOA_UTIL_H
#define ALLOCATOR_SOA_UTIL_H

#include "util/util.h"

__forceinline__ __device__ unsigned lane_id()
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

template<int W_MULT, class T, void(T::*func)(), typename AllocatorT>
__global__ void kernel_parallel_do(AllocatorT* allocator) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  allocator->template parallel_do_cuda<W_MULT, T, func>();
}

#endif  // ALLOCATOR_SOA_UTIL_H
