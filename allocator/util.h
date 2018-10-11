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

template<class T, typename AllocatorT>
__global__ void kernel_initialize_leq(AllocatorT* allocator) {
  allocator->template initialize_leq_work_bitmap<T>();
}

template<class T, typename AllocatorT>
__global__ void kernel_defrag_move(AllocatorT* allocator, int num_records) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_records*32) {
    allocator->template defrag_move<T>();
  }
}

template<class T, typename AllocatorT>
__global__ void kernel_defrag_scan(AllocatorT* allocator, int num_records) {
  allocator->template defrag_scan<T>(num_records);
}

template<typename T>
T copy_from_device(T* device_ptr) {
  T result;
  cudaMemcpy(&result, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
  return result;
}

#endif  // ALLOCATOR_SOA_UTIL_H
