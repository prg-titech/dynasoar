#include <chrono>
#include <stdio.h>

#include "linux_scalability.h"

// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


__global__ void  kernel_benchmark(int num_alloc, DummyClass** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  DummyClass** my_ptrs = ptrs + tid*num_alloc;

  for (int i = 0; i < num_alloc; ++i) {
    DummyClass* p = new(device_allocator) DummyClass();
    my_ptrs[i] = p;
    if (p == nullptr) {
      asm("trap;");
    }
  }
}


__global__ void  kernel_benchmark_2(int num_alloc, DummyClass** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  DummyClass** my_ptrs = ptrs + tid*num_alloc;

  for (int i = 0; i < num_alloc; ++i) {
    destroy(device_allocator, my_ptrs[i]);
  }
}



int main() {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  DummyClass** ptrs;
  cudaMalloc(
      &ptrs, sizeof(DummyClass*)*kNumBlocks*kNumThreads*kNumAllocPerThread);
  gpuErrchk(cudaDeviceSynchronize());

  auto time_before = std::chrono::system_clock::now();

  // Run benchmark.
  for (int i = 0; i < kNumIterations; ++i) {
    kernel_benchmark<<<kNumBlocks, kNumThreads>>>(kNumAllocPerThread, ptrs);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_benchmark_2<<<kNumBlocks, kNumThreads>>>(kNumAllocPerThread, ptrs);
    gpuErrchk(cudaDeviceSynchronize());
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_before;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

  printf("%lu, %lu\n", micros, allocator_handle->DBG_get_enumeration_time());

  return 0;
}