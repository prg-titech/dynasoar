#define NDEBUG
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024

__device__ int x;
__global__ void dummy_kernel() {
  x = 1;
}

__global__ void  benchmark(int num_iterations, void** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  void** my_ptrs = ptrs + tid*num_iterations;

  for (int i = 0; i < num_iterations; ++i) {
    my_ptrs[i] = malloc(8);
  }

  for (int i = 0; i < num_iterations; ++i) {
    free(my_ptrs[i]);
  }
}


int main() {
  void** ptr_storage;
  cudaMalloc((void**) &ptr_storage, sizeof(void*)*64*64*64*64);
  gpuErrchk(cudaDeviceSynchronize());

  // INIT MEMORY ALLOCATOR
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256U*1024U*1024U);

  dummy_kernel<<<64, 64>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto time_before = std::chrono::system_clock::now();
  benchmark<<<64, 256>>>(NUM_ALLOCS, ptr_storage);
  gpuErrchk(cudaDeviceSynchronize());
  auto time_after = std::chrono::system_clock::now();
  int time_running = std::chrono::duration_cast<std::chrono::microseconds>(
      time_after - time_before).count();
  printf("%i,%i\n", NUM_ALLOCS, time_running);
}

