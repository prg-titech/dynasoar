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


#include "allocator/soa_allocator.h"

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024

class DummyClass {
 public:
  static const uint8_t kTypeId = 0;
  static const int kObjectSize = ALLOC_SIZE;
  static const uint8_t kBlockSize = 64;

  SoaField<int, 0, 0> var;
};

__device__ SoaAllocator<64*64*64*64, DummyClass> memory_allocator;

__global__ void  benchmark(int num_iterations, DummyClass** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  DummyClass** my_ptrs = ptrs + tid*num_iterations;

  for (int k = 0; k < 1; ++k) {
    for (int i = 0; i < num_iterations; ++i) {
      my_ptrs[i] = memory_allocator.make_new<DummyClass>();
      //my_ptrs[i]->var = 1234;
    }

    for (int i = 0; i < num_iterations; ++i) {
      memory_allocator.free(my_ptrs[i]);
    }
  }
}

__device__ void initialize_allocator() {
  memory_allocator.initialize();
}

__global__ void init_memory_system() {
  initialize_allocator();
}

__device__ int x;
__global__ void dummy_kernel() {
  x = 1;
}

int main() {
  DummyClass** ptr_storage;
  cudaMalloc((void**) &ptr_storage, sizeof(void*)*64*64*64*64);
  gpuErrchk(cudaDeviceSynchronize());

  init_memory_system<<<256, 512>>>();
  gpuErrchk(cudaDeviceSynchronize());

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
