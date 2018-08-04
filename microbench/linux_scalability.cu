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


#include <cuda.h>

#define uint32 uint32_t
typedef unsigned int uint;


#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024

class DummyClass {
 public:
  static const uint8_t kTypeId = 0;
  static const int kObjectSize = ALLOC_SIZE;
  static const uint8_t kBlockSize = 64;

  char one_field;
};


__global__ void  benchmark(int num_iterations, void** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  void** my_ptrs = ptrs + tid*num_iterations;

  for (int k = 0; k < 1; ++k) {
    for (int i = 0; i < num_iterations; ++i) {
      DummyClass* p = (DummyClass*) malloc(ALLOC_SIZE); //new(alloc_handle.malloc(ALLOC_SIZE)) DummyClass();
      my_ptrs[i] = p;
      if (p == nullptr) {
        asm("trap;");
      }

      assert(my_ptrs[i] != nullptr);
      //*reinterpret_cast<int*>(my_ptrs[i]) = 1234;
    }

    for (int i = 0; i < num_iterations; ++i) {
      free(my_ptrs[i]);
    }
  }
}



int main() {
  void** ptr_storage;
  cudaMalloc((void**) &ptr_storage, sizeof(void*)*64*64*64*64);
  gpuErrchk(cudaDeviceSynchronize());

  // INIT MEMORY ALLOCATOR
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256U*1024U*1024U);


  auto time_before = std::chrono::system_clock::now();
  benchmark<<<64, 256>>>(NUM_ALLOCS, ptr_storage);
  gpuErrchk(cudaDeviceSynchronize());
  auto time_after = std::chrono::system_clock::now();
  int time_running = std::chrono::duration_cast<std::chrono::microseconds>(
      time_after - time_before).count();
  printf("%i,%i,%i\n", NUM_ALLOCS, ALLOC_SIZE, time_running);
}

