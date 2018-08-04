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
#include "mallocMC/mallocMC.hpp"

#define uint32 uint32_t
typedef unsigned int uint;

using namespace mallocMC;

struct ScatterHeapConfig : mallocMC::CreationPolicies::Scatter<>::HeapProperties{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashConfig : mallocMC::CreationPolicies::Scatter<>::HashingProperties{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterHeapConfig::pagesize pagesize;
};

typedef mallocMC::Allocator<
  CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
  DistributionPolicies::XMallocSIMD<DistributionConfig>,
  OOMPolicies::ReturnNull,
  ReservePoolPolicies::SimpleCudaMalloc,
  AlignmentPolicies::Noop
> ScatterAllocator;

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024

class DummyClass {
 public:
  static const uint8_t kTypeId = 0;
  static const int kObjectSize = ALLOC_SIZE;
  static const uint8_t kBlockSize = 64;

  char one_field;
};

__device__ int x;
__global__ void dummy_kernel() {
  x = 1;
}

__device__ ScatterAllocator::AllocatorHandle alloc_handle;

__global__ void  benchmark(int num_iterations, void** ptrs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  void** my_ptrs = ptrs + tid*num_iterations;

  for (int k = 0; k < 1; ++k) {
    for (int i = 0; i < num_iterations; ++i) {
      DummyClass* p =  (DummyClass*) (alloc_handle.malloc(ALLOC_SIZE));
      my_ptrs[i] = p;
      if (p == nullptr) {
        asm("trap;");
      }
      assert(my_ptrs[i] != nullptr);
      //*reinterpret_cast<int*>(my_ptrs[i]) = 1234;
    }

    for (int i = 0; i < num_iterations; ++i) {
      alloc_handle.free(my_ptrs[i]);
    }
  }
}


__global__ void copy_handle(ScatterAllocator::AllocatorHandle handle) {
  alloc_handle = handle;
}

void initHeap(int bytes) {
  auto* sa = new ScatterAllocator( 256U *1024U * 1024U ); // heap size of 512MiB
  copy_handle<<<1,1>>>(*sa);
  gpuErrchk(cudaDeviceSynchronize());
}

int main() {
  void** ptr_storage;
  cudaMalloc((void**) &ptr_storage, sizeof(void*)*64*64*64*64);
  gpuErrchk(cudaDeviceSynchronize());

  // INIT MEMORY ALLOCATOR
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256U*1024U*1024U);
  initHeap(1024U*1024*256);

  dummy_kernel<<<64, 64>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto time_before = std::chrono::system_clock::now();
  benchmark<<<64, 256>>>(NUM_ALLOCS, ptr_storage);
  gpuErrchk(cudaDeviceSynchronize());
  auto time_after = std::chrono::system_clock::now();
  int time_running = std::chrono::duration_cast<std::chrono::microseconds>(
      time_after - time_before).count();
  printf("%i,%i,%i\n", NUM_ALLOCS, ALLOC_SIZE, time_running);
}

