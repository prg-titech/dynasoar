#include "allocator/soa_allocator.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class DummyClass1 {
 public:
  static const int kObjectSize = 13;
};

class DummyClass2 {
 public:
  static const int kObjectSize = 16;
};

__device__ SoaAllocator<64*64*64, DummyClass1, DummyClass2> allocator;

__global__ void initialize_alloc() {
  allocator.initialize();
}

__global__ void simple_test() {
  DummyClass1* x = allocator.make_new<DummyClass1>();
  printf("ALLOC: %p\n", x);
}

int main() {
  initialize_alloc<<<16, 16>>>();
  gpuErrchk(cudaThreadSynchronize());

  simple_test<<<16, 16>>>();
  gpuErrchk(cudaThreadSynchronize());
}