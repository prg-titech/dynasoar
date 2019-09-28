#include "dynasoar.h"

#ifdef PARAM_HEAP_SIZE
static const int kHeapSize = PARAM_HEAP_SIZE;
#else
static const int kHeapSize = 64*64*64;
#endif  // PARAM_HEAP_SIZE


// Pre-declare all classes.
class Foo;


// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<kHeapSize, Foo>;


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

__device__ Foo* selected_object;

int h_accumulator = 0;

#if GCC_COMPILER
class Foo : public AllocatorT::Base {
#else
// Workaround for compiler other than GCC. Will fix this in future versions.
class Foo : public SoaBase<AllocatorT> {
#endif  // GCC_COMPILER
 public:
  // Pre-declare types of all fields.
  declare_field_types(Foo, int, int, int)
  
  // Declare fields.
  SoaField<Foo, 0> field0_;  // int
  SoaField<Foo, 1> field1_;  // int
  SoaField<Foo, 2> field2_;  // int
  
  __device__ Foo(int f0, int f1, int f2)
      : field0_(f0), field1_(f1), field2_(f2) {}
 
  __device__ __host__ int qux() {
    return field0_ + field1_ + field2_;
  }

  void add_to_accumulator() {
    h_accumulator += qux();
  }
};


__global__ void create_objects() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  auto* f = new(device_allocator) Foo(tid, 2, 3);

  if (tid == 35) {
    // Select object created by thread 35.
    selected_object = f;
  }
}


int main(int /*argc*/, char** /*argv*/) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Create 2048 objects of Foo.
  create_objects<<<32, 64>>>();
  cudaDeviceSynchronize();

  Foo* host_selected_object;
  cudaMemcpyFromSymbol(&host_selected_object, selected_object, sizeof(Foo*), 0,
                       cudaMemcpyDeviceToHost);

  int result = host_selected_object->qux();
  if (result == 35 + 2 + 3) {
    printf("First check passed!\n");
  } else {
    printf("ERROR: First check failed! Expected 35, but got %i\n", result);
    return 1;
  }

  allocator_handle->device_pointer()->template device_do<Foo>(&Foo::add_to_accumulator);
  int expected = 2048*(2+3) + 2048*2047/2;
  if (h_accumulator == expected) {
    printf("Second check passed!\n");
  } else {
    printf("ERROR: Second check failed! Expected %i, but got %i\n",
           expected, h_accumulator);
    return 1;
  }

  return 0;
}
