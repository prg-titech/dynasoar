#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

// Pre-declare all classes.
class Foo;
class Bar;

// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<64*64*64*64, Foo, Bar>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

class Foo : public SoaBase<AllocatorT> {
 public:
  // Pre-declare types of all fields.
  using FieldTypes = std::tuple<float, int, char>;
  
  // Declare fields.
  SoaField<Foo, 0> field1_;  // float
  SoaField<Foo, 1> field2_;  // int
  SoaField<Foo, 2> field3_;  // char
  
  __device__ Foo(float f1, int f2, char f3) : field1_(f1), field2_(f2), field3_(f3) {}
 
  __device__ void qux() {
    field1_ = field2_ + field3_;
  }
};

class Bar : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<int, int, int>;
};

__global__ void create_objects() {
  device_allocator->make_new<Foo>(1.0f, threadIdx.x, 2);
  // Delete objects with: device_allocator->free<Foo>(ptr)
}

int main(int argc, char** argv) {
  // Optional, for debugging.
  AllocatorT::DBG_print_stats();
  
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Create 2048 objects.
  create_objects<<<32, 64>>>();
  cudaDeviceSynchronize();

  // Call Foo::qux on all 2048 objects.
  allocator_handle->parallel_do<Foo, &Foo::qux>();
}
