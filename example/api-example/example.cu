#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

// Pre-declare all classes.
class Foo;
class Bar;

// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<64*64*64*64, Bar, Foo>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

class Bar : public SoaBase<AllocatorT> {
 public:
  declare_field_types(Bar, int, int, int)

  SoaField<Bar, 0> field1_;
  SoaField<Bar, 1> field2_;
  SoaField<Bar, 2> field3_;

  __device__ void foo(int v) {
    field1_ += v;
  }
};

class Foo : public SoaBase<AllocatorT> {
 public:
  // Pre-declare types of all fields.
  declare_field_types(Foo, float, int, char)
  
  // Declare fields.
  SoaField<Foo, 0> field1_;  // float
  SoaField<Foo, 1> field2_;  // int
  SoaField<Foo, 2> field3_;  // char
  
  __device__ Foo(float f1, int f2, char f3) : field1_(f1), field2_(f2), field3_(f3) {}
 
  __device__ void qux() {
    field1_ = field2_ + field3_;
  }

  __device__ void baz() {
    // Run in Bar::foo(42) sequentially for all objects of type Bar. Note that
    // Foo::baz may run in parallel though.
    device_allocator->template device_do<Bar>(&Bar::foo, 42);
  }
};

__global__ void create_objects() {
  new(device_allocator) Foo(1.0f, threadIdx.x, 2);
  // Delete objects with: destroy(device_allocator, ptr)
}

int main(int /*argc*/, char** /*argv*/) {
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
