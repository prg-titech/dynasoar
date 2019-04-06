#include "dynasoar.h"

class Foo;
class Bar;
using AllocatorT = SoaAllocator</*num_objs=*/ 262144, Foo, Bar>;


__device__ AllocatorT* device_allocator;        // device side
AllocatorHandle<AllocatorT>* allocator_handle;  // host side


class Bar : public AllocatorT::Base {
 public:
  declare_field_types(Bar, Foo*, int, int)

 private:
  Field<Bar, 0> the_first_field_;
  Field<Bar, 1> the_second_field_;
  Field<Bar, 2> the_third_field_;

 public:
  __device__ Bar(int a, int b)
      : the_first_field_(nullptr), the_second_field_(a), the_third_field_(b) {}

  __device__ void increment_by_one() {
    the_second_field_ += 1;
  }

  __device__ void increment_by_n(int n) {
    the_second_field_ += n;
  }

  __device__ void print_second() {
    printf("Second value: %i\n", (int) the_second_field_);
  }
};


class Foo : public AllocatorT::Base {
 public:
  declare_field_types(Foo, int, int, int)

 private:
  Field<Bar, 0> f0_;
  Field<Bar, 1> f1_;
  Field<Bar, 2> f2_;
};


__global__ void create_objs() {
  Bar* result = new(device_allocator) Bar(threadIdx.x, 5);
}


int main(int argc, char** argv) {
  // Some boilerplate code.... Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Allocate a few objects.
  create_objs<<<5, 10>>>();
  cudaDeviceSynchronize();

  // Run a do-all operations in parallel.
  allocator_handle->parallel_do<Bar, &Bar::increment_by_one>();

  // If a member function takes an argument, we have to specify its type here.
  allocator_handle->parallel_do<Bar, int, &Bar::increment_by_n>(/*n=*/ 10);

  // Now print some stuff.
  allocator_handle->parallel_do<Bar, &Bar::print_second>();
}
