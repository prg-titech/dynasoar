#include "dynasoar.h"

#ifdef PARAM_HEAP_SIZE
static const int kHeapSize = PARAM_HEAP_SIZE;
#else
static const int kHeapSize = 64*64*64;
#endif  // PARAM_HEAP_SIZE


// Pre-declare all classes.
class Foo;
class Bar;
class BarSub;


// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<kHeapSize, Bar, Foo, BarSub>;


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

#if GCC_COMPILER
class Bar : public AllocatorT::Base {
#else
// Workaround for compiler other than GCC. Will fix this in future versions.
class Bar : public SoaBase<AllocatorT> {
#endif  // GCC_COMPILER
 public:
  // Pre-declare types of all fields.
  declare_field_types(Bar, Foo*, int)

  __device__ Bar(Foo* f0, int f1) : field0_(f0), field1_(f1) {}

  Field<Bar, 0> field0_;
  Field<Bar, 1> field1_;

  __device__ void foo(int v);

  __device__ void assert_result() {
    if (field1_ != 1 + 2 + 3 + 4 + 5) {
      printf("Incorrect result!\n");
      asm("trap;");  // Force kernel to quit.
    }
  }
};


class BarSub : public Bar {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = Bar;

  __device__ BarSub(int index) : Bar(nullptr, 2), field2_(index) {}

  // Pre-declare types of all fields.
  declare_field_types(Bar, int)

  Field<BarSub, 0> field2_;
};


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
 
  __device__ int qux() {
    return field0_ + field1_ + field2_;
  }
};


__device__ void Bar::foo(int v) {
  field1_ += field0_->qux() + v;
}


__global__ void create_objects() {
  auto* f = new(device_allocator) Foo(1, 2, 3);
  // Delete objects with: destroy(device_allocator, ptr)

  new(device_allocator) Bar(f, 4);
}


int main(int /*argc*/, char** /*argv*/) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Create 2048 objects of Foo and Bar.
  create_objects<<<32, 64>>>();
  cudaDeviceSynchronize();

  // Call Bar::foo on all 2048 objects.
  allocator_handle->parallel_do<Bar, int, &Bar::foo>(5);

  // Check correctness.
  allocator_handle->parallel_do<Bar, &Bar::assert_result>();

  // Not checking anything in particular. Just make sure that there's no crash.
  allocator_handle->parallel_do_single_type<BarSub, Bar, &Bar::assert_result>();
  // Now create some BarSub objects. But don't run assert_result of them, as the
  // computation would fail.
  allocator_handle->parallel_new<BarSub>(32);
  allocator_handle->parallel_do_single_type<Bar, Bar, &Bar::assert_result>();

  printf("Check passed!\n");
  return 0;
}
