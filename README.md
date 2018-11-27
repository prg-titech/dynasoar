# SoaAlloc: A CUDA Framework for Single-Method Multiple-Objects Applications
SMMO (Single-Method Multiple-Objects) is a wide-spread pattern of parallel, object-oriented, high-performance code. It is OOP-speech for SIMD (Single-Instruction Multiple-Data) and means that a method should be executed for multiple or all objects of a type. As an example, an nbody simulation consists of `n` body objects, for each of which a `move` method for computing the next position of a body should be executed. SoaAlloc is a CUDA framework (C++ template library) that facilitates the development of such programs. The three main features of SoaAlloc are:

* SOA Data Layout: Objects are stored in the SIMD-friendly Structure of Arrays data layout.
* Dynamic Memory Management on Device: New objects can be created at any time in the CUDA kernel and existing objects can be destructed (`malloc`/`free`).
* Parallel Enumeration: SoaAlloc provides an efficient way to run a member function (method) for all objects of a type in parallel.

## Documentation/Papers
* Matthias Springer. [SoaAlloc: Accelerating Single-Method Multiple-Objects Applications on GPUs](https://arxiv.org/pdf/1809.07444). ACM Student Research Competition (SPLASH 2018).
* Matthias Springer, Hidehiko Masuhara. [SoaAlloc: A Lock-free Hierarchical Bitmap-based Object Allocator for GPUs](https://arxiv.org/pdf/1810.11765). ArXiv Preprint, 2018.
* Matthias Springer, Yaozhu Sun, Hidehiko Masuhara. [Inner Array Inlining for Structure of Arrays Layout](http://m-sp.org/downloads/array2018.pdf). ARRAY 2018.
* Matthias Springer, Hidehiko Masuhara. [Ikra-Cpp: A C++/CUDA DSL for Object-oriented Programming with Structure-of-Arrays Data Layout](http://m-sp.org/downloads/wpmvp2018.pdf). WPMVP 2018.
* Matthias Springer. [A C++/CUDA DSL for Object-oriented Programming with Structure-of-Arrays Data Layout](http://m-sp.org/downloads/cgo2018-src-abstract.pdf). ACM Student Research Competition (CGO 2018).

## Prerequisites
Tested with CUDA Toolkit 9.1 on a Nvidia Titan Xp machine (Ubuntu 16.04.1).

## API Example
This example does not compute anything meaningful and is only meant to show the API. Take a look at the code in the `example` directory for more interesting examples.

```c++
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

class Bar : public SoaBase<AllocatorT> { /* ... */ };

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
```
