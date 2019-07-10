# DynaSOAr: A CUDA Framework for Single-Method Multiple-Objects Applications
SMMO (Single-Method Multiple-Objects) is a wide-spread pattern of parallel, object-oriented, high-performance code. It is OOP-speech for SIMD (Single-Instruction Multiple-Data) and means that a method should be executed for all objects of a type. 

DynaSOAr comes with a parallel, lock-free, dynamic memory allocator that lets programmers create/delete objects in device code. In contrast to other allocators, this allocator is an *object allocator* for structured data. While other allocators allocate *X number of bytes*, this allocator can only allocate objects of C++ classes/structs that were defined within DynaSOAr. This allows us to apply additional data layout optimizations.

As an example, an nbody simulation consists of `n` body objects, for each of which a `move` method for computing the next position of a body should be executed. DynaSOAr is a CUDA framework (C++ template library) that facilitates the development of such programs. The four main features of DynaSOAr are:

* **SOA Data Layout:** Objects are stored in the SIMD-friendly Structure of Arrays data layout. Other layouts may be supported in the future.
* **Dynamic Memory Management on Device:** New objects can be created at any time in the CUDA kernel and existing objects can be deleted (`new`/`delete`).
* **Parallel Enumeration:** DynaSOAr provides an efficient way to run a member function (method) for all objects of a type in parallel.
* **Memory Defragmentation:** Can lower overall memory usage and speed up application code in the object space becomes to fragmented.

## Documentation/Papers
* Matthias Springer. [CompactGpu: Massivey GPU Memory Compaction](http://m-sp.org/downloads/pldi_2019_src.pdf). ACM Student Research Competition (PLDI 2019).
* Matthias Springer, Hidehiko Masuhara. [Massively Parallel GPU Memory Compaction](http://m-sp.org/downloads/ismm2019.pdf). ISMM 2019.
* Matthias Springer, Hidehiko Masuhara. [DynaSOAr: A Parallel Memory Allocator for Object-oriented Programming on GPUs with Efficient Memory Access](http://drops.dagstuhl.de/opus/volltexte/2019/10809/). ECOOP 2019.
* Matthias Springer. [SoaAlloc: Accelerating Single-Method Multiple-Objects Applications on GPUs](https://arxiv.org/pdf/1809.07444). ACM Student Research Competition (SPLASH 2018).
* Matthias Springer, Yaozhu Sun, Hidehiko Masuhara. [Inner Array Inlining for Structure of Arrays Layout](http://m-sp.org/downloads/array2018.pdf). ARRAY 2018.
* Matthias Springer, Hidehiko Masuhara. [Ikra-Cpp: A C++/CUDA DSL for Object-oriented Programming with Structure-of-Arrays Data Layout](http://m-sp.org/downloads/wpmvp2018.pdf). WPMVP 2018.
* Matthias Springer. [A C++/CUDA DSL for Object-oriented Programming with Structure-of-Arrays Data Layout](http://m-sp.org/downloads/cgo2018-src-abstract.pdf). ACM Student Research Competition (CGO 2018).

## Prerequisites
Tested with the CUDA Toolkit versions 9.1 (gcc version 5.4.0) and 10.1 on an Nvidia Titan Xp machine (Ubuntu 16.04.1). A device with a minimum compute capability of 5.0 is required. `libsdl2` is required for graphical visualizations in example code. We provide build scripts for compiling the examples. See the [Wiki](https://github.com/prg-titech/dynasoar/wiki/Benchmark-Applications) for more information.


## API Overview
All classes/structs that should be managed by DynaSOAr must inherit from `AllocatorT::Base`, where `AllocatorT` is the fully configured type of the allocator. The first template argument to `SoaAllocator` is the maximum number of objects that can exist within the allocator at any given time; this number determines the memory usage of the allocator. The following arguments are all classes/structs that are managed by DynaSOAr.

DynaSOAr has a host side API (`AllocatorHandle<AllocatorT>`) and a device side API (`AllocatorT`). The following functionality is provided with those APIs.
* `AllocatorHandle::AllocatorHandle()`: This constructor allocates all necessary memory on GPU.
* `AllocatorHandle::device_pointer()`: Returns a pointer to the device allocator handle (`AllocatorT*`).
* `AllocatorHandle::parallel_do<C, &C::foo>()`: Runs a member function `C::foo()` in parallel for all objects of type `C` that were created with the allocator. This will launch a CUDA kernel. This function returns when the CUDA kernel has finished processing all objects.
* `new(allocator) (args...)`: Creates a new object of type `C` and returns a pointer to the new object. `C` must be a type that is managed by the allocator. `allocator` is a pointer to the device allocator.
* `destroy(allocator, ptr)`: Deletes an existing object `ptr` of type `C` that was created with the allocator. This is similar to C++ `delete`.
* `AllocatorT::device_do<C>(&C::foo, args...)`: Runs `C::foo(args...)` for all objects of type `C` that were created with the allocator. Note that this does not spawn a new CUDA kernel; execution is sequential.
* `AllocatorT::parallel_defrag<C>()`: Runs a defragmentation pass on all objects of type `C`. This pass compacts objects and rewrites pointers to object that were relocated.

## API Example
This example does not compute anything meaningful and is only meant to show the API. Take a look at the [DynaSOAr tutorial](https://github.com/prg-titech/dynasoar/wiki/Tutorial) and at the code in the `example` directory for [more interesting examples](https://github.com/prg-titech/dynasoar/wiki/Benchmark-Applications).

```c++
#include "dynasoar.h"

// Pre-declare all classes.
class Foo;
class Bar;

// Declare allocator type. First argument is max. number of objects that can be created.
using AllocatorT = SoaAllocator<64*64*64*64, Foo, Bar>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

class Foo : public AllocatorT::Base {
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

class Bar : public SoaBase<AllocatorT> { /* ... */ };

__global__ void create_objects() {
  Foo* object = new(device_allocator) Foo(1.23f, 4, 0);
  // Delete object: destroy(device_allocator, object);
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
