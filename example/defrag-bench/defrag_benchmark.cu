#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

#include "configuration.h"

// Pre-declare all classes.
class C1;
class C2;

// Declare allocator type.
using AllocatorT = SoaAllocator<8*64*64*64*64, C1, C2>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

// 32 byte objects.
class C1 : public SoaBase<AllocatorT> {
 public:
  declare_field_types(C1, C2*, int, int, int, int, int, int)

  SoaField<C1, 0> other_;
  SoaField<C1, 1> id_;
  SoaField<C1, 2> int2_;
  SoaField<C1, 3> int3_;
  SoaField<C1, 4> int4_;
  SoaField<C1, 5> int5_;
  SoaField<C1, 6> int6_;

  __device__ C1(int id) : id_(id) {}
};

// 32 byte objects.
class C2 : public SoaBase<AllocatorT> {
 public:
  declare_field_types(C2, C1*, int, int, int, int, int, int)

  SoaField<C2, 0> other_;
  SoaField<C2, 1> id_;
  SoaField<C2, 2> int2_;
  SoaField<C2, 3> int3_;
  SoaField<C2, 4> int4_;
  SoaField<C2, 5> int5_;
  SoaField<C2, 6> int6_;

  __device__ C2(int id) : id_(id) {}

  __device__ void maybe_destroy_object() {
    if (id_ % kRetainFactor != 0) {
      destroy(device_allocator, this);
    }
  }
};

__global__ void create_objects() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSize; i += blockDim.x * gridDim.x) {
    C1* c1 = new(device_allocator) C1(i);
    C2* c2 = new(device_allocator) C2(i);
    c1->other_ = c2;
    c2->other_ = c1;
  }
}

int main(int /*argc*/, char** /*argv*/) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Create objects.
  create_objects<<<512, 512>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Destroy some objects.
  allocator_handle->parallel_do<C2, &C2::maybe_destroy_object>();
  gpuErrchk(cudaDeviceSynchronize());

  int total_time = 0;
  auto time_before = std::chrono::system_clock::now();

#ifdef OPTION_DEFRAG
  // Defragment C2.
  allocator_handle->parallel_defrag<C2>();
#endif  // OPTION_DEFRAG

  auto time_after = std::chrono::system_clock::now();
  int time_running = std::chrono::duration_cast<std::chrono::milliseconds>(
      time_after - time_before).count();
  total_time += time_running;

#ifdef OPTION_DEFRAG
  allocator_handle->DBG_print_defrag_time();
#endif  // OPTION_DEFRAG

  printf("%i\n", total_time);
}
