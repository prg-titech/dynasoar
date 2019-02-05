#include <curand_kernel.h>
#include <limits>

#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

#include "configuration.h"

static const int kIntMax = std::numeric_limits<int>::max();

// Pre-declare all classes.
class C1;
class C2;

// Declare allocator type.
using AllocatorT = SoaAllocator<16*64*64*64*64, C1, C2>;

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

// 32 byte objects.
class C1 : public SoaBase<AllocatorT> {
 public:
  declare_field_types(C1, C2*, int, int, int, int, int, int)

  SoaField<C1, 0> other_;
  SoaField<C1, 1> id_;
  SoaField<C1, 2> rand_num_;
  SoaField<C1, 3> int3_;
  SoaField<C1, 4> int4_;
  SoaField<C1, 5> int5_;
  SoaField<C1, 6> int6_;

  __device__ C1(int id, int rand_num)
      : id_(id), rand_num_(rand_num), other_(nullptr) {}
};

// 32 byte objects.
class C2 : public SoaBase<AllocatorT> {
 public:
  declare_field_types(C2, C1*, int, int, int, int, int, int)

  SoaField<C2, 0> other_;
  SoaField<C2, 1> id_;
  SoaField<C2, 2> rand_num_;
  SoaField<C2, 3> int3_;
  SoaField<C2, 4> int4_;
  SoaField<C2, 5> int5_;
  SoaField<C2, 6> int6_;

  __device__ C2(int id, int rand_num)
      : id_(id), rand_num_(rand_num), other_(nullptr) {}

  __device__ void maybe_destroy_object() {
    if (rand_num_ % kRetainFactor == 0) {
      if (other_ != nullptr) {
        other_->other_ = nullptr;
      }

      destroy(device_allocator, this);
    }
  }
};

__device__ C1* ptr_c1[kSize];
__device__ C2* ptr_c2[kSize];

__global__ void create_objects() {
  curandState_t random_state;
  curand_init(43, threadIdx.x + blockDim.x*blockIdx.x,
              0, &random_state);

  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSize; i += blockDim.x * gridDim.x) {
    ptr_c1[i] = new(device_allocator) C1(i, curand(&random_state) % kIntMax);
    ptr_c2[i] = new(device_allocator) C2(i, curand(&random_state) % kIntMax);
  }
}

__global__ void set_pointers() {
  curandState_t random_state;
  curand_init(42, threadIdx.x + blockDim.x*blockIdx.x,
              0, &random_state);

  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSize; i += blockDim.x * gridDim.x) {
    int other_idx = curand(&random_state) % kSize;
    ptr_c2[i]->other_ = ptr_c1[other_idx];
    ptr_c1[other_idx]->other_ = ptr_c2[i];
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

  set_pointers<<<512, 512>>>();
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
