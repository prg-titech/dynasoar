#include <curand_kernel.h>
#include <limits>
#include <algorithm>
#include <random>

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

__device__ unsigned long long int d_checksum;

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

  __device__ void compute_checksum();
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
    if (rand_num_ % kDeleteFactor == 0) {
      if (other_ != nullptr) {
        other_->other_ = nullptr;
      }

      destroy(device_allocator, this);
    }
  }
};

__device__ void C1::compute_checksum() {
  if (other_ != nullptr) {
    atomicAdd(&d_checksum,  (id_ * other_->id_) % 97);
  }
}

__global__ void kernel_create_objects(C1** ptr_c1, C2** ptr_c2) {
  curandState_t random_state;
  curand_init(43, threadIdx.x + blockDim.x*blockIdx.x,
              0, &random_state);

  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSize; i += blockDim.x * gridDim.x) {
    ptr_c1[i] = new(device_allocator) C1(i, curand(&random_state) % kIntMax);
    ptr_c2[i] = new(device_allocator) C2(i, curand(&random_state) % kIntMax);
  }
}

size_t h_assoc[kSize];

__global__ void kernel_set_pointers(C1** ptr_c1, C2** ptr_c2,
                                    size_t* d_assoc) {
  curandState_t random_state;
  curand_init(42, threadIdx.x + blockDim.x*blockIdx.x,
              0, &random_state);

  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSize; i += blockDim.x * gridDim.x) {
    ptr_c2[i]->other_ = ptr_c1[d_assoc[i]];
    ptr_c1[d_assoc[i]]->other_ = ptr_c2[i];
  }
}

void set_pointers(C1** ptr_c1, C2** ptr_c2) {
  for (size_t i = 0; i < kSize; ++i) {
    h_assoc[i] = i;
  }
  shuffle(h_assoc, h_assoc + kSize, std::default_random_engine(42));

  size_t* d_assoc;
  cudaMalloc(&d_assoc, sizeof(size_t)*kSize);
  cudaMemcpy(d_assoc, h_assoc, sizeof(size_t)*kSize,
             cudaMemcpyHostToDevice);

  kernel_set_pointers<<<512, 512>>>(ptr_c1, ptr_c2, d_assoc);
  gpuErrchk(cudaDeviceSynchronize());
}

int main(int /*argc*/, char** /*argv*/) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  C1** d_ptr_c1;
  C2** d_ptr_c2;
  gpuErrchk(cudaMalloc(&d_ptr_c1, sizeof(C1*)*kSize));
  gpuErrchk(cudaMalloc(&d_ptr_c2, sizeof(C2*)*kSize));

  // Create objects.
  kernel_create_objects<<<512, 512>>>(d_ptr_c1, d_ptr_c2);
  gpuErrchk(cudaDeviceSynchronize());

  set_pointers(d_ptr_c1, d_ptr_c2);

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
  allocator_handle->DBG_collect_stats();
  allocator_handle->DBG_print_collected_stats();
#endif  // OPTION_DEFRAG

  // Compute checksum.
  unsigned long long int h_checksum = 0;
  cudaMemcpyToSymbol(d_checksum, &h_checksum, sizeof(unsigned long long int),
                     0, cudaMemcpyHostToDevice);

  allocator_handle->parallel_do<C1, &C1::compute_checksum>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&h_checksum, d_checksum, sizeof(unsigned long long int),
                       0, cudaMemcpyDeviceToHost);
  printf("Checksum: %llu\n", h_checksum);

  printf("%i\n", total_time);
}
