#ifndef EXAMPLE_CONFIGURATION_BITMAP_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_BITMAP_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED


#include "bitmap/bitmap.h"
#include "../allocator_interface_adapter.h"


template<typename AllocatorT, typename AllocatorStateT>
__global__ void kernel_bitmap_alloc_init_bitmap(AllocatorStateT* state) {
  state->global_free.initialize(true);

  for (int i = 0; i < AllocatorT::kNumTypes; ++i) {
    state->allocated[i].initialize(false);
  }
}


template<class T, class BaseClass, void(BaseClass::*func)(),
         typename AllocatorStateT, typename BitmapT>
__global__ void kernel_bitmap_parallel_do_single_type(
    AllocatorStateT* state, BitmapT* bitmap) {
  const int num_objs = bitmap->scan_num_bits();
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_objs; i += blockDim.x * gridDim.x) {
    auto pos = bitmap->scan_get_index(i);
    T* obj = reinterpret_cast<T*>(
        state->data_storage + pos*AllocatorStateT::kObjectSize);
    (obj->*func)();
  }
}


template<class T, class BaseClass, typename P1, void(BaseClass::*func)(P1),
         typename AllocatorStateT, typename BitmapT>
__global__ void kernel_bitmap_parallel_do_single_type1(
    AllocatorStateT* state, BitmapT* bitmap, P1 p1) {
  const int num_objs = bitmap->scan_num_bits();
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_objs; i += blockDim.x * gridDim.x) {
    auto pos = bitmap->scan_get_index(i);
    T* obj = reinterpret_cast<T*>(
        state->data_storage + pos*AllocatorStateT::kObjectSize);
    (obj->*func)(p1);
  }
}


template<typename T, typename F, typename AllocatorStateT, typename... Args>
struct BitmapSequentialExecutor {
  __device__ static void device_do(uint32_t pos, F func,
                                   AllocatorStateT* state, Args... args) {
    T* obj = reinterpret_cast<T*>(
        state->data_storage + pos*AllocatorStateT::kObjectSize);
    (obj->*func)(args...);
  }
};


template<typename AllocatorT>
struct AllocatorState {
  static const bool kHasParallelDo = true;
  static const int kObjectSize = AllocatorT::kLargestTypeSize;

  char* data_storage;
  Bitmap<uint32_t, AllocatorT::kMaxObjects> global_free;
  Bitmap<uint32_t, AllocatorT::kMaxObjects> allocated[AllocatorT::kNumTypes];

  void initialize(size_t /*allocator_size*/) {  // ignored
    printf("%i\n\n", (int) kObjectSize);
    char* host_data_storage;
    cudaMalloc(&host_data_storage, 3ULL*kMallocHeapSize/4);
    assert(host_data_storage != nullptr);
    cudaMemcpy(&data_storage, &host_data_storage, sizeof(char*),
               cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_bitmap_alloc_init_bitmap<AllocatorT><<<128, 128>>>(this);
    gpuErrchk(cudaDeviceSynchronize());
  }

  template<class T, class BaseClass, void(BaseClass::*func)(), bool Scan>
  void parallel_do_single_type() {
    auto time_start = std::chrono::system_clock::now();

    const auto type_index = AllocatorT::template TypeHelper<T>::kIndex;
    if (Scan) {
      allocated[type_index].scan();
    }

    // Determine number of CUDA threads.
    uint32_t* d_num_obj_ptr = allocated[type_index].scan_num_bits_ptr();
    uint32_t num_obj = copy_from_device(d_num_obj_ptr);

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
        .count();
    bench_prefix_sum_time += micros;

    if (num_obj > 0) {
      kernel_bitmap_parallel_do_single_type<T, BaseClass, func><<<
          (num_obj + 256 - 1)/256, 256>>>(
              this, &allocated[AllocatorT::template TypeHelper<T>::kIndex]);
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template<class T, class BaseClass, typename P1, void(BaseClass::*func)(P1),
           bool Scan>
  void parallel_do_single_type(P1 p1) {
    auto time_start = std::chrono::system_clock::now();

    const auto type_index = AllocatorT::template TypeHelper<T>::kIndex;
    if (Scan) {
      allocated[type_index].scan();
    }

    // Determine number of CUDA threads.
    uint32_t* d_num_obj_ptr = allocated[type_index].scan_num_bits_ptr();
    uint32_t num_obj = copy_from_device(d_num_obj_ptr);

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
        .count();
    bench_prefix_sum_time += micros;

    if (num_obj > 0) {
      kernel_bitmap_parallel_do_single_type1<T, BaseClass, P1, func><<<
          (num_obj + 256 - 1)/256, 256>>>(
              this, &allocated[AllocatorT::template TypeHelper<T>::kIndex], p1);
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template<class T>
  __device__ T* allocate_new() {
    // Use malloc and placement-new so that we can catch OOM errors.
    auto slot = global_free.deallocate();
    allocated[AllocatorT::template TypeHelper<T>::kIndex].allocate<true>(slot);
    void* ptr = data_storage + slot*kObjectSize;
    assert(ptr != nullptr);
    return (T*) ptr; 
  }

  template<class T>
  __device__ void free(T* obj) {
    assert(obj != nullptr);
    assert(reinterpret_cast<uint64_t>(obj)
        >= reinterpret_cast<uint64_t>(data_storage));
    uint32_t slot = (reinterpret_cast<uint64_t>(obj)
        - reinterpret_cast<uint64_t>(data_storage)) / kObjectSize;
    assert((reinterpret_cast<uint64_t>(obj)
        - reinterpret_cast<uint64_t>(data_storage)) % kObjectSize == 0);
    allocated[AllocatorT::template TypeHelper<T>::kIndex].deallocate<true>(slot);
    global_free.allocate<true>(slot);
  }

  template<class T, typename F, typename... Args>
  __device__ void device_do(F func, Args... args) {
    // device_do iterates over objects in a block.
    allocated[AllocatorT::template TypeHelper<T>::kIndex].enumerate(
      &BitmapSequentialExecutor<T, F, AllocatorState<AllocatorT>, Args...>::device_do,
      func, this, args...);
  }
};


template<uint32_t N_Objects, class... Types>
using SoaAllocator = SoaAllocatorAdapter<AllocatorState, N_Objects, Types...>;

#endif  // EXAMPLE_CONFIGURATION_BITMAP_ALLOCATOR_CONFIG_H
