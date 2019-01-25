#ifndef ALLOCATOR_ALLOCATOR_HANDLE_H
#define ALLOCATOR_ALLOCATOR_HANDLE_H

#include "allocator/configuration.h"
#include "allocator/tuple_helper.h"
#include "allocator/util.h"

template<typename AllocatorT>
__global__ void init_allocator_kernel(AllocatorT* allocator,
                                      char* data_buffer) {
  allocator->initialize(data_buffer);
}

template<typename AllocatorT>
__global__ void kernel_print_state_stats(AllocatorT* allocator) {
  assert(gridDim.x == 1 && blockDim.x == 1);
  allocator->DBG_print_state_stats();
}

// A wrapper class for accessing the allocator from host side.
template<typename AllocatorT>
class AllocatorHandle {
 public:
  AllocatorHandle(const AllocatorHandle<AllocatorT>&) = delete;

  // Initialize the allocator: Allocator class and data buffer.
  AllocatorHandle() {
#ifndef NDEBUG
    int device_id;
    gpuErrchk(cudaGetDevice(&device_id));

    // Source: https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("Current Device Number: %d\n", device_id);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    // Query memory information.
    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Total global memory: %f MB\n",
           (total_mem/1000000.0));
    printf("  Available (free) global memory: %f MB\n\n",
           (free_mem/1000000.0));

    AllocatorT::DBG_print_stats();
#endif  // NDEBUG

    cudaMalloc(&allocator_, sizeof(AllocatorT));
    assert(allocator_ != nullptr);

    cudaMalloc(&data_buffer_, AllocatorT::kDataBufferSize);
#ifndef NDEBUG
    void* maybe_out_of_memory = nullptr;  // To show OOM text...
    assert(data_buffer_ != maybe_out_of_memory);
#endif  // NDEBUG

    init_allocator_kernel<<<256, 256>>>(allocator_, data_buffer_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  // Delete the allocator: Free all associated CUDA memory.
  ~AllocatorHandle() {
    cudaFree(allocator_);
    cudaFree(data_buffer_);
  }

  long unsigned int DBG_get_enumeration_time() {
    return allocator_->DBG_get_enumeration_time();
  }

  void DBG_collect_stats() {
    member_func_kernel<AllocatorT, &AllocatorT::DBG_collect_stats>
        <<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  void DBG_print_collected_stats() {
    member_func_kernel<AllocatorT, &AllocatorT::DBG_print_collected_stats>
        <<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  // Returns a device pointer to the allocator.
  AllocatorT* device_pointer() { return allocator_; }

  // Runs a member function T::func for all objects of a type on device.
  template<class T, void(T::*func)()>
  void parallel_do() {
    allocator_->parallel_do<T, func>();
  }

#ifdef OPTION_DEFRAG
  // Defrag/compact all objects of type T. Also updates all affected pointers
  // in the data buffer.
  // Should be invoked from host side.
  template<typename T, int NumRecords>
  void parallel_defrag(int min_num_compactions = 16) {
    allocator_->parallel_defrag<T, NumRecords>(min_num_compactions);
  }

  template<typename T>
  void parallel_defrag(int min_num_compactions = 16) {
    allocator_->parallel_defrag<T>(min_num_compactions);
  }
#endif  // OPTION_DEFRAG

  void DBG_print_state_stats() {
    kernel_print_state_stats<<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

#ifdef OPTION_DEFRAG
  void DBG_print_defrag_time() {
    allocator_->DBG_print_defrag_time();
  }
#endif  // OPTION_DEFRAG

 private:
  AllocatorT* allocator_ = nullptr;
  char* data_buffer_ = nullptr;
};

#endif  // ALLOCATOR_ALLOCATOR_HANDLE_H
