#ifndef ALLOCATOR_ALLOCATOR_HANDLE_H
#define ALLOCATOR_ALLOCATOR_HANDLE_H

#include "allocator/util.h"

template<typename AllocatorT>
__global__ void init_allocator_kernel(AllocatorT* allocator,
                                      char* data_buffer) {
  new(allocator) AllocatorT(data_buffer);
}

// A wrapper class for accessing the allocator from host side.
template<typename AllocatorT>
class AllocatorHandle {
 public:
  AllocatorHandle(const AllocatorHandle<AllocatorT>&) = delete;

  // Initialize the allocator: Allocator class and data buffer.
  AllocatorHandle() {
    cudaMalloc(&allocator_, sizeof(AllocatorT));
    assert(allocator_ != nullptr);

    cudaMalloc(&data_buffer_, AllocatorT::kDataBufferSize);
    assert(data_buffer_ != nullptr);

    init_allocator_kernel<<<256, 256>>>(allocator_, data_buffer_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  // Delete the allocator: Free all associated CUDA memory.
  ~AllocatorHandle() {
    cudaFree(allocator_);
    cudaFree(data_buffer_);
  }

  // Returns a device pointer to the allocator.
  AllocatorT* device_pointer() { return allocator_; }

  // Runs a member function T::func for all objects of a type on device.
  template<class T, void(T::*func)()>
  void parallel_do() {
    // Initialize iteration: Perform scan operation on bitmap.
    kernel_init_iteration<AllocatorT, T><<<128, 128>>>(allocator_);

    gpuErrchk(cudaDeviceSynchronize());
    allocator_->parallel_do<T, func>();
  }

  // Defrag/compact all objects of type T. Also updates all affected pointers
  // in the data buffer.
  template<class T>
  void parallel_defrag(int max_records, int min_records = 1) {
    allocator_->parallel_defrag<T>(max_records, min_records);
  }

 private:
  AllocatorT* allocator_ = nullptr;
  char* data_buffer_ = nullptr;
};

#endif  // ALLOCATOR_ALLOCATOR_HANDLE_H
