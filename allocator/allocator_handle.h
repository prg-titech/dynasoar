#ifndef ALLOCATOR_ALLOCATOR_HANDLE_H
#define ALLOCATOR_ALLOCATOR_HANDLE_H

template<typename AllocatorT>
__global__ void init_allocator_kernel(AllocatorT* allocator) {
  allocator->initialize();
}

template<typename AllocatorT>
class AllocatorHandle {
 public:
  AllocatorHandle(const AllocatorHandle<AllocatorT>&) = delete;

  AllocatorHandle() {
    cudaMalloc(&allocator_, sizeof(AllocatorT));
    assert(allocator_ != nullptr);

    init_allocator_kernel<<<256, 256>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  AllocatorT* device_pointer() { return allocator_; }

  template<int W_MULT, class T, void(T::*func)()>
  void parallel_do(int num_blocks, int num_threads) {
    allocator_->parallel_do<W_MULT, T, func>(num_blocks, num_threads);
  }

 private:
  AllocatorT* allocator_ = nullptr;
};

#endif  // ALLOCATOR_ALLOCATOR_HANDLE_H
