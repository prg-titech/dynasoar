#ifndef ALLOCATOR_SOA_EXECUTOR_H
#define ALLOCATOR_SOA_EXECUTOR_H

template<typename AllocatorT, typename T>
__global__ void kernel_init_iteration(AllocatorT* allocator) {
  allocator->template initialize_iteration<T>();
}

template<typename WrapperT, typename AllocatorT, typename... Args>
__global__ void kernel_parallel_do(AllocatorT* allocator, Args... args) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  WrapperT::parallel_do_cuda(allocator, args...);
}

template<typename AllocatorT, typename T, typename R,
         typename Base, typename... Args>
struct ParallelExecutor {
  using BlockHelperT = typename AllocatorT::template BlockHelper<T>;
  static const int kTypeIndex = BlockHelperT::kIndex;

  template<R (Base::*func)(Args...)>
  struct FunctionWrapper {
    using ThisClass = FunctionWrapper<func>;

    static void parallel_do(AllocatorT* allocator, int num_blocks,
                            int num_threads, int shared_mem_size,
                            Args... args) {
      allocator->allocated_[kTypeIndex].scan();
      kernel_parallel_do<ThisClass>
          <<<num_blocks, num_threads, shared_mem_size>>>(allocator, args...);
      gpuErrchk(cudaDeviceSynchronize());
    }

    static __DEV__ void parallel_do_cuda(AllocatorT* allocator,
                                         Args... args) {
      const uint32_t N_alloc =
          allocator->allocated_[kTypeIndex].scan_num_bits();
      const int num_objs = BlockHelperT::kSize;

      // Round to multiple of 64.
      int num_threads = ((blockDim.x * gridDim.x)/num_objs)*num_objs;
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < num_threads) {
        for (int j = tid/num_objs; j < N_alloc; j += num_threads/num_objs) {
          // i is the index of in the scan array.
          int block_idx = allocator->allocated_[kTypeIndex].scan_get_index(j);

          // TODO: Consider doing a scan over "allocated" bitmap.
          auto* block = allocator->template get_block<T>(block_idx);
          auto iteration_bitmap = block->iteration_bitmap;

          int thread_offset = tid % num_objs;
          // Advance bitmap to return thread_offset-th bit index.
          for (int i = 0; i < thread_offset; ++i) {
            // Clear last bit.
            iteration_bitmap &= iteration_bitmap - 1;
          }
          int obj_bit = __ffsll(iteration_bitmap);
          if (obj_bit > 0) {
            T* obj = allocator->template get_object<T>(block, obj_bit - 1);
            // call the function.
            (obj->*func)(args...);
          }
        }
      }
    }
  };
};

#endif  // ALLOCATOR_SOA_EXECUTOR_H