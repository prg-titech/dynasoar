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

// Run member function on allocator, then perform do-all operation.
template<typename WrapperT, typename PreT,
         typename AllocatorT, typename... Args>
__global__ void kernel_parallel_do_with_pre(AllocatorT* allocator, Args... args) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  PreT::run_pre(allocator, args...);
  WrapperT::parallel_do_cuda(allocator, args...);
}

// Helper data structure for running parallel_do on all subtypes.
template<typename AllocatorT, class BaseClass, void(BaseClass::*func)()>
struct ParallelDoTypeHelper {
  // Iterating over all types T in the allocator.
  template<typename IterT>
  struct InnerHelper {
    // IterT is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      static bool call(AllocatorT* allocator) {
        // Initialize iteration: Perform scan operation on bitmap.
        kernel_init_iteration<AllocatorT, IterT><<<128, 128>>>(allocator);
        gpuErrchk(cudaDeviceSynchronize());

        allocator->template parallel_do_single_type<IterT, BaseClass, func>();

        return true;  // true means "continue processing".
      }
    };

    // IterT is not a subclass of BaseClass. Skip.
    template<int Dummy>
    struct ClassSelector<false, Dummy> {
      static bool call(AllocatorT* /*allocator*/) {
        return true;
      }
    };

    bool operator()(AllocatorT* allocator) {
      return ClassSelector<std::is_base_of<BaseClass, IterT>::value, 0>
          ::call(allocator);
    }
  };
};

template<typename AllocatorT, typename IterT, typename T, typename R,
         typename Base, typename... Args>
struct ParallelExecutor {
  using BlockHelperIterT = typename AllocatorT::template BlockHelper<IterT>;
  static const int kTypeIndex = BlockHelperIterT::kIndex;
  static const int kSize = BlockHelperIterT::kSize;
  static const int kCudaBlockSize = 256;

  template<R (Base::*func)(Args...)>
  struct FunctionWrapper {
    using ThisClass = FunctionWrapper<func>;

    static void parallel_do(AllocatorT* allocator, int shared_mem_size,
                            Args... args) {
      allocator->allocated_[kTypeIndex].scan();

      // Determine number of CUDA threads.
      uint32_t* d_num_soa_blocks_ptr =
          &allocator->allocated_[AllocatorT::template BlockHelper<T>::kIndex]
              .data_.enumeration_result_size;
      uint32_t num_soa_blocks = copy_from_device(d_num_soa_blocks_ptr);
      uint32_t total_threads = num_soa_blocks * BlockHelperIterT::kSize;

      kernel_parallel_do<ThisClass>
          <<<(total_threads + kCudaBlockSize - 1)/kCudaBlockSize,
            kCudaBlockSize,
            shared_mem_size>>>(allocator, args...);
      gpuErrchk(cudaDeviceSynchronize());
    }

    template<void(AllocatorT::*pre_func)(Args...)>
    struct WithPre {
      using PreClass = WithPre<pre_func>;

      static void parallel_do(AllocatorT* allocator, int shared_mem_size,
                              Args... args) {
        allocator->allocated_[kTypeIndex].scan();

        // Determine number of CUDA threads.
        uint32_t* d_num_soa_blocks_ptr =
            &allocator->allocated_[AllocatorT::template BlockHelper<T>::kIndex]
                .data_.enumeration_result_size;
        uint32_t num_soa_blocks = copy_from_device(d_num_soa_blocks_ptr);
        uint32_t total_threads = num_soa_blocks * BlockHelperIterT::kSize;

        kernel_parallel_do_with_pre<ThisClass, PreClass>
            <<<(total_threads + kCudaBlockSize - 1)/kCudaBlockSize,
              kCudaBlockSize,
              shared_mem_size>>>(allocator, args...);
        gpuErrchk(cudaDeviceSynchronize());
      }

      __DEV__ static void run_pre(AllocatorT* allocator, Args... args) {
        (allocator->*pre_func)(args...);
      }
    };

    static __DEV__ void parallel_do_cuda(AllocatorT* allocator,
                                         Args... args) {
      const uint32_t N_alloc =
          allocator->allocated_[kTypeIndex].scan_num_bits();

      // Round to multiple of 64.
      int num_threads = ((blockDim.x * gridDim.x)/kSize)*kSize;
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < num_threads) {
        for (int j = tid/kSize; j < N_alloc; j += num_threads/kSize) {
          // i is the index of in the scan array.
          int block_idx = allocator->allocated_[kTypeIndex].scan_get_index(j);

          // TODO: Consider doing a scan over "allocated" bitmap.
          auto* block = allocator->template get_block<T>(block_idx);
          auto iteration_bitmap = block->iteration_bitmap;

          int thread_offset = tid % kSize;
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

template<typename T, typename F, typename AllocatorT, typename... Args>
struct SequentialExecutor {
  // Defined in soa_allocator.h.
  __DEV__ static void device_do(uint32_t block_idx, F func,
                                AllocatorT* allocator, Args... args);
};

#endif  // ALLOCATOR_SOA_EXECUTOR_H
