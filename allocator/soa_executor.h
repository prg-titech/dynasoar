#ifndef ALLOCATOR_SOA_EXECUTOR_H
#define ALLOCATOR_SOA_EXECUTOR_H

#include <chrono>
#include <limits>

// For benchmarks: Measure time spent outside of parallel sections.
// Measure time in microseconds because numbers are small.
long unsigned int bench_prefix_sum_time = 0;
static const int kMaxInt32 = std::numeric_limits<int>::max();

// TODO: Is it safe to make these static?
template<typename WrapperT, typename AllocatorT, typename... Args>
__global__ static void kernel_parallel_do(AllocatorT* allocator, Args... args) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  WrapperT::parallel_do_cuda(allocator, args...);
}

// Run member function on allocator, then perform do-all operation.
template<typename WrapperT, typename PreT,
         typename AllocatorT, typename... Args>
__global__ static void kernel_parallel_do_with_pre(AllocatorT* allocator,
                                                   Args... args) {
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

template<typename AllocatorT, class BaseClass, typename P1,
         void(BaseClass::*func)(P1)>
struct ParallelDoTypeHelperP1 {
  // Iterating over all types T in the allocator.
  template<typename IterT>
  struct InnerHelper {
    // IterT is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      static bool call(AllocatorT* allocator, P1 p1) {
        allocator->template parallel_do_single_type<IterT, BaseClass, P1, func>(p1);
        return true;  // true means "continue processing".
      }
    };

    // IterT is not a subclass of BaseClass. Skip.
    template<int Dummy>
    struct ClassSelector<false, Dummy> {
      static bool call(AllocatorT* /*allocator*/, P1 /*p1*/) {
        return true;
      }
    };

    bool operator()(AllocatorT* allocator, P1 p1) {
      return ClassSelector<std::is_base_of<BaseClass, IterT>::value, 0>
          ::call(allocator, p1);
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
      auto time_start = std::chrono::system_clock::now();
      auto time_end = time_start;

      // Initialize iteration: Perform scan operation on bitmap.
      allocator->allocated_[kTypeIndex].scan();

      // Determine number of CUDA threads.
      auto* d_num_soa_blocks_ptr =
          &allocator->allocated_[AllocatorT::template BlockHelper<IterT>::kIndex]
              .data_.scan_data.enumeration_result_size;
      auto num_soa_blocks = copy_from_device(d_num_soa_blocks_ptr);

      if (num_soa_blocks > 0) {
        member_func_kernel<AllocatorT,
                           &AllocatorT::template initialize_iteration<IterT>>
            <<<(num_soa_blocks + kCudaBlockSize - 1)/kCudaBlockSize,
               kCudaBlockSize>>>(allocator);
        gpuErrchk(cudaDeviceSynchronize());

        time_end = std::chrono::system_clock::now();

        auto total_threads = num_soa_blocks * kSize;
        kernel_parallel_do<ThisClass>
            <<<(total_threads + kCudaBlockSize - 1)/kCudaBlockSize,
              kCudaBlockSize,
              shared_mem_size>>>(allocator, args...);
        gpuErrchk(cudaDeviceSynchronize());
      } else {
        time_end = std::chrono::system_clock::now();
      }

      auto elapsed = time_end - time_start;
      auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
          .count();
      bench_prefix_sum_time += micros;
    }

    template<void(AllocatorT::*pre_func)(Args...)>
    struct WithPre {
      using PreClass = WithPre<pre_func>;

      static void parallel_do(AllocatorT* allocator, int shared_mem_size,
                              Args... args) {
        auto time_start = std::chrono::system_clock::now();
        auto time_end = time_start;

        allocator->allocated_[kTypeIndex].scan();

        // Determine number of CUDA threads.
        auto* d_num_soa_blocks_ptr =
            &allocator->allocated_[AllocatorT::template BlockHelper<IterT>::kIndex]
                .data_.scan_data.enumeration_result_size;
        auto num_soa_blocks = copy_from_device(d_num_soa_blocks_ptr);

        if (num_soa_blocks > 0) {
          member_func_kernel<AllocatorT,
                             &AllocatorT::template initialize_iteration<IterT>>
              <<<(num_soa_blocks + kCudaBlockSize - 1)/kCudaBlockSize,
                 kCudaBlockSize>>>(allocator);
          gpuErrchk(cudaDeviceSynchronize());

          time_end = std::chrono::system_clock::now();

          auto total_threads = num_soa_blocks * kSize;
          kernel_parallel_do_with_pre<ThisClass, PreClass>
              <<<(total_threads + kCudaBlockSize - 1)/kCudaBlockSize,
                kCudaBlockSize,
                shared_mem_size>>>(allocator, args...);
          gpuErrchk(cudaDeviceSynchronize());
        } else {
          time_end = std::chrono::system_clock::now();
        }


        auto elapsed = time_end - time_start;
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
            .count();
        bench_prefix_sum_time += micros;
      }

      __DEV__ static void run_pre(AllocatorT* allocator, Args... args) {
        (allocator->*pre_func)(args...);
      }
    };

    static __DEV__ void parallel_do_cuda(AllocatorT* allocator,
                                         Args... args) {
      const auto N_alloc =
          allocator->allocated_[kTypeIndex].scan_num_bits();

      // Round to multiple of kSize.
      int num_threads = ((blockDim.x * gridDim.x)/kSize)*kSize;
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < num_threads) {
        for (int j = tid/kSize; j < N_alloc; j += num_threads/kSize) {
          // i is the index of in the scan array.
          auto block_idx = allocator->allocated_[kTypeIndex].scan_get_index(j);
          assert(block_idx <= kMaxInt32/64);

          // TODO: Consider doing a scan over "allocated" bitmap.
          auto* block = allocator->template get_block<IterT>(block_idx);
          const auto& iteration_bitmap = block->iteration_bitmap;
          int thread_offset = tid % kSize;

          if ((iteration_bitmap & (1ULL << thread_offset)) != 0ULL) {
            IterT* obj = allocator->template get_object<IterT>(
                block, thread_offset);
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
  __DEV__ static void device_do(BlockIndexT block_idx, F func,
                                AllocatorT* allocator, Args... args);
};

#endif  // ALLOCATOR_SOA_EXECUTOR_H
