#ifndef ALLOCATOR_SOA_EXECUTOR_H
#define ALLOCATOR_SOA_EXECUTOR_H

#include <chrono>
#include <limits>

#include "util/util.h"

/**
 * For benchmarks: Measures time spent in parallel_do, but outside of CUDA
 * kernels. Measures time in microseconds because numbers are small.
 */
long unsigned int bench_prefix_sum_time = 0;

/**
 * Maximum representable 32-bit integer.
 */
static const int kMaxInt32 = std::numeric_limits<int>::max();

/**
 * A CUDA kernel that runs a method in parallel for all objects of a type.
 * Method and type are specified as part of \p WrapperT.
 * @tparam WrapperT Must be a template instantiation of
 *                  ParallelExecutor::FunctionArgTypesWrapper::FunctionWrapper.
 * @tparam AllocatorT Allocator type
 * @tparam Args Type of arguments passed to method
 * @param allocator Device allocator handle
 * @param args Arguments passed to method
 */
template<typename WrapperT, typename AllocatorT, typename... Args>
__global__ static void kernel_parallel_do(AllocatorT* allocator, Args... args) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  WrapperT::parallel_do_cuda(allocator, args...);
}

// Run member function on allocator, then perform do-all operation.
/**
 * Same as kernel_parallel_do(), but runs a member function of the allocator
 * before running the actual parallel do-all.
 * @tparam WrapperT Must be a template instantiation of
 *                  ParallelExecutor::FunctionArgTypesWrapper::FunctionWrapper.
 * @tparam AllocatorT Allocator type
 * @tparam Args Type of arguments passed to method
 * @tparam PreT A class that invoke the allocator member function
 * @param allocator Device allocator handle
 * @param args Arguments passed to method
 */
template<typename WrapperT, typename PreT,
         typename AllocatorT, typename... Args>
__global__ static void kernel_parallel_do_with_pre(AllocatorT* allocator,
                                                   Args... args) {
  // TODO: Check overhead of allocator pointer dereference.
  // There is definitely a 2% overhead or so.....
  PreT::run_pre(allocator, args...);
  WrapperT::parallel_do_cuda(allocator, args...);
}

/**
 * A helper class for running a parallel do-all operation on all subtypes of a
 * given base class. Internally, this helper class spawns a CUDA kernel for
 * each subtype. The nested class ParallelDoTypeHelperL3 is a functor to be
 * used with TupleHelper.
 * @tparam Args Member function parameter types
 */
template<typename... Args>
struct ParallelDoTypeHelperL1 {
  /**
   * @tparam AllocatorT Allocator type
   * @tparam BaseClass Base class
   * @tparam func Member function to be run
   * @tparam Scan Indicates if a scan operation should be run
   */
  template<typename AllocatorT, class BaseClass,
           void(BaseClass::*func)(Args...), bool Scan>
  struct ParallelDoTypeHelperL2 {
    // Iterating over all types T in the allocator.
    template<typename IterT>
    struct ParallelDoTypeHelperL3{
      // IterT is a subclass of BaseClass. Check if same type.
      template<bool Check, int Dummy>
      struct ClassSelector {
        static bool call(AllocatorT* allocator, Args...args) {
          allocator->template parallel_do_single_type<
              IterT, BaseClass, Args..., func, Scan>(
                  std::forward<Args>(args)...);
          return true;  // true means "continue processing".
        }
      };

      // IterT is not a subclass of BaseClass. Skip.
      template<int Dummy>
      struct ClassSelector<false, Dummy> {
        static bool call(AllocatorT* /*allocator*/, Args... /*args*/) {
          return true;
        }
      };

      bool operator()(AllocatorT* allocator, Args... args) {
        return ClassSelector<std::is_base_of<BaseClass, IterT>::value, 0>
            ::call(allocator, std::forward<Args>(args)...);
      }
    };
  };
};

template<bool Scan, typename AllocatorT, typename IterT, typename T>
struct ParallelExecutor {
  using BlockHelperIterT = typename AllocatorT::template BlockHelper<IterT>;
  static const int kTypeIndex = BlockHelperIterT::kIndex;
  static const int kSize = BlockHelperIterT::kSize;
  static const int kCudaBlockSize = 256;

  template<typename R, typename Base, typename... Args>
  struct FunctionArgTypesWrapper {

    template<R (Base::*func)(Args...)>
    struct FunctionWrapper {
      using ThisClass = FunctionWrapper<func>;

      static void parallel_do(AllocatorT* allocator, int shared_mem_size,
                              Args... args) {
        auto time_start = std::chrono::system_clock::now();
        auto time_end = time_start;

        if (Scan) {
          // Initialize iteration: Perform scan operation on bitmap.
          allocator->allocated_[kTypeIndex].scan();
        }

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
                shared_mem_size>>>(allocator, std::forward<Args>(args)...);
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

          if (Scan) {
            allocator->allocated_[kTypeIndex].scan();
          }

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
                  shared_mem_size>>>(allocator, std::forward<Args>(args)...);
            gpuErrchk(cudaDeviceSynchronize());
          } else {
            time_end = std::chrono::system_clock::now();
          }


          auto elapsed = time_end - time_start;
          auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
              .count();
          bench_prefix_sum_time += micros;
        }

        __device__ static void run_pre(AllocatorT* allocator, Args... args) {
          (allocator->*pre_func)(std::forward<Args>(args)...);
        }
      };

      static __device__ void parallel_do_cuda(AllocatorT* allocator,
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
              (obj->*func)(std::forward<Args>(args)...);
            }
          }
        }
      }
    };
  };
};

template<typename T, typename F, typename AllocatorT, typename... Args>
struct SequentialExecutor {
  // Defined in soa_allocator.h.
  __host__ __device__ static void device_do(BlockIndexT block_idx, F func,
                                            AllocatorT* allocator,
                                            Args... args);
};

#endif  // ALLOCATOR_SOA_EXECUTOR_H
