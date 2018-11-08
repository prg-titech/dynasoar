#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H

#include "allocator/tuple_helper.h"

template<typename AllocatorT, typename T>
__global__ void kernel_init_stream_compaction(AllocatorT* allocator) {
  allocator->template initialize_stream_compaction_array<T>();
}

// Execution Model: Single-Method Multiple-Objects
// - Run same method on multiple objects and allow creation/destruction of
//   objects during runtime.
// Examples:
// (a) Agent-based Simulation with Dynamic Agents:
//     Wa-Tor (predator-prey), Traffic Flow, n-body with Collisions,
//     Game of Life
// (b) Simulation with Incremental Cloning:
//     Evacuation
// (c) Graph/Tree Algorithms:
//     Barnes-Hut
// (d) Microbenchmarks

// One way to motivate SoaAlloc: Explain how to modify another GPU memory
// allocator for better performance.
// (1) Branch Divergence: If there are different classes, group by classes.
//     --> Maintain an array of object pointers per class.
// (2) Memory Locality: Before do-all, sort array of object pointers.
template<uint32_t N_Objects, class... Types>
class SoaAllocator {
 public:
  using ThisAllocator = SoaAllocator<N_Objects, Types...>;

  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    // Add object to pointer array.
    T* result =  new T(args...);
    auto pos = atomicAdd(&num_objects_[TYPE_INDEX(Types..., T)], 1);
    objects_[TYPE_INDEX(Types..., T)][pos] = result;
    return result;
  }

  template<class T>
  __DEV__ void free(T* obj) {
    auto pos = atomicAdd(&num_deleted_objects[TYPE_INDEX(Types..., T)], 1);
    deleted_objects_[TYPE_INDEX(Types..., T)][pos] = obj;
    delete obj;
  }

 private:
  static const int kNumTypes = sizeof...(Types);
  static const int kMaxObjects = N_Objects;
  static const unsigned int kInvalidObject =
      std::numeric_limits<unsigned int>::max();

  unsigned int num_objects_[kNumTypes];
  unsigned int num_deleted_objects[kNumTypes];

  // Array of pointers to all objects of a type.
  void* objects_[kNumTypes][kMaxObjects];

  // Array of deleted objects of a type.
  void* deleted_objects_[kNumTypes][kMaxObjects];

  unsigned int stream_compaction_array_[kMaxObjects];
  unsigned int stream_compaction_output_[kMaxObjects];
  unsigned int stream_compaction_temp_[3*kMaxObjects];

  template<typename T>
  __DEV__ void initialize_stream_compaction_array() {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_objects_[TYPE_INDEX(Types..., T)];
         i += blockDim.x * gridDim.x) {
      void* ptr = objects_[TYPE_INDEX(Types..., T)][i];
      bool object_deleted = false;

      // Check if ptr is in deleted object set (binary search).
      // TODO: Store deleted set in shared memory if possible.
      // TODO: Implement binary search.
      for (int j = 0; i < num_deleted_objects[TYPE_INDEX(Types..., T)]; ++j) {
        if (ptr == deleted_objects_[TYPE_INDEX(Types..., T)][j]) {
          object_deleted = true;
          break;
        }
      }

      // TODO: Is this really the best way? Check simulation paper.
      if (object_deleted) {
        stream_compaction_array_[j] = 0;
      } else {
        stream_compaction_array_[j] = 1;
      }
    }
  }

  template<typename T>
  unsigned int num_objects() {
    return read_from_device<unsigned int>(&num_objects_[TYPE_INDEX(Types..., T)]);
  }
};

template<typename AllocatorT>
class AllocatorHandle {
 public:
  template<int W_MULT, class T, void(T::*func)()>
  void parallel_do() {
    kernel_init_stream_compaction<ThisAllocator, T><<<128, 128>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());

    // Get total number of objects.
    unsigned int num_objects = allocator_->num_objects();

    // Run prefix sum algorithm.
    size_t temp_size = 3*num_objects;
    cub::DeviceScan::ExclusiveSum(allocator_->stream_compaction_temp_,
                                  temp_size,
                                  allocator_->stream_compaction_array_,
                                  allocator_->stream_compaction_output_,
                                  num_objects);

    // Compact array.
    kernel_init_iteration<AllocatorT, T><<<128, 128>>>(allocator_);

    gpuErrchk(cudaDeviceSynchronize());
    allocator_->parallel_do<W_MULT, T, func>();
  }

  // This is a no-op.
  template<class T>
  void parallel_defrag(int max_records, int min_records = 1) {}
};

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
