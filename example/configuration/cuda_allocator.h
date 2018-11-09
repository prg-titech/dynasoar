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

  template<typename T>
  struct TypeId {
    static const uint8_t value = TYPE_INDEX(Types..., T);
  };

  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    // Add object to pointer array.
    T* result =  new T(args...);
    result->set_type(TYPE_INDEX(Types..., T));
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

  // TODO: Implement.
  __DEV__ void DBG_print_state_stats() {}

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


template<typename C, int Field>
class SoaField {
 private:
  using T = typename SoaFieldHelper<C, Field>::type;

  // Data stored in here.
  T data_;

  __DEV__ T* data_ptr() const { return &data_; }

 public:
  // Field initialization.
  __DEV__ SoaField() {}
  __DEV__ explicit SoaField(const T& value) : data_(value);

  // Explicit conversion for automatic conversion to base type.
  __DEV__ operator T&() { return data_; }
  __DEV__ operator const T&() const { return data_; }

  // Custom address-of operator.
  __DEV__ T* operator&() { return &data_; }
  __DEV__ T* operator&() const { return &data_; }

  // Support member function calls.
  __DEV__ T& operator->() { return data_; }
  __DEV__ T& operator->() const { return data_; }

  // Dereference type in case of pointer type.
  __DEV__ typename std::remove_pointer<T>::type& operator*() {
    return *data_;
  }
  __DEV__ typename std::remove_pointer<T>::type& operator*() const {
    return *data_;
  }

  // Array access in case of device array.
  template<typename U = T>
  __DEV__ typename std::enable_if<is_device_array<U>::value,
                                  typename U::BaseType>::type&
  operator[](size_t pos) { return data_[pos]; }

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    data_ = value;
    return data_;
  }
};


template<class AllocatorT>
class SoaBase {
 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;

  __DEV__ uint8_t get_type() const { return dynamic_type_; }

  __DEV__ void set_type(uint8_t type_id) { dynamic_type_ = type_id; }

 private:
  uint8_t dynamic_type_;
};

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
