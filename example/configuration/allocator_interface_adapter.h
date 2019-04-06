#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H

#include <assert.h>
#include <chrono>
#include <cub/cub.cuh>

#include "allocator/tuple_helper.h"

#ifndef PARAM_HEAP_SIZE
// Custom allocator heap size: 8 GiB
static const size_t kMallocHeapSize = 8ULL*1024*1024*1024;
#else
// Heap size specified by parameter.
static const size_t kMallocHeapSize = PARAM_HEAP_SIZE;
#endif  // PARAM_HEAP_SIZE


#define declare_field_types(classname, ...) \
  __DEV__ void* operator new(size_t sz, typename classname::Allocator* allocator) { \
    return allocator->allocate_new<classname>(); \
  } \
  __DEV__ void* operator new(size_t sz, classname* ptr) { \
    return ptr; \
  } \
  __DEV__ void operator delete(void* ptr, typename classname::Allocator* allocator) { \
    allocator->free<classname>(reinterpret_cast<classname*>(ptr)); \
  } \
  __DEV__ void operator delete(void*, classname*) { \
    assert(false);  /* Construct must not throw exceptions. */ \
  } \
  using FieldTypes = std::tuple<__VA_ARGS__>;

// For benchmarks: Measure time spent outside of parallel sections.
long unsigned int bench_prefix_sum_time = 0;


template<typename AllocatorT, typename T>
__global__ void kernel_init_stream_compaction(AllocatorT* allocator) {
  allocator->template initialize_stream_compaction_array<T>();
}


template<typename AllocatorT, typename T>
__global__ void kernel_compact_object_array(AllocatorT* allocator) {
  allocator->template compact_object_array<T>();
}


template<typename AllocatorT>
__global__ void kernel_print_state_stats(AllocatorT* allocator) {
  allocator->DBG_print_state_stats();
}


template<typename AllocatorT, typename T>
__global__ void kernel_update_object_count(AllocatorT* allocator) {
  allocator->template update_object_count<T>();
}


template<typename AllocatorT, class T, class BaseClass,
         void(BaseClass::*func)()>
__global__ void kernel_parallel_do_single_type(AllocatorT* allocator,
                                               unsigned int num_obj) {
  allocator->template parallel_do_single_type<T, BaseClass, func>(num_obj);
}


template<typename AllocatorT, class T, class BaseClass, typename P1,
         void(BaseClass::*func)(P1)>
__global__ void kernel_parallel_do_single_type1(AllocatorT* allocator,
                                               unsigned int num_obj,
                                               P1 p1) {
  allocator->template parallel_do_single_type<T, BaseClass, P1, func>(num_obj, p1);
}


// Helper data structure for running parallel_do on all subtypes.
template<typename AllocatorT, class BaseClass, void(BaseClass::*func)(), bool Scan>
struct ParallelDoTypeHelper {
  // Iterating over all types T in the allocator.
  template<typename IterT>
  struct InnerHelper {
    // IterT is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      static bool call(AllocatorT* allocator) {
        allocator->template parallel_do_single_type<IterT, BaseClass, func, Scan>();
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


template<typename AllocatorT, class BaseClass, typename P1, void(BaseClass::*func)(P1), bool Scan>
struct ParallelDoTypeHelper1 {
  // Iterating over all types T in the allocator.
  template<typename IterT>
  struct InnerHelper {
    // IterT is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      static bool call(AllocatorT* allocator, P1 p1) {
        allocator->template parallel_do_single_type<IterT, BaseClass, P1, func, Scan>(p1);
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


template<class AllocatorT>
class SoaBase {
 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;

  __DEV__ uint8_t get_type() const { return dynamic_type_; }

  template<typename T>
  __DEV__ T* cast() {
    if (this != nullptr
        && get_type() == AllocatorT::template TypeId<T>::value) {
      return static_cast<T*>(this);
    } else {
      return nullptr;
    }
  }

 private:
  friend AllocatorT;

  uint8_t dynamic_type_;

  __DEV__ void set_type(uint8_t type_id) { dynamic_type_ = type_id; }
};


// TODO: Fix visiblity.
template<template<typename> typename AllocatorStateT,
         uint32_t N_Objects, class... Types>
class SoaAllocatorAdapter {
 public:
  using ThisAllocator = SoaAllocatorAdapter<
      AllocatorStateT, N_Objects, Types...>;
  using Base = SoaBase<ThisAllocator>;

  // Zero-initialization of arrays can take a long time.
  __DEV__ SoaAllocatorAdapter() = delete;

  template<typename T>
  struct TypeHelper {
    static const int kIndex = TYPE_INDEX(Types..., T);
  };

  long unsigned int DBG_get_enumeration_time() {
    // Convert microseconds to milliseconds.
    return bench_prefix_sum_time;
  }

  template<typename T>
  struct TypeId {
    static const uint8_t value = TypeHelper<T>::kIndex;
  };

  template<bool Scan, class T, void(T::*func)()>
  void parallel_do() {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelper<ThisAllocator, T, func, Scan>
        ::template InnerHelper>(this);
  }

  template<bool Scan, class T, typename P1, void(T::*func)(P1)>
  void parallel_do(P1 p1) {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelper1<ThisAllocator, T, P1, func, Scan>
        ::template InnerHelper>(this, p1);
  }

  template<class T, class BaseClass, void(BaseClass::*func)(), bool Scan,
           typename U = AllocatorStateT<ThisAllocator>>
  typename std::enable_if<U::kHasParallelDo, void>::type
  parallel_do_single_type() {
    allocator_state_.parallel_do_single_type<T, BaseClass, func, Scan>();
  }

  template<class T, class BaseClass, typename P1, void(BaseClass::*func)(P1),
           bool Scan, typename U = AllocatorStateT<ThisAllocator>>
  typename std::enable_if<U::kHasParallelDo, void>::type
  parallel_do_single_type(P1 p1) {
    allocator_state_.parallel_do_single_type<T, BaseClass, P1, func, Scan>(p1);
  }

  template<class T, class BaseClass, void(BaseClass::*func)(),
           bool Scan, typename U = AllocatorStateT<ThisAllocator>>
  typename std::enable_if<!U::kHasParallelDo, void>::type
  parallel_do_single_type() {
    // Get total number of objects.
    unsigned int num_objects = this->template num_objects<T>();

    if (num_objects > 0) {
      if (Scan) {
        auto time_start = std::chrono::system_clock::now();

        kernel_init_stream_compaction<ThisAllocator, T><<<
            (num_objects + 256 - 1)/256, 256>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        // Run prefix sum algorithm.
        // TODO: Prefix sum broken for num_objects < 256.
        auto prefix_sum_size = num_objects < 256 ? 256 : num_objects;
        size_t temp_size = 3*prefix_sum_size;
        cub::DeviceScan::ExclusiveSum(stream_compaction_temp_,
                                      temp_size,
                                      stream_compaction_array_,
                                      stream_compaction_output_,
                                      prefix_sum_size);
        gpuErrchk(cudaDeviceSynchronize());

        // Compact array.
        kernel_compact_object_array<ThisAllocator, T><<<
            (num_objects + 256 - 1)/256, 256>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        // Update arrays and counts.
        kernel_update_object_count<ThisAllocator, T><<<1, 1>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        auto time_end = std::chrono::system_clock::now();
        auto elapsed = time_end - time_start;
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
            .count();
        bench_prefix_sum_time += micros;
      }

      num_objects = this->template num_objects<T>();

      kernel_parallel_do_single_type<ThisAllocator, T, BaseClass, func><<<
          (num_objects + kCudaBlockSize - 1)/kCudaBlockSize,
          kCudaBlockSize>>>(this, num_objects);
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template<class T, class BaseClass, typename P1, void(BaseClass::*func)(P1),
           bool Scan, typename U = AllocatorStateT<ThisAllocator>>
  typename std::enable_if<!U::kHasParallelDo, void>::type
  parallel_do_single_type(P1 p1) {
    // Get total number of objects.
    unsigned int num_objects = this->template num_objects<T>();

    if (num_objects > 0) {
      if (Scan) {
        auto time_start = std::chrono::system_clock::now();

        kernel_init_stream_compaction<ThisAllocator, T><<<
            (num_objects + 256 - 1)/256, 256>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        // Run prefix sum algorithm.
        // TODO: Prefix sum broken for num_objects < 256.
        auto prefix_sum_size = num_objects < 256 ? 256 : num_objects;
        size_t temp_size = 3*prefix_sum_size;
        cub::DeviceScan::ExclusiveSum(stream_compaction_temp_,
                                      temp_size,
                                      stream_compaction_array_,
                                      stream_compaction_output_,
                                      prefix_sum_size);
        gpuErrchk(cudaDeviceSynchronize());

        // Compact array.
        kernel_compact_object_array<ThisAllocator, T><<<
            (num_objects + 256 - 1)/256, 256>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        // Update arrays and counts.
        kernel_update_object_count<ThisAllocator, T><<<1, 1>>>(this);
        gpuErrchk(cudaDeviceSynchronize());

        auto time_end = std::chrono::system_clock::now();
        auto elapsed = time_end - time_start;
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
            .count();
        bench_prefix_sum_time += micros;
      }

      num_objects = this->template num_objects<T>();

      kernel_parallel_do_single_type1<ThisAllocator, T, BaseClass, P1, func><<<
          (num_objects + kCudaBlockSize - 1)/kCudaBlockSize,
          kCudaBlockSize>>>(this, num_objects, p1);
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  // Pass in num_obj as parameter because this kernel might create new
  // objects and thus change the number of objects.
  template<class T, class BaseClass, void(BaseClass::*func)()>
  __DEV__ void parallel_do_single_type(unsigned int num_obj) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      T* obj = reinterpret_cast<T*>(objects_[TypeHelper<T>::kIndex][i]);
      (obj->*func)();
    }
  }

  template<class T, class BaseClass, typename P1, void(BaseClass::*func)(P1)>
  __DEV__ void parallel_do_single_type(unsigned int num_obj, P1 p1) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      T* obj = reinterpret_cast<T*>(objects_[TypeHelper<T>::kIndex][i]);
      (obj->*func)(p1);
    }
  }

  // Call a member function on all objects of type.
  // Device version (sequential).
  // TODO: This does not enumerate subtypes.
  template<class T, typename F, typename U = AllocatorStateT<ThisAllocator>,
           typename... Args>
  __DEV__ typename std::enable_if<!U::kHasParallelDo, void>::type
  device_do(F func, Args... args) {
    auto num_obj = num_objects_[TypeHelper<T>::kIndex];

    for (int i = 0; i < num_obj; ++i) {
      T* obj = reinterpret_cast<T*>(objects_[TypeHelper<T>::kIndex][i]);
      (obj->*func)(args...);
    }
  }

  template<class T, typename F, typename U = AllocatorStateT<ThisAllocator>,
           typename... Args>
  __DEV__ typename std::enable_if<U::kHasParallelDo, void>::type
  device_do(F func, Args... args) {
    allocator_state_.device_do<T, F, Args...>(func, args...);
  }

  template<class T>
  __DEV__ T* allocate_new() {
    // Add object to pointer array.
    T* result = allocator_state_.template allocate_new<T>();
    result->set_type(TypeHelper<T>::kIndex);

    if (!AllocatorStateT<ThisAllocator>::kHasParallelDo) {
      auto pos = atomicAdd(&num_objects_[TypeHelper<T>::kIndex], 1);
      assert(pos < kMaxObjects);
      objects_[TypeHelper<T>::kIndex][pos] = result;
    }

    return result;
  }

  // Helper data structure for freeing objects whose types are subtypes of the
  // declared type. BaseClass is the declared type.
  template<typename BaseClass>
  struct FreeHelper {
    // Iterating over all types T in the allocator.
    template<typename T>
    struct InnerHelper {
      // T is a subclass of BaseClass. Check if same type.
      template<bool Check, int Dummy>
      struct ClassSelector {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          if (obj->get_type() == TypeHelper<T>::kIndex) {
            allocator->free_typed(static_cast<T*>(obj));
            return false;  // No need to check other types.
          } else {
            return true;   // true means "continue processing".
          }
        }
      };

      // T is not a subclass of BaseClass. Skip.
      template<int Dummy>
      struct ClassSelector<false, Dummy> {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          return true;
        }
      };

      __DEV__ bool operator()(ThisAllocator* allocator, BaseClass* obj) {
        return ClassSelector<std::is_base_of<BaseClass, T>::value, 0>::call(
            allocator, obj);
      }
    };
  };

  template<class T>
  __DEV__ void free(T* obj) {
    uint8_t type_id = obj->get_type();
    if (type_id == TypeHelper<T>::kIndex) {
      free_typed(obj);
    } else {
      bool result = TupleHelper<Types...>
          ::template dev_for_all<FreeHelper<T>::InnerHelper>(this, obj);
      assert(result);  // true means type was found.
    }
  }

  template<class T>
  __DEV__ void free_typed(T* obj) {
    obj->~T();

    if (!AllocatorStateT<ThisAllocator>::kHasParallelDo) {
      auto pos = atomicAdd(&num_deleted_objects_[TypeHelper<T>::kIndex], 1);
      deleted_objects_[TypeHelper<T>::kIndex][pos] = obj;
    }

    allocator_state_.free<T>(obj);
  }

  // TODO: Implement missing DBG functions.
  template<class T>
  __DEV__ uint32_t DBG_allocated_slots() { return 0; }

  template<class T>
  __DEV__ uint32_t DBG_used_slots() { 
    return num_objects_[TypeHelper<T>::kIndex];
  }

  static void DBG_print_stats() {}

  template<class T>
  uint32_t DBG_host_allocated_slots() { return 0; }

  // TODO: Implement.
  template<class T>
  uint32_t DBG_host_used_slots() { return 0; }

  __DEV__ void DBG_print_state_stats() {}

  __DEV__ void* atomiCasPtr(void** addr, void* assumed, void* value) {
    auto* a_addr = reinterpret_cast<unsigned long long int*>(addr);
    auto a_assumed = reinterpret_cast<unsigned long long int>(assumed);
    auto a_value = reinterpret_cast<unsigned long long int>(value);

    return reinterpret_cast<void*>(atomicCAS(a_addr, a_assumed, a_value));
  }

  template<typename T>
  __DEV__ void initialize_stream_compaction_array() {
    auto num_obj = num_objects_[TypeHelper<T>::kIndex];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      void* ptr = objects_[TypeHelper<T>::kIndex][i];
      bool object_deleted = false;

      // TODO: Can use binary search?
      for (int j = 0; j < num_deleted_objects_[TypeHelper<T>::kIndex]; ++j) {
        void*& deleted_obj_ptr = deleted_objects_[TypeHelper<T>::kIndex][j];
        if (ptr == deleted_obj_ptr) {
          // Remove pointer from deleted set with CAS because new objects can
          // be allocated in the location of deleted objects in the same
          // iteration.
          if (atomiCasPtr(&deleted_obj_ptr, ptr, nullptr) == ptr) {
            object_deleted = true;
            break;
          }
        }
      }

      // TODO: Is this really the best way? Check simulation paper.
      stream_compaction_array_[i] = object_deleted ? 0 : 1;
    }
  }

  template<typename T>
  __DEV__ void compact_object_array() {
    auto num_obj = num_objects_[TypeHelper<T>::kIndex];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      if (stream_compaction_array_[i] == 1) {
        // Retain element.
        new_objects_[TypeHelper<T>::kIndex][stream_compaction_output_[i]] =
            objects_[TypeHelper<T>::kIndex][i];
      }
    }
  }

  template<typename T>
  __DEV__ void update_object_count() {
    // Update counts.
    auto num_obj = num_objects_[TypeHelper<T>::kIndex];
    assert(num_obj < kMaxObjects);
    assert(num_obj > 0);

    auto new_num_obj = stream_compaction_array_[num_obj - 1]
                       + stream_compaction_output_[num_obj - 1];
    assert(new_num_obj < kMaxObjects);
    assert(new_num_obj
           == num_obj - num_deleted_objects_[TypeHelper<T>::kIndex]);

    num_objects_[TypeHelper<T>::kIndex] = new_num_obj;
    num_deleted_objects_[TypeHelper<T>::kIndex] = 0;

    // Swap arrays.
    void** tmp = objects_[TypeHelper<T>::kIndex];
    objects_[TypeHelper<T>::kIndex] = new_objects_[TypeHelper<T>::kIndex];
    new_objects_[TypeHelper<T>::kIndex] = tmp;
  }

  template<typename T>
  unsigned int num_objects() {
    return read_from_device<unsigned int>(
        &num_objects_[TypeHelper<T>::kIndex]);
  }


  // Size of largest type in bytes.
  static const int kLargestTypeSize = TupleHelper<Types...>::kLargestTypeSize;

  static const int kCudaBlockSize = 256;
  static const int kNumTypes = sizeof...(Types);

  // We are using too much memory for aux. data structures.
  // TODO: Find a better way.
  static const int kMaxObjects = N_Objects;
  static const unsigned int kInvalidObject =
      std::numeric_limits<unsigned int>::max();

  unsigned int num_objects_[kNumTypes];
  unsigned int num_deleted_objects_[kNumTypes];

  // Array of pointers to all objects of a type.
  // 2 arrays which are swapped during compaction.
  void** objects_[kNumTypes];
  void** new_objects_[kNumTypes];

  // Array of deleted objects of a type.
  void* deleted_objects_[kNumTypes][kMaxObjects];

  // Input to prefix sum.
  unsigned int stream_compaction_array_[kMaxObjects];

  // Result of prefix sum.
  unsigned int stream_compaction_output_[kMaxObjects];

  // Temporary storage.
  unsigned int stream_compaction_temp_[3*kMaxObjects];

  // Allocator-specific state.
  AllocatorStateT<ThisAllocator> allocator_state_;
};


template<typename AllocatorT>
class AllocatorHandle {
 public:
  AllocatorHandle(size_t allocator_heap_size = 0) {
    static_assert(sizeof(size_t) == 8, "Expected 64 bit system");
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, kMallocHeapSize);

    if (allocator_heap_size == 0) {
      allocator_heap_size = kMallocHeapSize;
    }

#ifndef NDEBUG
    size_t heap_size;
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    assert(heap_size >= kMallocHeapSize);
#endif  // NDEBUG

    size_t allocated_bytes = 0;

    gpuErrchk(cudaMalloc(&allocator_, sizeof(AllocatorT)));
    assert(allocator_ != nullptr);
    allocated_bytes += sizeof(AllocatorT);

#ifndef NDEBUG
    printf("Max. #objects: %lu\n", (long unsigned int) kMaxObjects);
#endif  // NDEBUG

    for (int i = 0; i < kNumTypes; ++i) {
      cudaMalloc(&dev_ptr_objects_[i], sizeof(void*)*kMaxObjects);
      cudaMalloc(&dev_ptr_new_objects_[i], sizeof(void*)*kMaxObjects);
      allocated_bytes += 2*sizeof(void*)*kMaxObjects;
    }

    cudaMemcpy(allocator_->objects_, dev_ptr_objects_,
               sizeof(void**)*kNumTypes, cudaMemcpyHostToDevice);
    cudaMemcpy(allocator_->new_objects_, dev_ptr_new_objects_,
               sizeof(void**)*kNumTypes, cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("Trying to allocate for do-all helper: %f MB\n",
           allocated_bytes / 1024.0f / 1024.0f);
#endif  // NDEBUG

    gpuErrchk(cudaDeviceSynchronize());

    for (int i = 0; i < kNumTypes; ++i) {
      assert(dev_ptr_objects_[i] != nullptr);
      assert(dev_ptr_new_objects_[i] != nullptr);
    }

    // Initialize allocator.
    allocator_->allocator_state_.initialize(allocator_heap_size);
  }

  ~AllocatorHandle() {
    cudaFree(allocator_);
  }

  long unsigned int DBG_get_enumeration_time() {
    return allocator_->DBG_get_enumeration_time();
  }

  template<class T, void(T::*func)()>
  void parallel_do() {
    allocator_->parallel_do<true, T, func>();
  }

  template<class T, typename P1, void(T::*func)(P1)>
  void parallel_do(P1 p1) {
    allocator_->parallel_do<true, T, P1, func>(p1);
  }

  template<class T, void(T::*func)()>
  void fast_parallel_do() {
    allocator_->parallel_do<false, T, func>();
  }

  template<class T, typename P1, void(T::*func)(P1)>
  void fast_parallel_do(P1 p1) {
    allocator_->parallel_do<false, T, P1, func>(p1);
  }

  // This is a no-op.
  template<class T>
  void parallel_defrag(int /*max_records*/, int /*min_records = 1*/) {}

  void DBG_print_state_stats() {
    kernel_print_state_stats<<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  void DBG_collect_stats() {}

  void DBG_print_collected_stats() {}

  // Returns a device pointer to the allocator.
  AllocatorT* device_pointer() { return allocator_; }

 private:
  static const int kNumTypes = AllocatorT::kNumTypes;
  static const int kMaxObjects = AllocatorT::kMaxObjects;

  AllocatorT* allocator_;

  // Device pointers: arrays of void*.
  void** dev_ptr_objects_[AllocatorT::kNumTypes];
  void** dev_ptr_new_objects_[AllocatorT::kNumTypes];
};


template<typename C, int FieldIndex>
class SoaField {
 private:
  using T = typename SoaFieldHelper<C, FieldIndex>::type;

  // Data stored in here.
  T data_;

  __DEV__ const T* data_ptr() const { return &data_; }
  __DEV__ T* data_ptr() { return &data_; }

 public:
  // Field initialization.
  __DEV__ SoaField() {}
  __DEV__ explicit SoaField(const T& value) : data_(value) {}

  // Explicit conversion for automatic conversion to base type.
  __DEV__ operator T&() { return data_; }
  __DEV__ operator const T&() const { return data_; }

  // Custom address-of operator.
  __DEV__ T* operator&() { return &data_; }
  __DEV__ const T* operator&() const { return &data_; }

  // Support member function calls.
  __DEV__ T& operator->() { return data_; }
  __DEV__ const T& operator->() const { return data_; }

  // Explicitly get value. Just for better code readability.
  __DEV__ T& get() { return *data_ptr(); }
  __DEV__ const T& get() const { return *data_ptr(); }

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

  template<typename U = T>
  __DEV__ const typename std::enable_if<is_device_array<U>::value,
                                        typename U::BaseType>::type&
  operator[](size_t pos) const { return data_[pos]; }

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    data_ = value;
    return data_;
  }
};


template<typename C, int FieldIndex>
using Field = SoaField<C, FieldIndex>;


// TODO: Is it safe to make these static?
template<typename AllocatorT, typename T>
__DEV__ __forceinline__ static void destroy(AllocatorT* allocator, T* ptr) {
  allocator->template free<T>(ptr);
}


template<typename AllocatorT, typename C, int FieldIndex>
__DEV__ __forceinline__ static void destroy(AllocatorT* allocator,
                                            const SoaField<C, FieldIndex>& value) {
  allocator->template free(value.get());
}

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
