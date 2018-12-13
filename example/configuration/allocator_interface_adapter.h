#ifndef EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
#define EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H

#include <assert.h>
#include <cub/cub.cuh>

#include "allocator/tuple_helper.h"


// Defined by custom allocator.
void initialize_custom_allocator();


// Reads value at a device address and return it.
template<typename T>
T read_from_device(T* ptr) {
  T host_storage;
  cudaMemcpy(&host_storage, ptr, sizeof(T), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
  return host_storage;
}


template<typename AllocatorT, typename T>
__global__ void kernel_init_stream_compaction(AllocatorT* allocator) {
  allocator->template initialize_stream_compaction_array<T>();
}


template<typename AllocatorT, typename T>
__global__ void kernel_compact_object_array(AllocatorT* allocator) {
  allocator->template compact_object_array<T>();
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


// TODO: Fix visiblity.
template<uint32_t N_Objects, class... Types>
class SoaAllocator {
 public:
  using ThisAllocator = SoaAllocator<N_Objects, Types...>;

  // Zero-initialization of arrays can take a long time.
  __DEV__ SoaAllocator() = delete;

  template<typename T>
  struct TypeId {
    static const uint8_t value = TYPE_INDEX(Types..., T);
  };

  template<class T, void(T::*func)()>
  void parallel_do() {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelper<ThisAllocator, T, func>
        ::template InnerHelper>(this);
  }

  template<class T, class BaseClass, void(BaseClass::*func)()>
  void parallel_do_single_type() {
    // Get total number of objects.
    unsigned int num_objects = this->template num_objects<T>();

    if (num_objects > 0) {
      kernel_init_stream_compaction<ThisAllocator, T><<<128, 128>>>(this);
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
      kernel_compact_object_array<ThisAllocator, T><<<128, 128>>>(this);
      gpuErrchk(cudaDeviceSynchronize());

      // Update arrays and counts.
      kernel_update_object_count<ThisAllocator, T><<<1, 1>>>(this);
      gpuErrchk(cudaDeviceSynchronize());

      num_objects = this->template num_objects<T>();

      kernel_parallel_do_single_type<ThisAllocator, T, BaseClass, func><<<
          (num_objects + kCudaBlockSize - 1)/kCudaBlockSize,
          kCudaBlockSize>>>(this, num_objects);
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  // Pass in num_obj as parameter because this kernel might create new
  // objects and thus change the number of objects.
  template<class T, class BaseClass, void(BaseClass::*func)()>
  __DEV__ void parallel_do_single_type(unsigned int num_obj) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      T* obj = reinterpret_cast<T*>(objects_[TYPE_INDEX(Types..., T)][i]);
      (obj->*func)();
    }
  }

  // Call a member function on all objects of type.
  // Device version (sequential).
  // TODO: This does not enumerate subtypes.
  template<class T, typename F, typename... Args>
  __DEV__ void device_do(F func, Args... args) {
    auto num_obj = num_objects_[TYPE_INDEX(Types..., T)];

    for (int i = 0; i < num_obj; ++i) {
      T* obj = reinterpret_cast<T*>(objects_[TYPE_INDEX(Types..., T)][i]);
      (obj->*func)(args...);
    }
  }

  template<class T, typename... Args>
  __DEV__ T* external_allocator_make_new(Args... args);

  template<class T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    // Add object to pointer array.
    T* result = external_allocator_make_new<T>(args...);
    result->set_type(TYPE_INDEX(Types..., T));
    auto pos = atomicAdd(&num_objects_[TYPE_INDEX(Types..., T)], 1);
    objects_[TYPE_INDEX(Types..., T)][pos] = result;
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
          if (obj->get_type() == TYPE_INDEX(Types..., T)) {
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
  __DEV__ void external_allocator_free(T* obj);

  template<class T>
  __DEV__ void free(T* obj) {
    uint8_t type_id = obj->get_type();
    if (type_id == TYPE_INDEX(Types..., T)) {
      free_typed(obj);
    } else {
      bool result = TupleHelper<Types...>
          ::template dev_for_all<FreeHelper<T>::InnerHelper>(this, obj);
      assert(result);  // true means type was found.
    }
  }

  template<class T>
  __DEV__ void free_typed(T* obj) {
    auto pos = atomicAdd(&num_deleted_objects_[TYPE_INDEX(Types..., T)], 1);
    deleted_objects_[TYPE_INDEX(Types..., T)][pos] = obj;
    external_allocator_free<T>(obj);
  }

  // TODO: Implement missing DBG functions.
  template<class T>
  __DEV__ uint32_t DBG_allocated_slots() { return 0; }

  template<class T>
  __DEV__ uint32_t DBG_used_slots() { 
    return num_objects_[TYPE_INDEX(Types..., T)];
  }

  static void DBG_print_stats() {}

  __DEV__ void DBG_print_state_stats() {}

  template<typename T>
  __DEV__ void initialize_stream_compaction_array() {
    auto num_obj = num_objects_[TYPE_INDEX(Types..., T)];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      void* ptr = objects_[TYPE_INDEX(Types..., T)][i];
      bool object_deleted = false;

      // Check if ptr is in deleted object set (binary search).
      // TODO: Store deleted set in shared memory if possible.
      // TODO: Implement binary search.
      for (int j = 0; j < num_deleted_objects_[TYPE_INDEX(Types..., T)]; ++j) {
        if (ptr == deleted_objects_[TYPE_INDEX(Types..., T)][j]) {
          object_deleted = true;
          break;
        }
      }

      // TODO: Is this really the best way? Check simulation paper.
      stream_compaction_array_[i] = object_deleted ? 0 : 1;
    }
  }

  template<typename T>
  __DEV__ void compact_object_array() {
    auto num_obj = num_objects_[TYPE_INDEX(Types..., T)];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < num_obj; i += blockDim.x * gridDim.x) {
      if (stream_compaction_array_[i] == 1) {
        // Retain element.
        new_objects_[TYPE_INDEX(Types..., T)][stream_compaction_output_[i]] =
            objects_[TYPE_INDEX(Types..., T)][i];
      }
    }
  }

  template<typename T>
  __DEV__ void update_object_count() {
    // Update counts.
    auto num_obj = num_objects_[TYPE_INDEX(Types..., T)];
    assert(num_obj < kMaxObjects);
    assert(num_obj > 0);

    auto new_num_obj = stream_compaction_array_[num_obj - 1]
                       + stream_compaction_output_[num_obj - 1];
    assert(new_num_obj < kMaxObjects);

    num_objects_[TYPE_INDEX(Types..., T)] = new_num_obj;
    num_deleted_objects_[TYPE_INDEX(Types..., T)] = 0;

    // Swap arrays.
    void** tmp = objects_[TYPE_INDEX(Types..., T)];
    objects_[TYPE_INDEX(Types..., T)] = new_objects_[TYPE_INDEX(Types..., T)];
    new_objects_[TYPE_INDEX(Types..., T)] = tmp;
  }

  template<typename T>
  unsigned int num_objects() {
    return read_from_device<unsigned int>(
        &num_objects_[TYPE_INDEX(Types..., T)]);
  }


  static const int kCudaBlockSize = 256;
  static const int kNumTypes = sizeof...(Types);
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
};


template<typename AllocatorT>
class AllocatorHandle {
 public:
  AllocatorHandle() {
    initialize_custom_allocator();
    size_t allocated_bytes = 0;

    gpuErrchk(cudaMalloc(&allocator_, sizeof(AllocatorT)));
    assert(allocator_ != nullptr);
    allocated_bytes += sizeof(AllocatorT);

    for (int i = 0; i < kNumTypes; ++i) {
      gpuErrchk(cudaMalloc(&dev_ptr_objects_[i], sizeof(void*)*kMaxObjects));
      assert(dev_ptr_objects_[i] != nullptr);
      gpuErrchk(cudaMalloc(&dev_ptr_new_objects_[i],
                           sizeof(void*)*kMaxObjects));
      assert(dev_ptr_new_objects_[i] != nullptr);
      allocated_bytes += 2*sizeof(void*)*kMaxObjects;
    }

    cudaMemcpy(allocator_->objects_, dev_ptr_objects_,
               sizeof(void**)*kNumTypes, cudaMemcpyHostToDevice);
    cudaMemcpy(allocator_->new_objects_, dev_ptr_new_objects_,
               sizeof(void**)*kNumTypes, cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
    printf("Allocated bytes for do-all helper: %f MB\n",
           allocated_bytes / 1024.0f / 1024.0f);
#endif
  }

  ~AllocatorHandle() {
    cudaFree(allocator_);
  }

  // TODO: This function does not enumerate subtypes.
  template<class T, void(T::*func)()>
  void parallel_do() {
    allocator_->parallel_do<T, func>();
  }

  // This is a no-op.
  template<class T>
  void parallel_defrag(int /*max_records*/, int /*min_records = 1*/) {}

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

#endif  // EXAMPLE_CONFIGURATION_CUDA_ALLOCATOR_H
