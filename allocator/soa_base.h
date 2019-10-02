#ifndef ALLOCATOR_SOA_BASE_H
#define ALLOCATOR_SOA_BASE_H

#include "allocator/configuration.h"
#include "allocator/soa_block.h"
#include "allocator/soa_helper.h"
#include "allocator/soa_field.h"


#if GCC_COMPILER
/**
 * Predeclares all field types of a class.
 * @param classname Name of class
 * @param ... Types, separated by comma
 */
#define declare_field_types(classname, ...) \
  __device__ void* operator new(size_t sz, typename classname::Allocator* allocator) { \
    return allocator->allocate_new<classname>(); \
  } \
  __device__ void* operator new(size_t sz, classname* ptr) { \
    return ptr; \
  } \
  __device__ void operator delete(void* ptr, typename classname::Allocator* allocator) { \
    allocator->free<classname>(reinterpret_cast<classname*>(ptr)); \
  } \
  __device__ void operator delete(void*, classname*) { \
    assert(false);  \
  } \
  using FieldTypes = std::tuple<__VA_ARGS__>;
#else
#warning Using untested compiler. GCC recommended.
template<typename T>
__device__ void* __dynasoar_op_new(void* allocator) {
  return reinterpret_cast<typename T::Allocator*>(allocator)->template allocate_new<T>();
}

template<typename T>
__device__ void __dynasoar_op_delete(void* ptr, void* allocator) {
  reinterpret_cast<typename T::Allocator*>(allocator)->template free<T>(
      reinterpret_cast<T*>(ptr));
}

#define declare_field_types(classname, ...) \
  __device__ void* operator new(size_t sz, void* allocator) { \
    return __dynasoar_op_new<classname>(allocator); \
  } \
  __device__ void* operator new(size_t sz, classname* ptr) { \
    return ptr; \
  } \
  __device__ void operator delete(void* ptr, void* allocator) { \
    __dynasoar_op_delete<classname>(ptr, allocator); \
  } \
  __device__ void operator delete(void*, classname*) { \
    assert(false);  /* Construct must not throw exceptions. */ \
  } \
  using FieldTypes = std::tuple<__VA_ARGS__>;
#endif  // GCC_COMPILER


/**
 * Deletes an object of type or subtype \p T.
 * @param allocator Allocator that was used for allocating the object
 * @param ptr Object pointer
 * @tparam AllocatorT Allocator type
 * @tparam Type of object
 */
template<typename AllocatorT, typename T>
__device__ __forceinline__ static void destroy(AllocatorT* allocator, T* ptr) {
  allocator->template free<T>(ptr);
}


/**
 * Deletes an object of type or subtype \p T, where the object pointer is
 * stored in a field.
 * @param allocator Allocator that was used for allocating the object
 * @param value Field containing the pointer
 * @tparam AllocatorT Allocator type
 * @tparam C Class containing the field
 * @tparam FieldIndex Index of field within \p C
 */
template<typename AllocatorT, typename C, int FieldIndex>
__device__ __forceinline__ static void destroy(
    AllocatorT* allocator, const SoaField<C, FieldIndex>& value) {
  allocator->template free(value.get());
}


/**
 * All user-defined classes that are under control of the allocator should
 * inherit from this class; either directly, or by inheriting from a class that
 * inherits from this class at some point in its inheritance chain. This class
 * provides basic functionality such as a type cast operation and rewrite
 * logic for memory defragmentation.
 * @tparam AllocatorT Allocator type
 */
template<class AllocatorT>
class SoaBase {
 private:
  /**
   * Dummy field. Ensures that the C++ size of this class is 1.
   */
  char _;

 public:
  /**
   * Publicly accessible allocator type alias.
   */
  using Allocator = AllocatorT;

  /**
   * Subclasses that inherit from a class other than SoaBase must override
   * this type alias with the class that they are inheriting from.
   */
  using BaseClass = void;

  /**
   * Abstract classes should override this value with true.
   */
  static const bool kIsAbstract = false;

  /**
   * Returns the type ID of this object.
   */
  __device__ __host__ TypeIndexT get_type() const {
    return AllocatorT::get_type(this);
  }

  /**
   * Prints some information that is encoded in the (fake) location of this
   * object.
   */
  __device__ __host__ void DBG_print_ptr_decoding() const {
    char* block_ptr = PointerHelper::block_base_from_obj_ptr(this);
    int obj_id = PointerHelper::obj_id_from_obj_ptr(this);
    printf("%p = Block %p  |  Object %i\n", this, block_ptr, obj_id);
  }

  /**
   * Casts this object to an object of different type. Returns nullptr if this
   * object is not an object of the request type \p T. Note: This function is
   * similar to dynamic_cast<T*>.
   * @tparam T Requested type
   */
  template<typename T>
  __device__ __host__ T* cast() {
    if (this != nullptr
        && get_type() == AllocatorT::template BlockHelper<T>::kIndex) {
      return static_cast<T*>(this);
    } else {
      return nullptr;
    }
  }

  /**
   * Casts this object to an object of different type. Returns nullptr if this
   * object is not an object of the request type \p T. Note: This function is
   * similar to dynamic_cast<T*>.
   * @tparam T Requested type
   */
  template<typename T>
  __device__ __host__ const T* cast() const {
    if (this != nullptr
        && get_type() == AllocatorT::template BlockHelper<T>::kIndex) {
      return static_cast<const T*>(this);
    } else {
      return nullptr;
    }
  }

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  __device__ SoaBase<AllocatorT>** forwarding_pointer_address() const {
    char* block_base = PointerHelper::block_base_from_obj_ptr(this);
    // Address of SOA array.
    auto* soa_array = reinterpret_cast<SoaBase<AllocatorT>**>(
        block_base + kBlockDataSectionOffset);
    return soa_array + PointerHelper::obj_id_from_obj_ptr(this);
  }

  __device__ SoaBase<AllocatorT>* get_forwarding_pointer() const {
    return *forwarding_pointer_address();
  }

  __device__ void set_forwarding_pointer(SoaBase<AllocatorT>* ptr) {
    *forwarding_pointer_address() = ptr;
  }
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
};

#endif  // ALLOCATOR_SOA_BASE_H
