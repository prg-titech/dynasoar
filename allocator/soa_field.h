#ifndef ALLOCATOR_SOA_FIELD_H
#define ALLOCATOR_SOA_FIELD_H

#include "allocator/configuration.h"

/**
 * Contains various helper functions for extracting components from a fake
 * pointer etc.
 */
struct PointerHelper {
  /**
   * Bit mask of bits used for encoding object slot IDs/indexes.
   */
  static const uint8_t kObjectIdBitmask = 0x3F;

  /**
   * Bit mask of bits used for encoding block locations.
   */
  static const uintptr_t kBlockPtrBitmask = 0xFFFFFFFFFFC0;

  /**
   * Bit mask of bits used for encoding memory location-related information.
   */
  static const uintptr_t kMemAddrBitmask = kBlockPtrBitmask | kObjectIdBitmask;

  /**
   * Extracts the object slot ID from a fake pointer.
   * @param obj Fake pointer
   */
  __host__ __device__ static ObjectIndexT obj_id_from_obj_ptr(
      const void* obj) {
    uint8_t result =
        static_cast<uint8_t>(reinterpret_cast<uintptr_t&>(obj))
        & kObjectIdBitmask;
    // Truncate and reinterpret as ObjectIndexT.
    return reinterpret_cast<ObjectIndexT&>(result);
  }

  /**
   * Extracts the block location from a fake pointer. Note: Blocks are always
   * located at such memory locations that the least significant
   * kObjectIdBitmask many bits are zero. 
   * @param obj Fake pointer
   */
  __host__ __device__ static char* block_base_from_obj_ptr(const void* obj) {
    auto& ptr_base = reinterpret_cast<uintptr_t&>(obj);
    return reinterpret_cast<char*>(ptr_base & kBlockPtrBitmask);
  }

  /**
   * Replace block location and object slot ID in \p ptr. The other components
   * of the fake pointer remain the same.
   * @param ptr Fake pointer
   * @param block_base New block location
   * @param obj_id New object slot ID
   * @tparam T Static type of the fake pointer
   */
  template<typename T>
  __host__ __device__ static T* rewrite_pointer(T* ptr, void* block_base,
                                                ObjectIndexT obj_id) {
    auto& ptr_as_int = reinterpret_cast<uintptr_t&>(ptr);
    // Clear object ID and block base (48 bits).
    ptr_as_int &= ~((1ULL << 48) - 1);
    // Set object ID and block base.
    auto& u_obj_id = reinterpret_cast<uint8_t&>(obj_id);
    ptr_as_int |= u_obj_id;
    ptr_as_int |= reinterpret_cast<uintptr_t&>(block_base);

    return reinterpret_cast<T*>(ptr_as_int);
  }

  /**
   * Extracts the type ID from a fake pointer.
   * @param ptr Fake pointer
   */
  __host__ __device__ static TypeIndexT get_type(const void* ptr) {
    auto& ptr_base = reinterpret_cast<uintptr_t&>(ptr);
    uint8_t type_id = ptr_base >> 56;  // Truncated.
    return reinterpret_cast<TypeIndexT&>(type_id);
  }
};

/**
 * Field proxy type for fields of SOA-structured classes. This class contains
 * the logic for calculating the data location of a field from a fake pointer,
 * along with various operator overloadings for a seamless integration in C++.
 * Note: This class contains most of the logic that is referred to as the
 * "data layout DSL" in the paper.
 * @tparam C Class in which the field is defined
 * @tparam FieldIndex Index of the field
 */
template<typename C, int FieldIndex>
class SoaField {
 private:
  /**
   * Type of field, as predeclared in the class definition.
   */
  using T = typename SoaFieldHelper<C, FieldIndex>::type;

  /**
   * By declaring an array of size 0, some compilers (e.g., GCC) treat this
   * class as C++ size 0 (i.e., sizeof(SoaField<...>) == 0). This simplifies
   * certain address computations and solves problems with zero initialization.
   * Technically, this is a hack and violates the C++ standard. Check WPMVP
   * paper for more details ("zero addressing mode").
   */
  char _[0];

  /**
   * Calculates the physical memory location of this field.
   */
  __host__ __device__ T* data_ptr() const {
    // All SoaField<> have size 0. Offset of all fields is 1.
    auto ptr_base = reinterpret_cast<uintptr_t>(this) - 1;
    return data_ptr_from_obj_ptr(reinterpret_cast<C*>(ptr_base));
  }

  /**
   * Calculates the physical memory location of this field.
   */
  __host__ __device__ volatile T* volatile_data_ptr() const {
    return data_ptr();
  }

 public:
  /**
   * Calculates the physical memory location of this field from a fake object
   * pointer.
   * @param obj Fake object pointer
   */
  __host__ __device__ static T* data_ptr_from_obj_ptr(C* obj) {
    auto& ptr_base = reinterpret_cast<uintptr_t&>(obj);

#ifndef NDEBUG
    // Check for nullptr.
    uintptr_t null_pointer_dereferenced = ptr_base;  // For better error msg.
    assert(null_pointer_dereferenced != 0);

    // Check if encoded type ID is consistent with type of pointer.
    assert(C::Allocator::template is_type<C>(obj));
#endif  // NDEBUG

    // Block size (N_T), i.e., number of object slots in this block.
    uint8_t u_block_size = ptr_base >> 48;  // Truncated.
    auto& block_size = reinterpret_cast<ObjectIndexT&>(u_block_size);

    // Object slot ID.
    ObjectIndexT obj_id = PointerHelper::obj_id_from_obj_ptr(obj);
    assert(obj_id < block_size);

    // Base address of the block.
    char* block_base = PointerHelper::block_base_from_obj_ptr(obj);
    return data_ptr_from_location(block_base, block_size, obj_id);
  }

  /**
   * Calculates the physical memory location of this field based on certain
   * given components.
   * @param block_base Address of the block
   * @param obj_id Object slot ID
   * @tparam BlockSize Block capacity
   */
  template<int BlockSize>
  __host__ __device__ static T* data_ptr_from_location(char* block_base,
                                                       ObjectIndexT obj_id) {
    assert(obj_id < BlockSize);
    // Address of SOA array.
    T* soa_array = reinterpret_cast<T*>(
        block_base + kBlockDataSectionOffset
        + BlockSize*SoaFieldHelper<C, FieldIndex>::kOffset);
    return soa_array + obj_id;
  }

  __host__ __device__ static T* data_ptr_from_location(char* block_base,
                                                       ObjectIndexT block_size,
                                                       ObjectIndexT obj_id) {
    assert(obj_id < block_size);
    // Address of SOA array.
    T* soa_array = reinterpret_cast<T*>(
        block_base + kBlockDataSectionOffset
        + block_size*SoaFieldHelper<C, FieldIndex>::kOffset);
    return soa_array + obj_id;
  }

  /**
   * Field initialization.
   * TODO: Force zero initialization of field? Seems like CUDA does not do this
   * by default.
   */
  __host__ __device__ SoaField() {}

  /**
   * Initializes the field. This constructor is invoked by field initializers.
   * TODO: Initializer for field types is missing.
   */
  __host__ __device__ explicit SoaField(const T& value) {
    *data_ptr() = value;
  }

  /**
   * Implicit conversion operator for automatic conversion to base type.
   * I.e., unwrapping the proxy.
   */
  __host__ __device__ operator T&() { return *data_ptr(); }

  /**
   * Implicit conversion operator for automatic conversion to base type.
   * I.e., unwrapping the proxy.
   */
  __host__ __device__ operator const T&() const { return *data_ptr(); }

  /**
   * Explicitly get value. Can be used internally. Just for better code
   * readability.
   */
  __host__ __device__ T& get() { return *data_ptr(); }

  /**
   * Explicitly get value. Can be used internally. Just for better code
   * readability.
   */
  __host__ __device__ const T& get() const { return *data_ptr(); }

  /**
   * Explicitly get value. Can be used internally. Just for better code
   * readability.
   */
  __host__ __device__ volatile T& volatile_get() {
    return *volatile_data_ptr();
  }

  /**
   * Explicitly get value. Can be used internally. Just for better code
   * readability.
   */
  __host__ __device__ const volatile T& volatile_get() const {
    return *volatile_data_ptr();
  }

  /**
   * Returns the address of the physical memory location of this field.
   */
  __host__ __device__ T* operator&() { return data_ptr(); }

  /**
   * Returns the address of the physical memory location of this field.
   */
  __host__ __device__ const T* operator&() const { return data_ptr(); }

  /**
   * This operator must be overloaded to support method calls if \p T is a
   * class/struct type. Similar to std::unique_ptr.
   */
  __host__ __device__ T& operator->() { return *data_ptr(); }

  /**
   * This operator must be overloaded to support method calls if \p T is a
   * class/struct type. Similar to std::unique_ptr.
   */
  __host__ __device__ const T& operator->() const { return *data_ptr(); }

  /**
   * Dereferences the value of this field if \p T is a pointer type.
   */
  __host__ __device__ typename std::remove_pointer<T>::type& operator*() {
    return **data_ptr();
  }

  /**
   * Dereferences the value of this field if \p T is a pointer type.
   */
  __host__ __device__ typename std::remove_pointer<T>::type& operator*()
      const {
    return **data_ptr();
  }

  /**
   * Allows this field to be used like an array.
   * @param pos Array index
   */
  template<typename U = T>
  __host__ __device__ typename std::enable_if<is_device_array<U>::value,
                                              typename U::BaseType>::type&
  operator[](size_t pos) {
    return (*data_ptr())[pos];
  }


  /**
   * Allows this field to be used like an array.
   * @param pos Array index
   */
  template<typename U = T>
  __host__ __device__ const typename std::enable_if<is_device_array<U>::value,
                                                    typename U::BaseType>
      ::type& operator[](size_t pos) const {
    return (*data_ptr())[pos];
  }

  /**
   * Assignment operator of assigning values of type \p T.
   * @param value Value to be assigned
   */
  __host__ __device__ T& operator=(const T& value) {
    *data_ptr() = value;
    return *data_ptr();
  }

  /**
   * Assignment operator of assigning values stored in another field.
   * TODO: Assignment operator for other field types is missing.
   * @param field Field to be assigned
   */
  __host__ __device__ T& operator=(const SoaField<C, FieldIndex>& field) {
    *data_ptr() = field.get();
    return *data_ptr();
  }
};


/**
 * Shortcut to SoaField.
 */
template<typename C, int FieldIndex>
using Field = SoaField<C, FieldIndex>;

#endif  // ALLOCATOR_SOA_FIELD_H
