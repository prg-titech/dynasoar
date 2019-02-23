#ifndef ALLOCATOR_SOA_FIELD_H
#define ALLOCATOR_SOA_FIELD_H

#include "allocator/configuration.h"

struct PointerHelper {
  static const uint8_t kObjectIdBitmask = 0x3F;
  static const uintptr_t kBlockPtrBitmask = 0xFFFFFFFFFFC0;

  __DEV__ static ObjectIndexT obj_id_from_obj_ptr(const void* obj) {
    uint8_t result =
        static_cast<uint8_t>(reinterpret_cast<uintptr_t&>(obj))
        & kObjectIdBitmask;
    // Truncate and reinterpret as ObjectIndexT.
    return reinterpret_cast<ObjectIndexT&>(result);
  }

  __DEV__ static char* block_base_from_obj_ptr(const void* obj) {
    auto& ptr_base = reinterpret_cast<uintptr_t&>(obj);
    return reinterpret_cast<char*>(ptr_base & kBlockPtrBitmask);
  }

  // Replace block base and object ID in ptr.
  template<typename T>
  __DEV__ static T* rewrite_pointer(T* ptr, void* block_base,
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

  __DEV__ static TypeIndexT get_type(const void* ptr) {
    auto& ptr_base = reinterpret_cast<uintptr_t&>(ptr);
    uint8_t type_id = ptr_base >> 56;  // Truncated.
    return reinterpret_cast<TypeIndexT&>(type_id);
  }
};

// Wrapper type for fields of SOA-structured classes. This class contains the
// logic for calculating the data location of a field from an object
// identifier.
template<typename C, int Field>
class SoaField {
 private:
  using T = typename SoaFieldHelper<C, Field>::type;

  // Dummy field forces size of class to be zero.
  char _[0];

  // Calculate data pointer from address.
  __DEV__ T* data_ptr() const {
    // All SoaField<> have size 0. Offset of all fields is 1.
    auto ptr_base = reinterpret_cast<uintptr_t>(this) - sizeof(C); //- 1;
    return data_ptr_from_obj_ptr(reinterpret_cast<C*>(ptr_base));
  }

 public:
  __DEV__ static T* data_ptr_from_obj_ptr(C* obj) {
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

  template<int BlockSize>
  __DEV__ static T* data_ptr_from_location(char* block_base,
                                           ObjectIndexT obj_id) {
    assert(obj_id < BlockSize);
    // Address of SOA array.
    T* soa_array = reinterpret_cast<T*>(
        block_base + kBlockDataSectionOffset
        + BlockSize*SoaFieldHelper<C, Field>::kOffset);
    return soa_array + obj_id;
  }

  __DEV__ static T* data_ptr_from_location(char* block_base,
                                           ObjectIndexT block_size,
                                           ObjectIndexT obj_id) {
    assert(obj_id < block_size);
    // Address of SOA array.
    T* soa_array = reinterpret_cast<T*>(
        block_base + kBlockDataSectionOffset
        + block_size*SoaFieldHelper<C, Field>::kOffset);
    return soa_array + obj_id;
  }

  // Field initialization.
  __DEV__ SoaField() {}
  __DEV__ explicit SoaField(const T& value) { *data_ptr() = value; }

  // Implicit conversion operator for automatic conversion to base type.
  __DEV__ operator T&() { return *data_ptr(); }
  __DEV__ operator const T&() const { return *data_ptr(); }

  // Explicitly get value. Just for better code readability.
  __DEV__ T& get() { return *data_ptr(); }
  __DEV__ const T& get() const { return *data_ptr(); }

  // Custom address-of operator.
  __DEV__ T* operator&() { return data_ptr(); }
  __DEV__ const T* operator&() const { return data_ptr(); }

  // Support member function calls.
  __DEV__ T& operator->() { return *data_ptr(); }
  __DEV__ const T& operator->() const { return *data_ptr(); }

  // Dereference type in case of pointer type.
  __DEV__ typename std::remove_pointer<T>::type& operator*() {
    return **data_ptr();
  }
  __DEV__ typename std::remove_pointer<T>::type& operator*() const {
    return **data_ptr();
  }

  // Array access in case of device array.
  template<typename U = T>
  __DEV__ typename std::enable_if<is_device_array<U>::value,
                                  typename U::BaseType>::type&
  operator[](size_t pos) {
    return (*data_ptr())[pos];
  }

  template<typename U = T>
  __DEV__ const typename std::enable_if<is_device_array<U>::value,
                                        typename U::BaseType>::type&
  operator[](size_t pos) const {
    return (*data_ptr())[pos];
  }

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    *data_ptr() = value;
    return *data_ptr();
  }
};

#endif  // ALLOCATOR_SOA_FIELD_H
