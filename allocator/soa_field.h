#ifndef ALLOCATOR_SOA_FIELD_H
#define ALLOCATOR_SOA_FIELD_H

#include "allocator/configuration.h"

struct PointerHelper {
  __DEV__ static uint8_t obj_id_from_obj_ptr(const void* obj) {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(obj);
    return static_cast<uint8_t>(ptr_base)
        & static_cast<uint8_t>(0x3F);  // Truncated.
  }

  __DEV__ static char* block_base_from_obj_ptr(const void* obj) {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(obj);
    return reinterpret_cast<char*>(
        ptr_base & static_cast<uintptr_t>(0xFFFFFFFFFFC0));
  }

  // Replace block base and object ID in ptr.
  template<typename T>
  __DEV__ static T* rewrite_pointer(T* ptr, void* block_base,
                                    uint8_t obj_id) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    // Clear object ID and block base (48 bits).
    ptr_as_int &= ~((1ULL << 48) - 1);
    // Set object ID and block base.
    ptr_as_int |= obj_id;
    ptr_as_int |= reinterpret_cast<uintptr_t>(block_base);

    return reinterpret_cast<T*>(ptr_as_int);
  }
};

// Wrapper type for fields of SOA-structured classes. This class contains the
// logic for calculating the data location of a field from an object
// identifier.
template<typename C, int Field>
class SoaField {
 public:
  using T = typename SoaFieldHelper<C, Field>::type;

  // Calculate data pointer from address.
  __DEV__ T* data_ptr() const {
    // Base address of the pointer, i.e., without the offset of the SoaField
    // type.
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(this)
        - sizeof(SoaField<C, Field>)
            * (Field + SoaClassUtil<typename C::BaseClass>::kNumFields);
    return data_ptr_from_obj_ptr(reinterpret_cast<C*>(ptr_base));
  }

 public:
  __DEV__ static T* data_ptr_from_obj_ptr(C* obj) {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(obj);
    // Block size (N_T), i.e., number of object slots in this block.
    uint8_t block_size = ptr_base >> 48;  // Truncated.
    // Object slot ID.
    uint8_t obj_id = PointerHelper::obj_id_from_obj_ptr(obj);
    // Base address of the block.
    char* block_base = PointerHelper::block_base_from_obj_ptr(obj);
    return data_ptr_from_location(block_base, block_size, obj_id);
  }

  __DEV__ static T* data_ptr_from_location(char* block_base,
                                           uint8_t block_size,
                                           uint8_t obj_id) {
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

  // Custom address-of operator.
  __DEV__ T* operator&() { return data_ptr(); }
  __DEV__ T* operator&() const { return data_ptr(); }

  // Support member function calls.
  __DEV__ T& operator->() { return *data_ptr(); }
  __DEV__ T& operator->() const { return *data_ptr(); }

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

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    *data_ptr() = value;
    return *data_ptr();
  }
};

#endif  // ALLOCATOR_SOA_FIELD_H
