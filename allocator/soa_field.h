#ifndef ALLOCATOR_SOA_FIELD_H
#define ALLOCATOR_SOA_FIELD_H

#include "allocator/configuration.h"

// Wrapper type for fields of SOA-structured classes. This class contains the
// logic for calculating the data location of a field from an object
// identifier.
template<typename T, int Field, int Offset>
class SoaField {
 private:
  // Calculate data pointer from address.
  __DEV__ T* data_ptr() const {
    // Base address of the pointer, i.e., without the offset of the SoaField
    // type.
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(this)
        - sizeof(SoaField<T, Field, Offset>)*Field;
    // Block size (N_T), i.e., number of object slots in this block.
    uint8_t block_size = ptr_base >> 48;  // Truncated.
    // Object slot ID.
    uint8_t obj_id = static_cast<uint8_t>(ptr_base)
        & static_cast<uint8_t>(0x3F);  // Truncated.
    // Base address of the block.
    uintptr_t block_base = ptr_base & static_cast<uintptr_t>(0xFFFFFFFFFFC0);
    assert(obj_id < block_size);
    // Address of SOA array.
    T* soa_array = reinterpret_cast<T*>(
        block_base + kBlockDataSectionOffset + block_size*Offset);
    return soa_array + obj_id;
  }

 public:
  // Field initialization.
  __DEV__ SoaField() {}
  __DEV__ explicit SoaField(const T& value) { *data_ptr() = value; }

  // Explicit conversion for automatic conversion to base type.
  __DEV__ operator T&() { return *data_ptr(); }
  __DEV__ operator const T&() const { return *data_ptr(); }

  // Custom address-of operator.
  __DEV__ T* operator&() { return data_ptr(); }

  // Support member function calls.
  __DEV__ T operator->() { return *data_ptr(); }

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    *data_ptr() = value;
    return *data_ptr();
  }
};

#endif  // ALLOCATOR_SOA_FIELD_H
