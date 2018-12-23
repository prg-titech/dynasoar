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

  __DEV__ static uint8_t get_type(const void* ptr) {
    auto ptr_base = reinterpret_cast<uintptr_t>(ptr);
    uint8_t type_id = ptr_base >> 56;  // Truncated.
    return type_id;
  }
};

// Wrapper type for fields of SOA-structured classes. This class contains the
// logic for calculating the data location of a field from an object
// identifier.
template<typename C, int Field>
class SoaField {
 protected:
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

  __DEV__ volatile T* data_ptr() const volatile {
    return const_cast<const SoaField<C, Field>*>(this)->data_ptr();
  }

 public:
  __DEV__ static T* data_ptr_from_obj_ptr(C* obj) {
    uintptr_t ptr_base = reinterpret_cast<uintptr_t>(obj);

#ifndef NDEBUG
    // Check for nullptr.
    uintptr_t null_pointer_dereferenced = ptr_base;  // For better error msg.
    assert(null_pointer_dereferenced != 0);

    // Check if encoded type ID is consistent with type of pointer.
    assert(C::Allocator::template is_type<C>(obj));
#endif  // NDEBUG

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
  __DEV__ explicit SoaField(const T& value) { *this = value; }

  // Implicit conversion operator for automatic conversion to base type.
  __DEV__ operator T&() { return *data_ptr(); }
  __DEV__ operator const T&() const { return *data_ptr(); }
  __DEV__ operator volatile T&() volatile { return *data_ptr(); }
  __DEV__ operator const volatile T&() const volatile { return *data_ptr(); }

  // Force conversion.
  __DEV__ T& get() { return *data_ptr(); }
  __DEV__ const T& get() const { return *data_ptr(); }
  __DEV__ volatile T& get() volatile { return *data_ptr(); }
  __DEV__ const volatile T& get() const volatile { return *data_ptr(); }

  // Custom address-of operator.
  __DEV__ T* operator&() { return data_ptr(); }
  __DEV__ const T* operator&() const { return data_ptr(); }
  __DEV__ volatile T* operator&() volatile { return data_ptr(); }
  __DEV__ const volatile T* operator&() const volatile { return data_ptr(); }

  // Member function calls on fields of pointer types.
  template<typename U = T>
  __DEV__ typename std::enable_if<std::is_pointer<U>::value, T>::type&
  operator->() { return *data_ptr(); }

  template<typename U = T>
  __DEV__ const typename std::enable_if<std::is_pointer<U>::value, T>::type&
  operator->() const { return *data_ptr(); }

  template<typename U = T>
  __DEV__ volatile typename std::enable_if<std::is_pointer<U>::value, T>::type&
  operator->() volatile { return *data_ptr(); }

  template<typename U = T>
  __DEV__ const volatile typename std::enable_if<std::is_pointer<U>::value,
                                                 T>::type&
  operator->() const volatile{ return *data_ptr(); }

  // Member function calls on non-pointer types.
  template<typename U = T>
  __DEV__ typename std::enable_if<!std::is_pointer<U>::value, T*>::type
  operator->() { return data_ptr(); }

  template<typename U = T>
  __DEV__ typename std::enable_if<!std::is_pointer<U>::value, const T*>::type
  operator->() const { return data_ptr(); }

  template<typename U = T>
  __DEV__ typename std::enable_if<!std::is_pointer<U>::value,
                                  volatile T*>::type
  operator->() volatile { return data_ptr(); }

  template<typename U = T>
  __DEV__ typename std::enable_if<!std::is_pointer<U>::value,
                                  const volatile T*>::type
  operator->() const volatile { return data_ptr(); }

  // Dereference type in case of pointer type.
  __DEV__ typename std::remove_pointer<T>::type& operator*() {
    return **data_ptr();
  }
  __DEV__ typename std::remove_pointer<T>::type& operator*() const {
    return **data_ptr();
  }
  __DEV__ typename std::remove_pointer<T>::type& operator*() volatile {
    return **data_ptr();
  }
  __DEV__ typename std::remove_pointer<T>::type& operator*() const volatile {
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

  template<typename U = T>
  __DEV__ volatile typename std::enable_if<is_device_array<U>::value,
                                           typename U::BaseType>::type&
  operator[](size_t pos) volatile {
    return (*data_ptr())[pos];
  }

  template<typename U = T>
  __DEV__ const volatile typename std::enable_if<is_device_array<U>::value,
                                                 typename U::BaseType>::type&
  operator[](size_t pos) const volatile {
    return (*data_ptr())[pos];
  }

  // Assignment operator.
  __DEV__ T& operator=(const T& value) {
    *data_ptr() = value;
    return *data_ptr();
  }

  __DEV__ volatile T& operator=(const T& value) volatile {
    *data_ptr() = value;
    return *data_ptr();
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 4, U>::type
  atomic_cas(T assumed, T value) {
    auto* ptr_assumed = reinterpret_cast<unsigned int*>(&assumed);
    auto* ptr_value = reinterpret_cast<unsigned int*>(&value);
    auto* ptr_addr = reinterpret_cast<unsigned int*>(data_ptr());

    auto result = atomicCAS(ptr_addr, *ptr_assumed, *ptr_value);
    return *reinterpret_cast<U*>(&result);
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 8, U>::type
  atomic_cas(T assumed, T value) {
    auto* ptr_assumed = reinterpret_cast<unsigned long long int*>(&assumed);
    auto* ptr_value = reinterpret_cast<unsigned long long int*>(&value);
    auto* ptr_addr = reinterpret_cast<unsigned long long int*>(data_ptr());

    auto result = atomicCAS(ptr_addr, *ptr_assumed, *ptr_value);
    return *reinterpret_cast<U*>(&result);
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 4, U>::type
  atomic_read() {
    T dummy = 0;
    return atomic_cas(dummy, dummy);
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 8, U>::type
  atomic_read() {
    T dummy = 0;
    return atomic_cas(dummy, dummy);
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 4, void>::type
  atomic_write(T value) {
    auto* ptr_addr = reinterpret_cast<unsigned int*>(data_ptr());
    auto* ptr_value = reinterpret_cast<unsigned int*>(&value);
    atomicExch(ptr_addr, *ptr_value);
  }

  template<typename U = T>
  __DEV__ typename std::enable_if<sizeof(U) == 8, void>::type
  atomic_write(T value) {
    auto* ptr_addr = reinterpret_cast<unsigned long long int*>(data_ptr());
    auto* ptr_value = reinterpret_cast<unsigned long long int*>(&value);
    atomicExch(ptr_addr, *ptr_value);
  }

  __DEV__ volatile SoaField<C, Field>& as_volatile() {
    volatile SoaField<C, Field>* this_v = this;
    return *this_v;
  }

  __DEV__ const volatile SoaField<C, Field>& as_volatile() const {
    const volatile SoaField<C, Field>* this_v = this;
    return *this_v;
  }
};

#endif  // ALLOCATOR_SOA_FIELD_H
