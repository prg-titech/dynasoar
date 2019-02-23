#ifndef ALLOCATOR_SOA_BASE_H
#define ALLOCATOR_SOA_BASE_H

#include "allocator/configuration.h"
#include "allocator/soa_block.h"
#include "allocator/soa_helper.h"
#include "allocator/soa_field.h"

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
  using FieldTypes = std::tuple<__VA_ARGS__>; \
  char _ ## classname;

// TODO: Is it safe to make these static?
template<typename AllocatorT, typename T>
__DEV__ __forceinline__ static void destroy(AllocatorT* allocator, T* ptr) {
  allocator->template free<T>(ptr);
}


template<typename AllocatorT, typename C, int Field>
__DEV__ __forceinline__ static void destroy(AllocatorT* allocator,
                                     const SoaField<C, Field>& value) {
  allocator->template free(value.get());
}


// User-defined classes should inherit from this class.
template<class AllocatorT>
class SoaBase {
 private:
  // Class should have size 1.
  char _;

 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;

  __DEV__ TypeIndexT get_type() const { return AllocatorT::get_type(this); }

  __DEV__ void DBG_print_ptr_decoding() const {
    char* block_ptr = PointerHelper::block_base_from_obj_ptr(this);
    int obj_id = PointerHelper::obj_id_from_obj_ptr(this);
    printf("%p = Block %p  |  Object %i\n", this, block_ptr, obj_id);
  }

  template<typename T>
  __DEV__ T* cast() {
    if (this != nullptr
        && get_type() == AllocatorT::template BlockHelper<T>::kIndex) {
      return static_cast<T*>(this);
    } else {
      return nullptr;
    }
  }

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  __DEV__ SoaBase<AllocatorT>** forwarding_pointer_address() const {
    char* block_base = PointerHelper::block_base_from_obj_ptr(this);
    // Address of SOA array.
    auto* soa_array = reinterpret_cast<SoaBase<AllocatorT>**>(
        block_base + kBlockDataSectionOffset);
    return soa_array + PointerHelper::obj_id_from_obj_ptr(this);
  }

  __DEV__ SoaBase<AllocatorT>* get_forwarding_pointer() const {
    return *forwarding_pointer_address();
  }

  __DEV__ void set_forwarding_pointer(SoaBase<AllocatorT>* ptr) {
    *forwarding_pointer_address() = ptr;
  }
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
};

#endif  // ALLOCATOR_SOA_BASE_H
