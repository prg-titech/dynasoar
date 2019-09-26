#ifndef ALLOCATOR_TYPE_SYSTEM_H
#define ALLOCATOR_TYPE_SYSTEM_H

#include "allocator/soa_base.h"

// Helper data structure for checking if a pointer (encoded type ID) belongs
// to BaseClass or one of its subclasses.
template<typename ThisAllocator, typename BaseClass>
struct PointerTypeChecker {
  // Iterating over all types T in the allocator.
  template<typename T>
  struct InnerHelper {
    // T is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      __device__ __host__ static bool call(const BaseClass* obj) {
        if (obj->get_type() == ThisAllocator::template BlockHelper<T>::kIndex) {
          return false;  // No need to check other types.
        } else {
          return true;   // true means "continue processing".
        }
      }
    };

    // T is not a subclass of BaseClass. Skip.
    template<int Dummy>
    struct ClassSelector<false, Dummy> {
      __device__ __host__ static bool call(const BaseClass* obj) {
        return true;
      }
    };

    __device__ __host__ bool operator()(const BaseClass* obj) {
      return ClassSelector<std::is_base_of<BaseClass, T>::value, 0>::call(obj);
    }
  };
};

#endif  // ALLOCATOR_TYPE_SYSTEM_H
