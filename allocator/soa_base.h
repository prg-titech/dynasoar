#ifndef ALLOCATOR_SOA_BASE_H
#define ALLOCATOR_SOA_BASE_H

// User-defined classes should inherit from this class.
template<class AllocatorT>
class SoaBase {
 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;

  __DEV__ uint8_t get_type() {
    return AllocatorT::get_type(this);
  }
};

#endif  // ALLOCATOR_SOA_BASE_H
