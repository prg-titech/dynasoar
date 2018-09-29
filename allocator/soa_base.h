#ifndef ALLOCATOR_SOA_BASE_H
#define ALLOCATOR_SOA_BASE_H

// User-defined classes should inherit from this class.
template<class AllocatorT>
class SoaBase {
 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;
};

#endif  // ALLOCATOR_SOA_BASE_H
