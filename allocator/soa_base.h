#ifndef ALLOCATOR_SOA_BASE_H
#define ALLOCATOR_SOA_BASE_H

#include "allocator/soa_defrag.h"
#include "allocator/soa_helper.h"

// User-defined classes should inherit from this class.
template<class AllocatorT>
class SoaBase {
 public:
  using Allocator = AllocatorT;
  using BaseClass = void;
  static const bool kIsAbstract = false;

  __DEV__ uint8_t get_type() const { return AllocatorT::get_type(this); }

  template<typename ClassIterT, typename ScanClassT>
  __DEV__ void rewrite_object(AllocatorT* allocator, int num_records) {
    // Load records into shared memory.
    extern __shared__ DefragRecord<typename AllocatorT::BlockBitmapT> records[];
    assert(blockDim.x > num_records);
    if (threadIdx.x < num_records) {
      // One thread per record.
      records[threadIdx.x] = allocator->defrag_records_[threadIdx.x];
    }

    __syncthreads();

    SoaClassHelper<ScanClassT>::template dev_for_all<ClassIterT::FieldUpdater,
                                                     /*IterateBase=*/ true>(
        allocator, static_cast<ScanClassT*>(this), num_records);
  }
};

#endif  // ALLOCATOR_SOA_BASE_H
