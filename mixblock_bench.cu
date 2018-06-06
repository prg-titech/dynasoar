#include <iostream>
#include <stdio.h>
#include <tuple>
#include <assert.h>
#include <limits>
#include <typeinfo>

template<int NumTypes, typename IndexType>
struct MixblockHeader {
  // Offset in bytes, indicating the beginning of a data section for a
  // specific type.
  uint32_t offset[NumTypes];

  // Stored in header, because only needed for allocation.
  IndexType free[NumTypes];

};
