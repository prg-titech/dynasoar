// #define NDEBUG

#include <iostream>
#include <stdio.h>


#include "allocator/block.h"
#include "allocator/block_manager.h"
#include "allocator/storage.h"

struct DummyClass : public AosoaLayoutBase<DummyClass> {
  using FieldsTuple = std::tuple<int, double, char, char>;
  static const uint32_t kSoaSize = 64;
};


__global__ void kernel(uintptr_t* l) {
  uintptr_t block_loc = reinterpret_cast<uintptr_t>(storage);
  block_loc = ((block_loc + kMaxId - 1) / kMaxId) * kMaxId;

  DummyClass::initialize_block(block_loc);

/*
  uintptr_t x;
  for (int i = 0; i < 64; ++i) {
    x = DummyClass::try_allocate_in_block(block_loc);
  }

  BlockHeader::from_block(block_loc).print_debug();

  DummyClass::get<0>(x) = 123;
  printf("ALLOC RESULT: %p\n", x);
  */
}

int main() {
  kernel<<<1,1>>>(nullptr);
  gpuErrchk(cudaDeviceSynchronize());
}
