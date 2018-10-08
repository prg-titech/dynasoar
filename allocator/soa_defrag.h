#ifndef ALLOCATOR_SOA_DEFRAG_H
#define ALLOCATOR_SOA_DEFRAG_H

template<typename BitmapT>
struct DefragRecord {
  BitmapT source_bitmap;
  BitmapT target_bitmap;
  uint32_t source_block_idx;
  uint32_t target_block_idx;
};

#endif  // ALLOCATOR_SOA_DEFRAG_H
