#ifndef ALLOCATOR_SOA_DEFRAG_H
#define ALLOCATOR_SOA_DEFRAG_H

template<typename BitmapT>
struct DefragRecord {
  // Previous allocation bitmap. (Not free bitmap!)
  BitmapT source_bitmap;
  // New location in target bitmap. (For rewriting.)
  BitmapT target_bitmap;
  uint32_t source_block_idx;
  uint32_t target_block_idx;
};

#endif  // ALLOCATOR_SOA_DEFRAG_H
