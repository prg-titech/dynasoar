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

  __DEV__ bool operator==(const DefragRecord<BitmapT>& other) const {
    return source_bitmap == other.source_bitmap
        && target_bitmap == other.target_bitmap
        && source_block_idx == other.source_block_idx
        && target_block_idx == other.target_block_idx;
  }
};

#endif  // ALLOCATOR_SOA_DEFRAG_H
