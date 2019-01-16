#ifndef ALLOCATOR_SOA_DEFRAG_RECORD_H
#define ALLOCATOR_SOA_DEFRAG_RECORD_H

static const int kMaxDefragRecords = 8192;

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

  template<typename RecordsT>
  __DEV__ void copy_from(const RecordsT& records, int idx) {
    source_bitmap = records.source_bitmap[idx];
    target_bitmap = records.target_bitmap[idx];
    source_block_idx = records.source_block_idx[idx];
    target_block_idx = records.target_block_idx[idx];
  }
};

template<typename BitmapT, int N>
struct SoaDefragRecords {
  BitmapT source_bitmap[N];
  BitmapT target_bitmap[N];
  uint32_t source_block_idx[N];
  uint32_t target_block_idx[N];

  template<typename RecordsT>
  __DEV__ void copy_from(int dest, const RecordsT& records, int idx) {
    source_bitmap[dest] = records.source_bitmap[idx];
    target_bitmap[dest] = records.target_bitmap[idx];
    source_block_idx[dest] = records.source_block_idx[idx];
    target_block_idx[dest] = records.target_block_idx[idx];
  }
};

#endif  // ALLOCATOR_SOA_DEFRAG_RECORD_H
