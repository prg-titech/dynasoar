#ifndef ALLOCATOR_SOA_DEFRAG_RECORD_H
#define ALLOCATOR_SOA_DEFRAG_RECORD_H

static const int kMaxDefragRecords = 8192;

template<typename BitmapT>
struct DefragRecord {
  // Previous allocation bitmap. (Not free bitmap!)
  BitmapT source_bitmap;
  // New location in target bitmap. (For rewriting.)
  BitmapT target_bitmap[kDefragFactor];
  uint32_t source_block_idx;
  uint32_t target_block_idx[kDefragFactor];

  template<typename RecordsT>
  __DEV__ void copy_from(const RecordsT& records, int idx) {
    source_bitmap = records.source_bitmap[idx];
    source_block_idx = records.source_block_idx[idx];

    for (int i = 0; i < kDefragFactor; ++i) {
      target_bitmap[i] = records.target_bitmap[i][idx];
      target_block_idx[i] = records.target_block_idx[i][idx];
    }
  }
};

template<typename BitmapT, int N>
struct SoaDefragRecords {
  BitmapT source_bitmap[N];
  BitmapT target_bitmap[kDefragFactor][N];
  uint32_t source_block_idx[N];
  uint32_t target_block_idx[kDefragFactor][N];
};

#endif  // ALLOCATOR_SOA_DEFRAG_RECORD_H
