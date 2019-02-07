#ifndef ALLOCATOR_SOA_DEFRAG_RECORD_H
#define ALLOCATOR_SOA_DEFRAG_RECORD_H

template<typename BitmapT>
struct DefragRecord {
#ifndef OPTION_DEFRAG_USE_GLOBAL
  // Store only source block idx in use_global mode.
  // Previous allocation bitmap. (Not free bitmap!)
  BitmapT source_bitmap;
  // New location in target bitmap. (For rewriting.)
  BitmapT target_bitmap[kDefragFactor];
#endif  // OPTION_DEFRAG_USE_GLOBAL

  BlockIndexT source_block_idx;

#ifndef OPTION_DEFRAG_USE_GLOBAL
  // Store only source block idx in use_global mode.
  BlockIndexT target_block_idx[kDefragFactor];
#endif   // OPTION_DEFRAG_USE_GLOBAL

  template<typename RecordsT>
  __DEV__ void copy_from(const RecordsT& records, int idx) {
#ifndef OPTION_DEFRAG_USE_GLOBAL
    source_bitmap = records.source_bitmap[idx];
#endif  // OPTION_DEFRAG_USE_GLOBAL

    source_block_idx = records.source_block_idx[idx];

#ifndef OPTION_DEFRAG_USE_GLOBAL
    for (int i = 0; i < kDefragFactor; ++i) {
      target_bitmap[i] = records.target_bitmap[i][idx];
      target_block_idx[i] = records.target_block_idx[i][idx];
    }
#endif  // OPTION_DEFRAG_USE_GLOBAL
  }
};

template<typename BitmapT, int N>
struct SoaDefragRecords {
  BitmapT source_bitmap[N];
  BitmapT target_bitmap[kDefragFactor][N];
  BlockIndexT source_block_idx[N];
  BlockIndexT target_block_idx[kDefragFactor][N];
};

static const int kSharedMemorySize = 48*1024;

#if defined(OPTION_DEFRAG_FORWARDING_POINTER)
// Shared memory is not used.
static const int kMaxDefragRecords = 64*64*64;
#elif defined(OPTION_DEFRAG_USE_GLOBAL)
// Store only source_block_idx in shared memory.
static const int kMaxDefragRecords = kSharedMemorySize / sizeof(BlockIndexT);
#else
// Maximum number of defragmentation records.
static const int kMaxDefragRecords = kSharedMemorySize
    / sizeof(DefragRecord<unsigned long long int>);
#endif  // OPTION_DEFRAG_USE_GLOBAL

#endif  // ALLOCATOR_SOA_DEFRAG_RECORD_H
