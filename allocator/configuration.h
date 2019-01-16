#ifndef ALLOCATOR_AOS_CONFIGURATION_H
#define ALLOCATOR_AOS_CONFIGURATION_H

// Active defragmentation support.
#define OPTION_DEFRAG

// Defragmentation factor.
// 1 -- 50% -- Merge 1 block into 1 block
// 2 -- 66% -- Merge 2 blocks into 1 block
// 3 -- 75% -- Merge 3 blocks into 1 block
static const int kDefragFactor = 2;

// Data section begins after 128 bytes. This leaves enough space for bitmaps
// and other data structures in blocks.
static const int kBlockDataSectionOffset = 64;

#endif  // ALLOCATOR_AOS_CONFIGURATION_H
