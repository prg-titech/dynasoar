#ifndef ALLOCATOR_AOS_CONFIGURATION_H
#define ALLOCATOR_AOS_CONFIGURATION_H

// Active defragmentation support.
#define OPTION_DEFRAG

// Store defragmentation records in global memory.
#define OPTION_DEFRAG_USE_GLOBAL

#define OPTION_DEFRAG_FORWARDING_POINTER

// Defragmentation factor.
// 1 -- 50% -- Merge 1 block into 1 block
// 2 -- 66% -- Merge 2 blocks into 1 block
// 3 -- 75% -- Merge 3 blocks into 1 block
// leq_threshold = kDefragFactor / (kDefragFactor + 1)
static const int kDefragFactor = 1;

// Leave at least that many block, i.e., do not defragment too much.
static const int kMinDefragRetainBlocks = 128;

// Data section begins after 128 bytes. This leaves enough space for bitmaps
// and other data structures in blocks.
static const int kBlockDataSectionOffset = 64;

static const int kMinDefragCandidates = 512;

static const int kEffectiveMinDefragCandidates =
    kMinDefragCandidates - kMinDefragRetainBlocks > 1
    ? kMinDefragCandidates - kMinDefragRetainBlocks : 1;

static const int kFindActiveBlockRetries = 5;

using BlockIndexT = int;

using TypeIndexT = int8_t;
static_assert(sizeof(TypeIndexT) == 1, "Invalid TypeIndexT.");

using ObjectIndexT = int8_t;
static_assert(sizeof(ObjectIndexT) == 1, "Invalid ObjectIndexT.");

#endif  // ALLOCATOR_AOS_CONFIGURATION_H
