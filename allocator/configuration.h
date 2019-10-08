#ifndef ALLOCATOR_AOS_CONFIGURATION_H
#define ALLOCATOR_AOS_CONFIGURATION_H

/**
 * This macro actives defragmentation support. DynaSOAr maintains
 * defragmentation candidate (block state) bitmaps and certain other data
 * structures only if this macro is defined.
 */
// #define OPTION_DEFRAG

/**
 * This macro instructs DynaSOAr to store only the source block IDs of all
 * defragmentation records in shared memory. All other defragmentation record
 * components are stored in global memory. If this macro is undefined, then
 * all defragmentation record components are stored in shared memory,
 * reducing the number of blocks that can be processed in one pass. If the
 * macro OPTION_DEFRAG_FORWARDING_POINTER is defined, then this macro
 * (defined or undefined) has no effect, because defragmentation records are
 * not recomputed.
 */
#define OPTION_DEFRAG_USE_GLOBAL


/**
 * Instructs DynaSOAr to store forwarding pointers in source blocks instead of
 * recomputing forwarding pointers on demand. (Default)
 */
#define OPTION_DEFRAG_FORWARDING_POINTER

/**
 * Only for benchmarks: Print additional statistics about defragmentation.
 */
//#define OPTION_DEFRAG_BENCH

/**
 * Defragmentation factor "n". See paper. E.g.:
 * - 1 -- 50% -- Merge 1 block into 1 block
 * - 2 -- 66% -- Merge 2 blocks into 1 block
 * - 3 -- 75% -- Merge 3 blocks into 1 block
 * leq_threshold = kDefragFactor / (kDefragFactor + 1)
 */
static const int kDefragFactor = 5;

/**
 * Leave at least that many block, i.e., do not defragment too much.
 */
static const int kMinDefragRetainBlocks = 16;

/**
 * Amount of padding (bytes) in blocks before the data segment. 64 bytes leave
 * enough space for object bitmaps and other fields.
 */
static const int kBlockDataSectionOffset = 64;

/**
 * Minimum number of guaranteed compactions for starting a defragmentation
 * pass. If a pass would eliminate less than kMinDefragCandidates many
 * candidates, DynaSOAr will not start a defragmentation pass.
 */
static const int kMinDefragCandidates = 1;

/**
 * Number of active block lookup attempts before a new block is initialized.
 */
static const int kFindActiveBlockRetries = 5;

/**
 * Data type for block indices/IDs.
 */
using BlockIndexT = int;

/**
 * Data type for type identifiers.
 */
using TypeIndexT = int8_t;
static_assert(sizeof(TypeIndexT) == 1, "Invalid TypeIndexT.");

/**
 * Data type for object IDs.
 */
using ObjectIndexT = int8_t;
static_assert(sizeof(ObjectIndexT) == 1, "Invalid ObjectIndexT.");

#endif  // ALLOCATOR_AOS_CONFIGURATION_H
