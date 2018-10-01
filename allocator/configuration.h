#ifndef ALLOCATOR_AOS_CONFIGURATION_H
#define ALLOCATOR_AOS_CONFIGURATION_H

// Data section begins after 128 bytes. This leaves enough space for bitmaps
// and other data structures in blocks.
static const int kBlockDataSectionOffset = 64;

#endif  // ALLOCATOR_AOS_CONFIGURATION_H
