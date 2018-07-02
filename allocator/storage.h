#ifndef ALLOCATOR_STORAGE_H
#define ALLOCATOR_STORAGE_H

// Storage size: 1 GB.
static const uint64_t kStorageSize = 1073741824;

__device__ char storage[kStorageSize];

#endif  // ALLOCATOR_STORAGE_H
