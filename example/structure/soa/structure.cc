#include <chrono>

#include "rendering.h"
#include "structure.h"

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

