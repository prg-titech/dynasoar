#ifndef EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H
#define EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H

#ifndef PARAM_MAX_OBJ
// Note: The internal data structures of DynaSOAr (e.g., bitmaps) also require
// memory. We have to take this into account when comparing memory usage etc.
static const int kNumObjects = 63*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#ifdef PARAM_SIZE
static const int kNumAllocPerThread = PARAM_SIZE;
#else
static const int kNumAllocPerThread = 1024;
#endif  // PARAM_SIZE

static const int kNumThreads = 256;
static const int kNumBlocks = 64;
static const int kNumIterations = 1;

#endif  // EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H
