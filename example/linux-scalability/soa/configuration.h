#ifndef EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H
#define EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H

static const int kTotalNumObjects = 62*64*64*64;
static const int kNumAllocPerThread = 500;
static const int kNumThreads = 256;
static const int kNumBlocks = 64;
static const int kNumIterations = 1;

#endif  // EXAMPLE_LINUX_SCALABILITY_CONFIGURATION_H
