#ifndef EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H

static const int kSeed = 42;
static const int kMaxVelocity = 10;
static const int kMaxDegree = 4;
static const int kNumIntersections = 17500;
static const float kCellLength = 0.005f;
static const float kProducerRatio = 0.02f;
static const float kTargetRatio = 0.002f;
static const int kNumIterations = 1000;
static const float kCarAllocationRatio = 0.02f;
static const bool kOptionRender = false;
static const bool kOptionPrintStats = true;

// Only for baseline version.
static const int kMaxNumCells = 5000000;
static const int kMaxNumCars = 5000000;

#endif  // EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H
