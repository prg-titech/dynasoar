#ifndef EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H

static const int kSeed = 42;
static const int kMaxVelocity = 10;
static const int kMaxDegree = 4;
static const int kNumIntersections = 500;
static const float kCellLength = 0.005f;
static const float kProducerRatio = 0.02f;
static const float kTargetRatio = 0.002f;
static const int kNumIterations = 2000;
static const float kCarAllocationRatio = 0.1f;
static const bool kOptionRender = false;

// Only for baseline version.
static const int kMaxNumCells = 150000;
static const int kMaxNumCars = 150000;

#endif  // EXAMPLE_TRAFFIC_SOA_CONFIGURATION_H
