#ifndef EXAMPLE_TRAFFIC_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_CONFIGURATION_H

static const int kSeed = 42;
static const int kMaxVelocity = 10;
static const int kMaxDegree = 4;

#ifndef PARAM_SIZE
static const int kNumIntersections = 17500;
#else
static const int kNumIntersections = PARAM_SIZE;
#endif  // PARAM_SIZE

static const float kCellLength = 0.005f;
static const float kProducerRatio = 0.02f;
static const float kTargetRatio = 0.003f;
static const int kNumIterations = 12000;
static const float kCarAllocationRatio = 0.02f;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 32*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

// Only for baseline version.
static const int kMaxNumCells = kNumObjects;
static const int kMaxNumCars = kNumObjects;

#endif  // EXAMPLE_TRAFFIC_CONFIGURATION_H
