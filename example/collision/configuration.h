#ifndef EXAMPLE_COLLISION_CONFIGURATION_H
#define EXAMPLE_COLLISION_CONFIGURATION_H

// Simulation parameters.
static const float kMergeThreshold = 0.005;

#ifdef PARAM_SIZE
static const int kNumBodies = PARAM_SIZE;
#else
static const int kNumBodies = 40000;
#endif  // PARAM_SIZE

static const float kMaxMass = 75;
static const int kSeed = 42;

#ifdef PARAM_DELTA_T
static const float kTimeInterval = PARAM_DELTA_T;
#else
static const float kTimeInterval = 0.05f;
#endif  // PARAM_DELTA_T

#ifdef PARAM_NUM_ITER
static const int kIterations = PARAM_NUM_ITER;
#else
static const int kIterations = 2000;
#endif  // PARAM_NUM_ITER

static const float kDampeningFactor = 0.05f;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_COLLISION_CONFIGURATION_H
