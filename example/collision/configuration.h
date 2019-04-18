#ifndef EXAMPLE_COLLISION_CONFIGURATION_H
#define EXAMPLE_COLLISION_CONFIGURATION_H

// Simulation parameters.
static const float kMergeThreshold = 0.005;

#ifdef PARAM_SIZE
static const int kNumBodies = PARAM_SIZE;
#else
static const int kNumBodies = 150000;
#endif  // PARAM_SIZE

static const float kMaxMass = 75;
static const int kSeed = 42;
static const float kTimeInterval = 0.01;
static const int kIterations = 10000;
static const float kDampeningFactor = 0.05f;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_COLLISION_CONFIGURATION_H
