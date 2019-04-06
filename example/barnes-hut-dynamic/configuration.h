#ifndef EXAMPLE_BARNES_HUT_DYNAMIC_CONFIGURATION_H
#define EXAMPLE_BARNES_HUT_DYNAMIC_CONFIGURATION_H

static const float kDistThreshold = 0.25f;
static const float kDampeningFactor = 0.05f;
static const int kIterations = 4000;
static const float kDt = 0.0075f;
static const float kGravityConstant = 6.673e-11;   // gravitational constant

#ifdef PARAM_SIZE
static const int kNumBodies = PARAM_SIZE;
#else
static const int kNumBodies = 300000;
#endif  // PARAM_SIZE


static const float kMaxMass = 85.0f;
static const int kSeed = 42;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_BARNES_HUT_DYNAMIC_CONFIGURATION_H
