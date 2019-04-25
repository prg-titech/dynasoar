#ifndef EXAMPLE_NBODY_CONFIGURATION_H
#define EXAMPLE_NBODY_CONFIGURATION_H

static const int kSeed = 42;
static const float kMaxMass = 1000.0f;

#ifdef PARAM_NUM_ITER
static const int kNumIterations = PARAM_NUM_ITER;
#else
static const int kNumIterations = 3000;
#endif  // PARAM_NUM_ITER

#ifdef PARAM_SIZE
static const int kNumBodies = PARAM_SIZE;
#else
static const int kNumBodies = 20000;
#endif  // PARAM_SIZE

#ifdef PARAM_DELTA_T
static const float kDt = PARAM_DELTA_T;
#else
static const float kDt = 0.02f;
#endif  // PARAM_DELTA_T

static const float kGravityConstant = 6.673e-11;  // gravitational constant
static const float kDampeningFactor = 0.05f;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_NBODY_CONFIGURATION_H
