#ifndef EXAMPLE_NBODY_SOA_CONFIGURATION_H
#define EXAMPLE_NBODY_SOA_CONFIGURATION_H

static const int kSeed = 42;
static const float kMaxMass = 1000.0f;
static const int kNumIterations = 3000;

#ifdef PARAM_SIZE
static const int kNumBodies = PARAM_SIZE;
#else
static const int kNumBodies = 20000;
#endif  // PARAM_SIZE

static const float kDt = 0.02f;
static const float kGravityConstant = 6.673e-11;  // gravitational constant
static const float kDampeningFactor = 0.05f;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_NBODY_SOA_CONFIGURATION_H
