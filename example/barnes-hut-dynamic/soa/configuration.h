#ifndef EXAMPLE_BARNES_HUT_DYNAMIC_SOA_CONFIGURATION_H
#define EXAMPLE_BARNES_HUT_DYNAMIC_SOA_CONFIGURATION_H

static const float kDistThreshold = 0.5f;
static const float kDampeningFactor = 0.05f;
static const int kIterations = 2000;
static const float kDt = 0.02f;
static const float kGravityConstant = 6.673e-11;   // gravitational constant
static const int kNumBodies = 10000;
static const float kMaxMass = 1000.0f;
static const int kSeed = 42;

#endif  // EXAMPLE_BARNES_HUT_DYNAMIC_SOA_CONFIGURATION_H
