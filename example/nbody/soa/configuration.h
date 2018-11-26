#ifndef EXAMPLE_NBODY_SOA_CONFIGURATION_H
#define EXAMPLE_NBODY_SOA_CONFIGURATION_H

namespace nbody {

static const int kSeed = 42;
static const int kScalingFactor = 100;
static const float kMaxMass = 1000.0f;
static const int kNumIterations = 500;
static const int kNumBodies = 36;
static const float kDt = 0.5f;
static const float kGravityConstant = 6.673e-11;  // gravitational constant

}  // namespace nbody

#endif  // EXAMPLE_NBODY_SOA_CONFIGURATION_H
