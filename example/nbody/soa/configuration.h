#ifndef EXAMPLE_NBODY_SOA_CONFIGURATION_H
#define EXAMPLE_NBODY_SOA_CONFIGURATION_H

namespace nbody {

#define OPTION_DRAW false
static const int kSeed = 42;
static const float kMaxMass = 1000.0f;
static const int kNumIterations = 500;
static const int kNumBodies = 16000;
static const float kDt = 0.5f;
static const float kGravityConstant = 6.673e-11;  // gravitational constant
static const float kDampeningFactor = 0.2f;

}  // namespace nbody

#endif  // EXAMPLE_NBODY_SOA_CONFIGURATION_H
