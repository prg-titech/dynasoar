#ifndef EXAMPLE_NBODY_SOA_CONFIGURATION_H
#define EXAMPLE_NBODY_SOA_CONFIGURATION_H

namespace nbody {

static const int kSeed = 42;
static const float kMaxMass = 1000.0f;
static const int kNumIterations = 3000;
static const int kNumBodies = 20000;
static const float kDt = 0.02f;
static const float kGravityConstant = 6.673e-11;  // gravitational constant
static const float kDampeningFactor = 0.05f;
static const bool kOptionRender = false;

}  // namespace nbody

#endif  // EXAMPLE_NBODY_SOA_CONFIGURATION_H
