#ifndef EXAMPLE_COLLISION_SOA_CONFIGURATION_H
#define EXAMPLE_COLLISION_SOA_CONFIGURATION_H

// Simulation parameters.
static const bool kOptionRender = false;
static const float kMergeThreshold = 0.001;
static const int kNumBodies = 25000;
static const float kMaxMass = 75;
static const int kSeed = 42;
static const float kTimeInterval = 0.15;
static const int kIterations = 4000;
static const float kDampeningFactor = 0.05f;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

#endif  // EXAMPLE_COLLISION_SOA_CONFIGURATION_H
