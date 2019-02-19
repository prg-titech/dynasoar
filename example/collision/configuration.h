#ifndef EXAMPLE_COLLISION_SOA_CONFIGURATION_H
#define EXAMPLE_COLLISION_SOA_CONFIGURATION_H

// Simulation parameters.
static const bool kOptionRender = false;
static const float kMergeThreshold = 0.0005;
static const int kNumBodies = 150000;
static const float kMaxMass = 75;
static const int kSeed = 42;
static const float kTimeInterval = 0.01;
static const int kIterations = 10000;
static const float kDampeningFactor = 0.05f;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

static const bool kOptionPrintStats = false;

#endif  // EXAMPLE_COLLISION_SOA_CONFIGURATION_H
