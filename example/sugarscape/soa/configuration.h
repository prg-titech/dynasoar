#ifndef EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H
#define EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H

// Size of simulation.
static const int kSizeX = 400;
static const int kSizeY = 400;
static const int kSeed = 42;

// For initialization only.
static const float kProbMale = 0.12;
static const float kProbFemale = 0.15;

// Simulation constants.
static const int kNumIterations = 100;
static const int kMaxVision = 10;
static const int kMaxAge = 100;
static const int kMaxEndowment = 200;
static const int kMaxMetabolism = 20;
static const int kSugarCapacity = 250;

// Debug/rendering
static const bool kOptionRender = true;

#endif  // EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H
