#ifndef EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H
#define EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H

// Size of simulation.
static const int kSize = 1500;
static const int kSeed = 42;

// For initialization only.
//static const float kProbMale = 0.0;
//static const float kProbFemale = 0.0;

static const float kProbMale = 0.12;
static const float kProbFemale = 0.15;

// Simulation constants.
static const int kNumIterations = 1000;
static const int kMaxVision = 2;
static const int kMaxAge = 100;
static const int kMaxEndowment = 200;
static const int kMaxMetabolism = 80;
static const int kSugarCapacity = 3500;
static const int kMaxSugarDiffusion = 60;
static const float kSugarDiffusionRate = 0.125;
static const int kMinMatingAge = 22;

// Debug/rendering
static const bool kOptionRender = false;

#endif  // EXAMPLE_SUGARSCAPE_SOA_CONFIGURATION_H
