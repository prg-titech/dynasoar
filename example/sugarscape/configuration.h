#ifndef EXAMPLE_SUGARSCAPE_CONFIGURATION_H
#define EXAMPLE_SUGARSCAPE_CONFIGURATION_H

// Size of simulation.
#ifndef PARAM_SIZE
static const int kSize = 1700;
#else
static const int kSize = PARAM_SIZE;
#endif  // PARAM_SIZE

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

static const int kSeed = 42;

// For initialization only.
//static const float kProbMale = 0.0;
//static const float kProbFemale = 0.0;

static const float kProbMale = 0.12;
static const float kProbFemale = 0.15;

// Simulation constants.
static const int kNumIterations = 10000;
static const int kMaxVision = 2;
static const int kMaxAge = 100;
static const int kMaxEndowment = 200;
static const int kMaxMetabolism = 80;
static const int kSugarCapacity = 3500;
static const int kMaxSugarDiffusion = 60;
static const float kSugarDiffusionRate = 0.125;
static const int kMinMatingAge = 22;
static const int kMaxChildren = 8;

// Helper data structure for rendering and checksum computation.
struct CellInfo {
  int sugar;
  char agent_type;
};


#endif  // EXAMPLE_SUGARSCAPE_CONFIGURATION_H
