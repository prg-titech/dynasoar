#ifndef EXAMPLE_WA_TOR_SOA_CONFIGURATION_H
#define EXAMPLE_WA_TOR_SOA_CONFIGURATION_H

// Size of simulation.
static const int kSeed = 42;

#ifdef PARAM_SIZE_X
static const int kSizeX = PARAM_SIZE_X;
#else
static const int kSizeX = 2048;
#endif  // PARAM_SIZE_X

#ifdef PARAM_SIZE_Y
static const int kSizeY = PARAM_SIZE_Y;
#else
static const int kSizeY = 2560;
#endif  // PARAM_SIZE_Y

// Simulation constants.
static const int kSpawnThreshold = 4;
static const int kEngeryBoost = 4;
static const int kEngeryStart = 2;
static const bool kOptionSharkDie = true;
static const bool kOptionFishSpawn = true;
static const bool kOptionSharkSpawn = true;
static const int kNumIterations = 500;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_WA_TOR_SOA_CONFIGURATION_H
