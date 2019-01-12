#ifndef EXAMPLE_WA_TOR_SOA_CONFIGURATION_H
#define EXAMPLE_WA_TOR_SOA_CONFIGURATION_H

#include "extra_config.h"

// Size of simulation.
static const int kSeed = 42;
static const int kSizeX = 2048;

// Allow 256MB heap size.
// Simulation constants.
static const int kSpawnThreshold = 4;
static const int kEngeryBoost = 4;
static const int kEngeryStart = 2;
static const bool kOptionSharkDie = true;
static const bool kOptionFishSpawn = true;
static const bool kOptionSharkSpawn = true;
static const bool kOptionDefrag = false;
static const bool kOptionPrintStats = false;
static const int kNumIterations = 500;

#endif  // EXAMPLE_WA_TOR_SOA_CONFIGURATION_H
