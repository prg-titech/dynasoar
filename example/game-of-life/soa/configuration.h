#ifndef EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
#define EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H

#include "example/game-of-life/soa/dataset_loader.h"

#define OPTION_DRAW false

using CellT = int;

extern dataset_t dataset;

static const int kNumIterations = 100;

#endif  // EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
