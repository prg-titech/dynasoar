#ifndef EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
#define EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H

#include "example/game-of-life/soa/dataset_loader.h"

using CellT = int;

extern dataset_t dataset;

static const int kNumIterations = 300;  // 10000
static const bool kOptionRender = false;

#endif  // EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
