#ifndef EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
#define EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H

#include "dataset_loader.h"

using CellT = int;

extern dataset_t dataset;

static const int kNumIterations = 300;  // 10000
static const bool kOptionRender = false;
static const bool kOptionPrintStats = false;

#endif  // EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
