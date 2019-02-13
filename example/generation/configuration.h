#ifndef EXAMPLE_GENERATION_CONFIGURATION_H
#define EXAMPLE_GENERATION_CONFIGURATION_H

#include "dataset_loader.h"

extern dataset_t dataset;

// TODO: Should be constants instead of macros.
//#define kNumStates 50
//#define kStayAlive { false, false, true, false, false, false, false, false }
//#define kSpawnNew  { false, true, false, true, false, false, false, false }

#define kNumStates 30
#define kStayAlive { true, false, true, true, false, true, true, true, true }
#define kSpawnNew  { false, false, false, true, true, false, true, false, true }


static const int kNumIterations = 1000;  // 10000
static const bool kOptionRender = false;
static const bool kOptionPrintStats = true;
static const bool kOptionDefrag = true;

static const int kSize = 25000;

#endif  // EXAMPLE_GENERATION_CONFIGURATION_H

