#ifndef EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
#define EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H

// Runs about 90 seconds on my machine (Titan Xp).
// Note: SoaAlloc does not pay off with small problem sizes. Probably because
// overheads due to kernel launch and pre-iteration bitmap scans dominate.

static const int kNumComputeIterations = 40;
static const int kMaxDegree = 5;
static const float kDt = 0.01f;
static const int kNumSteps = 100;  // 7500
static const int kMaxNodes = 50000;  // 500000
static const int kMaxSprings = kMaxNodes*kMaxDegree / 2;
static const bool kOptionRender = false;
static const float kVelocityDampening = 0.0f; //0.00005; // Percentage value
static const bool kOptionPrintStats = true;
static const bool kOptionDefrag = true;

// TODO: Should be a constant.
#define kMaxDistance 32768

static const char kTypeNodeBase = 1;
static const char kTypeAnchorNode = 2;
static const char kTypeAnchorPullNode = 3;
static const char kTypeNode = 4;

#endif  // EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
