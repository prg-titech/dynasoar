#ifndef EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
#define EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H

static const int kNumComputeIterations = 60;
static const int kMaxDegree = 5;
static const float kDt = 0.01f;
static const int kNumSteps = 750;
static const int kMaxNodes = 1000;
static const int kMaxSprings = kMaxNodes*kMaxDegree; ///2;
static const bool kOptionRender = false;
static const float kVelocityDampening = 0.0f; //0.00005; // Percentage value

static const char kTypeNodeBase = 1;
static const char kTypeAnchorNode = 2;
static const char kTypeAnchorPullNode = 3;
static const char kTypeNode = 4;

#endif  // EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
