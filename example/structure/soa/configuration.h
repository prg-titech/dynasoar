#ifndef EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
#define EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H

static const int kNumComputeIterations = 60;
static const int kMaxDegree = 5;
static const float kDt = 0.01f;
static const int kNumSteps = 100;
static const int kMaxNodes = 100;
static const int kMaxSprings = kMaxNodes*kMaxDegree;
static const bool kOptionRender = true;
static const float kVelocityDampening = 0.0f; //0.00005; // Percentage value

#endif  // EXAMPLE_STRUCTURE_SOA_CONFIGURATION_H
