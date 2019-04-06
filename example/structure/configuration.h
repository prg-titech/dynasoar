#ifndef EXAMPLE_STRUCTURE_CONFIGURATION_H
#define EXAMPLE_STRUCTURE_CONFIGURATION_H

// Runs about 90 seconds on my machine (Titan Xp).
// Note: DynaSOAr does not pay off with small problem sizes. Probably because
// overheads due to kernel launch and pre-iteration bitmap scans dominate.

static const int kNumComputeIterations = 40;
static const int kMaxDegree = 5;
static const float kDt = 0.2f;
static const int kNumSteps = 500;

#ifdef PARAM_SIZE
static const int kMaxNodes = PARAM_SIZE;
#else
static const int kMaxNodes = 1500000;
#endif  // PARAM_SIZE_X

static const int kMaxSprings = kMaxNodes*kMaxDegree / 2;
static const float kVelocityDampening = 0.0f; //0.00005; // Percentage value

// TODO: Should be a constant.
#define kMaxDistance 32768

static const char kTypeNodeBase = 1;
static const char kTypeAnchorNode = 2;
static const char kTypeAnchorPullNode = 3;
static const char kTypeNode = 4;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 2*64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ


// Helper data structure for loading and rendering.
struct SpringInfo {
  float p1_x, p1_y, p2_x, p2_y;
  float force, max_force;
};


#endif  // EXAMPLE_STRUCTURE_CONFIGURATION_H
