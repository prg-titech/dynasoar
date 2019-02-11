#include "dataset.h"

float random_float(float a, float b) {
  return a + static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(b-a));
}

void random_dataset(Dataset& result) {
  srand(42);

  float spring_min = 3.0f;
  float spring_max = 5.0f;
  float max_force = 0.5f;
  float mass_min = 500.0f;
  float mass_max = 500.0f;

  int num_nodes = 0.65*kMaxNodes;
  int num_pull_nodes = 0.25*kMaxNodes;
  int num_anchor_nodes = 0.1*kMaxNodes;
  int num_springs = 0.7*kMaxSprings;
  int num_total_nodes = num_nodes + num_pull_nodes + num_anchor_nodes;

  float border_margin = 0.35f;

  int i = 0;
  for (; i < num_nodes; ++i) {
    float mass = random_float(mass_min, mass_max);
    float pos_x = random_float(border_margin, 1.0 - border_margin);
    float pos_y = random_float(border_margin, 1.0 - border_margin);
    result.nodes.push_back(DsNode(kTypeNode, pos_x, pos_y, 0.0f, 0.0f, mass));
  }

  for (; i < num_nodes + num_pull_nodes; ++i) {
    float pos_x = random_float(border_margin, 1.0 - border_margin);
    float pos_y = random_float(border_margin, 1.0 - border_margin);
    float vel_x = random_float(-0.05, 0.05);
    float vel_y = random_float(-0.05, 0.05);
    result.nodes.push_back(DsNode(kTypeAnchorPullNode,
                                  pos_x, pos_y, vel_x, vel_y, 0.0f));
  }

  for (; i < num_nodes + num_pull_nodes + num_anchor_nodes; ++i) {
    float pos_x = random_float(border_margin, 1.0 - border_margin);
    float pos_y = random_float(border_margin, 1.0 - border_margin);
    result.nodes.push_back(DsNode(kTypeAnchorNode,
                                  pos_x, pos_y, 0.0f, 0.0f, 0.0f));
  }

  for (i = 0; i < num_springs; ++i) {
    int p1 = -1;
    int p2 = -1;

    while (p1 == -1) {
      int n = rand() % num_total_nodes;

      if (result.nodes[n].num_springs < kMaxDegree) {
        p1 = n;
      }
    }

    while (p2 == -1) {
      int n = rand() % num_total_nodes;

      if (result.nodes[n].num_springs < kMaxDegree && n != p1) {
        p2 = n;
      }
    }

    float spring_factor = random_float(spring_min, spring_max);
    result.springs.push_back(DsSpring(p1, p2, spring_factor, max_force));
    ++result.nodes[p1].num_springs;
    ++result.nodes[p2].num_springs;
  };
}
