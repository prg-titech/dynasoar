#ifndef EXAMPLE_STRUCTURE_SOA_DATASET_H
#define EXAMPLE_STRUCTURE_SOA_DATASET_H

#include <vector>
#include <stdlib.h>

#include "configuration.h"
#include "rendering.h"

struct DsNode {
  DsNode(char p_type, float p_pos_x, float p_pos_y, float p_vel_x,
         float p_vel_y, float p_mass)
      : pos_x(p_pos_x), pos_y(p_pos_y), vel_x(p_vel_x), vel_y(p_vel_y),
        mass(p_mass), type(p_type), num_springs(0) {}

  float pos_x, pos_y, vel_x, vel_y, mass;
  char type;
  int num_springs;
};

struct DsSpring {
  DsSpring(int p_p1, int p_p2, float p_spring_factor, float p_max_force)
      : p1(p_p1), p2(p_p2), spring_factor(p_spring_factor),
        max_force(p_max_force) {}

  int p1, p2;
  float spring_factor, max_force;
};

struct Dataset {
  std::vector<DsNode> nodes;
  std::vector<DsSpring> springs;
};

void random_dataset(Dataset& result);

#endif  // EXAMPLE_STRUCTURE_SOA_DATASET_H
