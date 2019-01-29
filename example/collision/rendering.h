#ifndef EXAMPLE_COLLISION_SOA_RENDERING_H
#define EXAMPLE_COLLISION_SOA_RENDERING_H

#include "configuration.h"

void init_renderer();
void close_renderer();
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass, int num_bodies = kNumBodies);

#endif  // EXAMPLE_COLLISION_SOA_RENDERING_H

