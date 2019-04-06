#ifndef EXAMPLE_BARNES_HUT_RENDERING_H
#define EXAMPLE_BARNES_HUT_RENDERING_H

#include "configuration.h"

void init_renderer();
void close_renderer();
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass, int num_bodies,
          float* host_Tree_p1_x, float* host_Tree_p1_y,
          float* host_Tree_p2_x, float* host_Tree_p2_y, int Tree_num_nodes);

#endif  // EXAMPLE_BARNES_HUT_RENDERING_H

