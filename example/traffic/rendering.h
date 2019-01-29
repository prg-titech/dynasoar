#ifndef EXAMPLE_TRAFFIC_SOA_RENDERING_H
#define EXAMPLE_TRAFFIC_SOA_RENDERING_H

#include "configuration.h"

void init_renderer();
void close_renderer();
void draw(float* host_Cell_pos_x, float* host_Cell_pos_y,
          bool* host_Cell_occupied, int num_cells);

#endif  // EXAMPLE_TRAFFIC_SOA_RENDERING_H

