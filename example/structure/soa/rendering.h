#ifndef EXAMPLE_STRUCTURE_SOA_RENDERING_H
#define EXAMPLE_STRUCTURE_SOA_RENDERING_H

struct SpringInfo {
  float p1_x, p1_y, p2_x, p2_y;
  float force, max_force;
};


void init_renderer();
void draw(int num_springs, SpringInfo* springs);
void close_renderer();

#endif  // EXAMPLE_STRUCTURE_SOA_RENDERING_H
