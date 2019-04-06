#ifndef EXAMPLE_NBODY_SOA_RENDERING_H
#define EXAMPLE_NBODY_SOA_RENDERING_H

void init_renderer();
void close_renderer();
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass);

#endif  // EXAMPLE_NBODY_SOA_RENDERING_H
