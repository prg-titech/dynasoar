#ifndef EXAMPLE_NBODY_SOA_RENDERING_H
#define EXAMPLE_NBODY_SOA_RENDERING_H

void init_renderer();
void close_renderer();
void init_frame();
void show_frame();
void draw_body(float pos_x, float pos_y, float mass);

#endif  // EXAMPLE_NBODY_SOA_RENDERING_H
