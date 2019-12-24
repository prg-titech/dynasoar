#ifndef EXAMPLE_COLLISION_SOA_RENDERING_H
#define EXAMPLE_COLLISION_SOA_RENDERING_H

#include "configuration.h"

void init_renderer();
void close_renderer();

void init_frame();
void show_frame();
void draw_body(float pos_x, float pos_y, float mass, float max_mass);
void maybe_draw_line(float pos_x, float pos_x2, float pos_y, float pos_y2);

#endif  // EXAMPLE_COLLISION_SOA_RENDERING_H

