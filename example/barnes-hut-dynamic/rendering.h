#ifndef EXAMPLE_BARNES_HUT_RENDERING_H
#define EXAMPLE_BARNES_HUT_RENDERING_H

#include "configuration.h"

void init_renderer();
void close_renderer();

void init_frame();
void show_frame();

void draw_tree_node(float x1, float y1, float x2, float y2);
void draw_body(float x, float y, float mass, float max_mass);

#endif  // EXAMPLE_BARNES_HUT_RENDERING_H

