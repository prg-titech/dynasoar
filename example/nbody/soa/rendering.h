#ifndef RENDERING_H
#define RENDERING_H

namespace nbody {

void init_renderer();
void close_renderer();
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass);

}  // namespace nbody

#endif  // RENDERING_H
