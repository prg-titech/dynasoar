#ifndef EXAMPLE_NBODY_SOA_NBODY_H
#define EXAMPLE_NBODY_SOA_NBODY_H

#include <curand_kernel.h>

#include "allocator_config.h"

namespace nbody {

// Pre-declare all classes.
class Body;

using AllocatorT = SoaAllocator<64*64*64*64, Body>;

class Body : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      float,          // pos_x_
      float,          // pos_y_,
      float,          // vel_x_,
      float,          // vel_y_,
      float,          // force_x_,
      float,          // force_y_,
      float>;         // mass_

 private:
  SoaField<Body, 0> pos_x_;
  SoaField<Body, 1> pos_y_;
  SoaField<Body, 2> vel_x_;
  SoaField<Body, 3> vel_y_;
  SoaField<Body, 4> force_x_;
  SoaField<Body, 5> force_y_;
  SoaField<Body, 6> mass_;

 public:
  __DEV__ Body(float pos_x, float pos_y, float vel_x, float vel_y, float mass);

  __DEV__ void compute_force();

  __DEV__ void apply_force(Body* other);

  __DEV__ void update();

  __DEV__ void add_checksum();

  __DEV__ void add_to_draw_array();
};

}  // nbody

#endif  // EXAMPLE_NBODY_SOA_NBODY_H
