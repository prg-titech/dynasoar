#ifndef EXAMPLE_NBODY_SOA_NBODY_H
#define EXAMPLE_NBODY_SOA_NBODY_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"


// Pre-declare all classes.
class Body;

using AllocatorT = SoaAllocator<kNumObjects, Body>;

class Body : public AllocatorT::Base {
 public:
  declare_field_types(
      Body,
      float,          // pos_x_
      float,          // pos_y_,
      float,          // vel_x_,
      float,          // vel_y_,
      float,          // force_x_,
      float,          // force_y_,
      float)          // mass_

 private:
  Field<Body, 0> pos_x_;
  Field<Body, 1> pos_y_;
  Field<Body, 2> vel_x_;
  Field<Body, 3> vel_y_;
  Field<Body, 4> force_x_;
  Field<Body, 5> force_y_;
  Field<Body, 6> mass_;

 public:
  __device__ Body(float pos_x, float pos_y, float vel_x, float vel_y,
                  float mass);

  __device__ void compute_force();

  __device__ void apply_force(Body* other);

  __device__ void update();

  void add_checksum();

  // Only for rendering.
  __device__ __host__ float pos_x() const { return pos_x_; }
  __device__ __host__ float pos_y() const { return pos_y_; }
  __device__ __host__ float mass() const { return mass_; }
};

#endif  // EXAMPLE_NBODY_SOA_NBODY_H
