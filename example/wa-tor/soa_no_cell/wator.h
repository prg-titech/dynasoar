#ifndef EXAMPLE_WA_TOR_SOA_WATOR_H
#define EXAMPLE_WA_TOR_SOA_WATOR_H

#include <curand_kernel.h>

#include "allocator_config.h"


// Pre-declare all classes.
class Agent;
class Fish;
class Shark;

using AllocatorT = SoaAllocator<64*64*64*64, Agent, Fish, Shark>;
using IndexT = int;


class Agent : public SoaBase<AllocatorT> {
 public:
  define_field_types(
      Agent,
      curandState_t,    // random_state_
      IndexT,           // position_
      IndexT)           // new_position_

  static const bool kIsAbstract = true;

 protected:
  SoaField<Agent, 0> random_state_;
  SoaField<Agent, 1> position_;
  SoaField<Agent, 2> new_position_;

 public:
  __device__ Agent(int seed);

  __device__ IndexT position() const;

  __device__ curandState_t& random_state();

  __device__ void set_new_position(IndexT new_pos);

  __device__ void set_position(IndexT cell);
};


class Fish : public Agent {
 public:
  define_field_types(
      Fish,
      uint32_t)        // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Fish, 0> egg_timer_;

 public:
  __device__ Fish(int seed);

  __device__ void prepare();

  __device__ void update();
};


class Shark : public Agent {
 public:
  define_field_types(
      Shark,
      uint32_t,        // energy_
      uint32_t)        // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Shark, 0> energy_;
  SoaField<Shark, 1> egg_timer_;

 public:
  __device__ Shark(int seed);

  __device__ void prepare();

  __device__ void update();
};

#endif  // EXAMPLE_WA_TOR_SOA_WATOR_H
