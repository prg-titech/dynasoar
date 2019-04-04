#ifndef EXAMPLE_WA_TOR_SOA_WATOR_H
#define EXAMPLE_WA_TOR_SOA_WATOR_H

#include <curand_kernel.h>

#include "allocator_config.h"


// Pre-declare all classes.
class Agent;
class Cell;
class Fish;
class Shark;

using AllocatorT = SoaAllocator<kNumObjects, Agent, Fish, Shark, Cell>;

class Cell : public AllocatorT::Base {
 public:
  // Sanity check for DeviceArray.
  static_assert(sizeof(DeviceArray<bool, 5>) == 5, "Size mismatch.");

  declare_field_types(
      Cell,
      curandState_t,                 // random_state_ (48 bytes)
      DeviceArray<Cell*, 4>,         // neighbors_
      Agent*,                        // agent_
      DeviceArray<bool, 5>)          // neighbor_request_

 private:
  Field<Cell, 0> random_state_;

  // left, top, right, bottom
  Field<Cell, 1> neighbors_;

  Field<Cell, 2> agent_;

  // left, top, right, bottom, self
  Field<Cell, 3> neighbor_request_;

 public:
  __device__ Cell(int cell_id);

  __device__ Agent* agent() const;

  __device__ void decide();

  __device__ void enter(Agent* agent);

  __device__ bool has_fish() const;

  __device__ bool has_shark() const;

  __device__ bool is_free() const;

  __device__ void kill();

  __device__ void leave() ;

  __device__ void prepare();

  __device__ curandState_t& random_state();

  __device__ void set_neighbors(Cell* left, Cell* top,
                                Cell* right, Cell* bottom);

  __device__ void request_random_fish_neighbor();

  __device__ void request_random_free_neighbor();

  template<bool(Cell::*predicate)() const>
  __device__ bool request_random_neighbor(curandState_t& random_state);

  __device__ void add_to_checksum();
};


class Agent : public SoaBase<AllocatorT> {
 public:
  declare_field_types(
      Agent,
      curandState_t,    // random_state_
      Cell*,            // position_
      Cell*)            // new_position_

  static const bool kIsAbstract = true;

 protected:
  Field<Agent, 0> random_state_;
  Field<Agent, 1> position_;
  Field<Agent, 2> new_position_;

 public:
  __device__ Agent(int seed);

  __device__ Cell* position() const;

  __device__ curandState_t& random_state();

  __device__ void set_new_position(Cell* new_pos);

  __device__ void set_position(Cell* cell);
};


class Fish : public Agent {
 public:
  declare_field_types(
      Fish,
      uint32_t)        // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  Field<Fish, 0> egg_timer_;

 public:
  __device__ Fish(int seed);

  __device__ void prepare();

  __device__ void update();
};


class Shark : public Agent {
 public:
  declare_field_types(
      Shark,
      uint32_t,        // energy_
      uint32_t)        // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  Field<Shark, 0> energy_;
  Field<Shark, 1> egg_timer_;

 public:
  __device__ Shark(int seed);

  __device__ void prepare();

  __device__ void update();
};

#endif  // EXAMPLE_WA_TOR_SOA_WATOR_H
