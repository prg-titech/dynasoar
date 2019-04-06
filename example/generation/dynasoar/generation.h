#ifndef EXAMPLE_GENERATION_SOA_GENERATION_H
#define EXAMPLE_GENERATION_SOA_GENERATION_H

#include "allocator_config.h"
#include "../configuration.h"

// Pre-declare all classes.
class Cell;
class Agent;
class Alive;
class Candidate;

using AllocatorT = SoaAllocator<kNumObjects, Cell, Agent, Alive, Candidate>;


static const char kActionNone = 0;
static const char kActionDie = 1;
static const char kActionSpawnAlive = 2;


class Cell : public AllocatorT::Base {
 public:
  declare_field_types(
      Cell,
      Agent*,   // agent_
      int)      // reserved_

  SoaField<Cell, 0> agent_;

  SoaField<Cell, 1> reserved_;

  __device__ Cell();

  __device__ Agent* agent();

  __device__ bool is_empty();
};


class Agent : public AllocatorT::Base {
 public:
  declare_field_types(
      Agent,
      int,   // cell_id_
      int)   // action_

  static const bool kIsAbstract = true;

 protected:
  SoaField<Agent, 0> cell_id_;
  SoaField<Agent, 1> action_;

 public:
  __device__ Agent(int cell_id);

  __device__ int num_alive_neighbors();

  __device__ int cell_id();
};


class Alive : public Agent {
 public:
  declare_field_types(
      Alive,
      int,      // state_
      bool)     // is_new_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Alive, 0> state_;
  SoaField<Alive, 1> is_new_;

 public:
  __device__ Alive(int cell_id);

  __device__ void create_candidates();

  __device__ void prepare();

  __device__ void update();

  __device__ int state() const { return state_; }

#ifdef OPTION_RENDER
  // Only for rendering.
  __device__ void update_render_array();
#endif  // OPTION_RENDER

  // Only for checksum computation.
  __device__ void update_checksum();
};


class Candidate : public Agent {
 public:
  declare_field_types(Candidate)  // no additional fields

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

  __device__ Candidate(int cell_id);

  __device__ void prepare();

  __device__ void update();

#ifdef OPTION_RENDER
  // Only for rendering.
  __device__ void update_render_array();
#endif  // OPTION_RENDER

  // Only for debugging.
  __device__ void update_counter();
};


#endif  // EXAMPLE_GENERATION_SOA_GENERATION_H
