#ifndef EXAMPLE_GAME_OF_LIFE_SOA_GOL_H
#define EXAMPLE_GAME_OF_LIFE_SOA_GOL_H

#include "allocator_config.h"

// Pre-declare all classes.
class Cell;
class Agent;
class Alive;
class Candidate;

using AllocatorT = SoaAllocator<8*64*64*64*64, Cell, Agent, Alive, Candidate>;


static const int kActionNone = 0;
static const int kActionDie = 1;
static const int kActionSpawnAlive = 2;


class Cell : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<Agent*>;  // agent_

  SoaField<Cell, 0> agent_;

  __device__ Cell();

  __device__ Agent* agent();

  __device__ bool is_empty();
};


class Agent : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      int,  // cell_id_
      char>;  // action_

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
  using FieldTypes = std::tuple<bool>;  // is_new_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Alive, 0> is_new_;

 public:
  __device__ Alive(int cell_id);

  __device__ void create_candidates();

  __device__ void maybe_create_candidate(int x, int y);

  __device__ void prepare();

  __device__ void update();

  // Only for rendering.
  __device__ void update_render_array();

  // Only for checksum computation.
  __device__ void update_checksum();
};


class Candidate : public Agent {
 public:
  // TODO: This should be empty but it cannot be at the moment.
  using FieldTypes = std::tuple<>;  // no additional fields

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

  __device__ Candidate(int cell_id);

  __device__ void prepare();

  __device__ void update();

  // Only for debugging.
  __device__ void update_counter();
};


#endif  // EXAMPLE_GAME_OF_LIFE_SOA_GOL_H
