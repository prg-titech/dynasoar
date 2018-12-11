#ifndef EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H
#define EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "configuration.h"

// Pre-declare all classes.
class Agent;
class Male;
class Female;
class Cell;

using AllocatorT = SoaAllocator<64*64*64*64, Agent, Male, Female, Cell>;

class Cell : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      curandState_t,    // random_state_
      Agent*,           // agent_
      int,              // sugar_
      int,              // sugar_capacity_
      int,              // grow_rate_
      int>;             // cell_id_

 private:
  SoaField<Cell, 0> random_state_;
  SoaField<Cell, 1> agent_;
  SoaField<Cell, 2> sugar_;
  SoaField<Cell, 3> sugar_capacity_;
  SoaField<Cell, 4> grow_rate_;
  SoaField<Cell, 5> cell_id_;

 public:
  __device__ Cell(int seed, int sugar, int sugar_capacity, int grow_rate,
                  int cell_id);

  __device__ void decide_permission();

  __device__ bool is_free();

  __device__ void enter(Agent* agent);

  __device__ void leave();

  __device__ int sugar();

  __device__ void take_sugar(int amount);

  __device__ void grow_sugar();

  __device__ float random_float();

  __device__ int random_int(int a, int b);

  __device__ int cell_id();

  __device__ Agent* agent();

  // Only for rendering.
  __device__ void add_to_draw_array();
};


class Agent : public SoaBase<AllocatorT> {
 public:
  static const bool kIsAbstract = true;

  using FieldTypes = std::tuple<
      curandState_t,    // random_state_
      Cell*,            // cell_
      Cell*,            // cell_request_
      int,              // vision_
      int,              // age_
      int,              // max_age_
      int,              // sugar_
      int,              // metabolism_
      int,              // endowment_
      bool>;            // permission_

 protected:
  SoaField<Agent, 0> random_state_;
  SoaField<Agent, 1> cell_;
  SoaField<Agent, 2> cell_request_;
  SoaField<Agent, 3> vision_;
  SoaField<Agent, 4> age_;
  SoaField<Agent, 5> max_age_;
  SoaField<Agent, 6> sugar_;
  SoaField<Agent, 7> metabolism_;
  SoaField<Agent, 8> endowment_;
  SoaField<Agent, 9> permission_;

 public:
  __device__ Agent(Cell* cell, int vision, int age, int max_age, int endowment,
                   int metabolism);

  __device__ void prepare_move();

  __device__ void update_move();

  __device__ void give_permission();

  __device__ void age_and_metabolize();

  __device__ void harvest_sugar();

  __device__ bool ready_to_mate();

  __device__ Cell* cell_request();

  __device__ int sugar();

  __device__ int endowment();

  __device__ int vision();

  __device__ int max_age();

  __device__ int metabolism();

  __device__ void take_sugar(int amount);

  __device__ float random_float();
};


class Male : public Agent {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = Agent;
  using FieldTypes = std::tuple<
      Female*,    // female_request_
      bool>;      // proposal_accepted_

 private:
  SoaField<Male, 0> female_request_;
  SoaField<Male, 1> proposal_accepted_;

 public:
  __device__ Male(Cell* cell, int vision, int age, int max_age, int endowment,
                  int metabolism);

  __device__ Female* female_request();

  __device__ void accept_proposal();

  __device__ void propose();

  __device__ void propose_offspring_target();

  __device__ void mate();
};


class Female : public Agent {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = Agent;
  using FieldTypes = std::tuple<char>;  // dummy_

  __device__ Female(Cell* cell, int vision, int age, int max_age,
                    int endowment, int metabolism);

  __device__ void decide_proposal();
};


#endif  // EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H
