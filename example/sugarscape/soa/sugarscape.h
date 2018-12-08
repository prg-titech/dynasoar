#ifndef EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H
#define EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H

#include <curand_kernel.h>

#include "example/configuration/soa_alloc.h"
//#include "example/configuration/cuda_allocator.h"

static const int kMaxVision = 10;

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

 private:
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

  __device__ void give_permission();
};


class Male : public Agent {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = Agent;
  using FieldTypes = std::tuple<char>;  // dummy_

  __device__ Male(Cell* cell, int vision, int age, int max_age, int endowment,
                  int metabolism);
};


class Female : public Agent {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = Agent;
  using FieldTypes = std::tuple<char>;  // dummy_

  __device__ Female(Cell* cell, int vision, int age, int max_age,
                    int endowment, int metabolism);
};


#endif  // EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_H
