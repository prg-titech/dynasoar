#ifndef EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_NO_CELL_H
#define EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_NO_CELL_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"

// Pre-declare all classes.
class Agent;
class Male;
class Female;

using AllocatorT = SoaAllocator<kNumObjects, Agent, Male, Female>;


class Agent : public AllocatorT::Base {
 public:
  static const bool kIsAbstract = true;

  declare_field_types(
      Agent,
      curandState_t,    // random_state_
      int,              // cell_
      int,              // cell_request_
      int,              // vision_
      int,              // age_
      int,              // max_age_
      int,              // sugar_
      int,              // metabolism_
      int,              // endowment_
      bool)             // permission_

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
  __device__ Agent(int cell, int vision, int age, int max_age, int endowment,
                   int metabolism);

  __device__ void prepare_move();

  __device__ void update_move();

  __device__ void give_permission();

  __device__ void age_and_metabolize();

  __device__ void harvest_sugar();

  __device__ bool ready_to_mate();

  __device__ int cell_request();

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

  declare_field_types(
      Male,
      Female*,    // female_request_
      bool)       // proposal_accepted_

 private:
  SoaField<Male, 0> female_request_;
  SoaField<Male, 1> proposal_accepted_;

 public:
  __device__ Male(int cell, int vision, int age, int max_age, int endowment,
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

  declare_field_types(
      Female,
      int,            // max_children_
      int)            // num_children_

 private:
  SoaField<Female, 0> max_children_;
  SoaField<Female, 1> num_children_;

 public:
  __device__ Female(int cell, int vision, int age, int max_age,
                    int endowment, int metabolism, int max_children);

  __device__ void decide_proposal();

  __device__ void increment_num_children() { ++num_children_; }

  __device__ int max_children() {  return max_children_; }
};


#endif  // EXAMPLE_SUGARSCAPE_SOA_SUGARSCAPE_NO_CELL_H
