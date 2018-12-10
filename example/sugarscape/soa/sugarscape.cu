#include <chrono>

#include "example/sugarscape/soa/sugarscape.h"

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

static const int kSizeX = 400;
static const int kSizeY = 400;
static const int kSeed = 42;

// For initialization only.
static const float kProbMale = 0.12;
static const float kProbFemale = 0.15;

// Simulation constants.
static const int kMaxVision = 10;
static const int kMaxAge = 100;
static const int kMaxEndowment = 200;
static const int kMaxMetabolism = 20;

__device__ Cell* cells[kSizeX*kSizeY];


__device__ Cell::Cell(int seed, int sugar, int sugar_capacity, int grow_rate,
                      int cell_id)
    : agent_(nullptr), sugar_(sugar), sugar_capacity_(sugar_capacity),
      grow_rate_(grow_rate), cell_id_(cell_id) {
  curand_init(seed, cell_id, 0, &random_state_);
}


__device__ Agent::Agent(Cell* cell, int vision, int age, int max_age,
                        int endowment, int metabolism)
    : cell_(cell), cell_request_(nullptr), vision_(vision), age_(age),
      max_age_(max_age), sugar_(endowment), endowment_(endowment),
      metabolism_(metabolism), permission_(false) {
  curand_init(cell->random_int(0, kSizeX*kSizeY), 0, 0, &random_state_);
}


__device__ Male::Male(Cell* cell, int vision, int age, int max_age,
                      int endowment, int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism),
      proposal_accepted_(false), female_request_(nullptr) {}


__device__ Female::Female(Cell* cell, int vision, int age, int max_age,
                          int endowment, int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism) {}



__device__ void Agent::give_permission() { permission_ = true; }


__device__ void Agent::age_and_metabolize() {
  bool dead = false;

  age_ = age_ + 1;
  dead = age_ > max_age_;

  sugar_ -= metabolism_;
  dead = dead || sugar_ <= 0;

  if (dead) {
    cell_->leave();
    device_allocator->free<Agent>(this);
  }
}


__device__ void Agent::prepare_move() {
  // Move to cell with the most sugar.
  Cell* target_cell = nullptr;
  int target_sugar = 0;

  int this_x = cell_->cell_id() % kSizeX;
  int this_y = cell_->cell_id() / kSizeY;

  for (int dx = -vision_; dx < vision_ + 1; ++dx) {
    for (int dy = -vision_; dy < vision_ + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
        int n_id = nx + ny*kSizeX;
        Cell* n_cell = cells[n_id];

        if (n_cell->is_free()) {
          if (n_cell->sugar() > target_sugar) {
            target_cell = n_cell;
            target_sugar = n_cell->sugar();
          }
        }
      }
    }
  }

  cell_request_ = target_cell;
}


__device__ void Agent::update_move() {
  if (permission_ == true) {
    // Have permission to enter the cell.
    assert(cell_request_->is_free());
    cell_->leave();
    cell_request_->enter(this);
    cell_ = cell_request_;

    harvest_sugar();
  }

  cell_request_ = nullptr;
  permission_ = false;
}


__device__ void Agent::harvest_sugar() {
  // Harvest as much sugar as possible.
  int amount = cell_->sugar();
  cell_->take_sugar(amount);
  sugar_ += amount;
}


__device__ bool Agent::ready_to_mate() {
  // Half of endowment of sugar will go to the child. And the parent still
  // needs some sugar to survive.
  return (sugar_ >= endowment_ * 2 / 3) && age_ >= 18;
}


__device__ Cell* Agent::cell_request() { return cell_request_; }


__device__ int Agent::sugar() { return sugar_; }


__device__ int Agent::vision() { return vision_; }


__device__ int Agent::endowment() { return endowment_; }


__device__ int Agent::metabolism() { return metabolism_; }


__device__ void Agent::take_sugar(int amount) { sugar_ -= amount; }


__device__ float Agent::random_float() {
  return curand_uniform(&random_state_);
}


__device__ int Cell::cell_id() { return cell_id_; }


__device__ float Cell::random_float() {
  return curand_uniform(&random_state_);
}


__device__ int Cell::random_int(int a, int b) {
  return curand(&random_state_) % (b - a) + a;
}


__device__ void Cell::decide_permission() {
  Agent* selected_agent = nullptr;
  int turn = 0;
  int this_x = cell_id_ % kSizeX;
  int this_y = cell_id_ / kSizeY;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
        int n_id = nx + ny*kSizeX;
        Cell* n_cell = cells[n_id];
        Agent* n_agent = n_cell->agent_;

        if (n_agent->cell_request() == this) {
          ++turn;

          // Select cell with probability 1/turn.
          if (random_float() <= 1.0f/turn) {
            selected_agent = n_agent;
          }
        }
      }
    }
  }

  assert((turn == 0) == (selected_agent == nullptr));

  if (selected_agent != nullptr) {
    selected_agent->give_permission();
  }
}


__device__ bool Cell::is_free() { return agent_ == nullptr; }


__device__ void Cell::enter(Agent* agent) {
  assert(agent_ == nullptr);
  agent_ = agent;
}


__device__ void Cell::leave() {
  assert(agent_ != nullptr);
  agent_ = nullptr;
}


__device__ int Cell::sugar() { return sugar_; }


__device__ void Cell::take_sugar(int amount) { sugar_ -= amount; }


__device__ void Cell::grow_sugar() {
  sugar_ += min(sugar_capacity_ - sugar_, grow_rate_);
}


__device__ Agent* Cell::agent() { return agent_; }


__device__ void Male::propose() {
  if (ready_to_mate()) {
    // Propose to female with highest endowment.
    Female* target_agent = nullptr;
    int target_sugar = -1;

    int this_x = cell_->cell_id() % kSizeX;
    int this_y = cell_->cell_id() / kSizeY;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
          int n_id = nx + ny*kSizeX;
          Cell* n_cell = cells[n_id];
          Female* n_female = n_cell->agent()->cast<Female>();

          if (n_female != nullptr && n_female->ready_to_mate()) {
            if (n_female->sugar() > target_sugar) {
              target_agent = n_female;
              target_sugar = n_female->sugar();
            }
          }
        }
      }
    }

    assert((target_sugar == -1) == (target_agent == nullptr));
    female_request_ = target_agent;
  }
}


__device__ void Male::accept_proposal() {
  proposal_accepted_ = true;
}


__device__ void Male::propose_offspring_target() {
  if (proposal_accepted_) {
    // Select a random cell.
    Cell* target_cell = nullptr;
    int turn = 0;

    int this_x = cell_->cell_id() % kSizeX;
    int this_y = cell_->cell_id() / kSizeY;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
          int n_id = nx + ny*kSizeX;
          Cell* n_cell = cells[n_id];

          if (n_cell->is_free()) {
            ++turn;

            // Select cell with probability 1/turn.
            if (random_float() <= 1.0f/turn) {
              target_cell = n_cell;
            }
          }
        }
      }
    }

    cell_request_ = target_cell;
  }
}


__device__ void Male::mate() {
  if (proposal_accepted_ && permission_) {
    // Take sugar from endowment.
    int c_endowment = (endowment_ + female_request_->endowment()) / 2;
    sugar_ -= endowment_ / 2;
    female_request_->take_sugar(female_request_->endowment() / 2);

    // Calculate other properties.
    int c_vision = (vision_ + female_request_->vision()) / 2;
    int c_max_age = (max_age_ + female_request_->max_age()) / 2;
    int c_metabolism = (metabolism_ + female_request_->metabolism()) / 2;

    // Create agent.
    Agent* child;
    if (random_float() <= 0.5f) {
      child = device_allocator->make_new<Male>(
          cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else {
      child = device_allocator->make_new<Female>(
          cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    }

    // Add agent to target cell.
    cell_request_->enter(child);
  }

  proposal_accepted_ = false;
  female_request_ = nullptr;
  cell_request_ = nullptr;
}


__device__ void Female::decide_proposal() {
  Male* selected_agent = nullptr;
  int selected_sugar = -1;
  int this_x = cell_->cell_id() % kSizeX;
  int this_y = cell_->cell_id() / kSizeY;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
        int n_id = nx + ny*kSizeX;
        Cell* n_cell = cells[n_id];
        Male* n_male = n_cell->agent()->cast<Male>();

        if (n_male != nullptr) {
          if (n_male->sugar() > selected_sugar) {
            selected_agent = n_male;
            selected_sugar = n_male->sugar();
          }
        }
      }
    }
  }

  assert((selected_sugar == -1) == (selected_agent == nullptr));

  if (selected_agent != nullptr) {
    selected_agent->accept_proposal();
  }
}


void step() {
  allocator_handle->parallel_do<Cell, &Cell::grow_sugar>();

  allocator_handle->parallel_do<Agent, &Agent::age_and_metabolize>();
  allocator_handle->parallel_do<Agent, &Agent::prepare_move>();
  allocator_handle->parallel_do<Cell, &Cell::decide_permission>();
  allocator_handle->parallel_do<Agent, &Agent::update_move>();

  allocator_handle->parallel_do<Male, &Male::propose>();
  allocator_handle->parallel_do<Female, &Female::decide_proposal>();
  allocator_handle->parallel_do<Male, &Male::propose_offspring_target>();
  allocator_handle->parallel_do<Cell, &Cell::decide_permission>();
  allocator_handle->parallel_do<Male, &Male::mate>();
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    cells[i] = device_allocator->make_new<Cell>(
        kSeed, /*sugar=*/ 0, /*sugar_capacity=*/ 250, /*grow_rate=*/ 5,
        /*cell_id=*/ i);
  }
}


__global__ void create_agents() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    float r = cells[i]->random_float();
    int c_vision = kMaxVision/2 + cells[i]->random_int(0, kMaxVision/2);
    int c_max_age = kMaxAge*2/3 + cells[i]->random_int(0, kMaxAge/3);
    int c_endowment = kMaxEndowment/4
                      + cells[i]->random_int(0, kMaxEndowment*3/4);
    int c_metabolism = kMaxMetabolism/2
                       + cells[i]->random_int(0, kMaxMetabolism/2);
    Agent* agent = nullptr;

    if (r < kProbMale) {
      // Create male agent.
      agent = device_allocator->make_new<Male>(
          cells[i], c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else if (r < kProbMale + kProbFemale) {
      // Create female agent.
      agent = device_allocator->make_new<Female>(
          cells[i], c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    }   // else: Do not create agent.

    if (agent != nullptr) {
      cells[i]->enter(agent);
    }
  }
}


void initialize_simulation() {
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  create_agents<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


int main(int argc, char** argv) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  initialize_simulation();

  return 0;
}