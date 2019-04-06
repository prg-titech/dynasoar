#include <chrono>

#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER

#include "sugarscape.h"


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ Cell* cells[kSize*kSize];


__device__ Cell::Cell(int seed, int sugar, int sugar_capacity,
                      int max_grow_rate, int cell_id)
    : agent_(nullptr), sugar_(sugar), sugar_capacity_(sugar_capacity),
      cell_id_(cell_id) {
  curand_init(seed, cell_id, 0, &random_state_);

  // Set random grow rate.
  float r = curand_uniform(&random_state_);

  if (r <= 0.02) {
    grow_rate_ = max_grow_rate;
  } else if (r <= 0.04) {
    grow_rate_ = 0.5*max_grow_rate;
  } else if (r <= 0.08) {
    grow_rate_ = 0.25*max_grow_rate;
  } else {
    grow_rate_ = 0;
  }
}


__device__ Agent::Agent(Cell* cell, int vision, int age, int max_age,
                        int endowment, int metabolism)
    : cell_(cell), cell_request_(nullptr), vision_(vision), age_(age),
      max_age_(max_age), sugar_(endowment), endowment_(endowment),
      metabolism_(metabolism), permission_(false) {
  assert(cell != nullptr);
  curand_init(cell->random_int(0, kSize*kSize), 0, 0, &random_state_);
}


__device__ Male::Male(Cell* cell, int vision, int age, int max_age,
                      int endowment, int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism),
      proposal_accepted_(false), female_request_(nullptr) {}


__device__ Female::Female(Cell* cell, int vision, int age, int max_age,
                          int endowment, int metabolism, int max_children)
    : Agent(cell, vision, age, max_age, endowment, metabolism),
      num_children_(0), max_children_(max_children) {}



__device__ void Agent::give_permission() { permission_ = true; }


__device__ void Agent::age_and_metabolize() {
  bool dead = false;

  age_ = age_ + 1;
  dead = age_ > max_age_;

  sugar_ -= metabolism_;
  dead = dead || sugar_ <= 0;

  if (dead) {
    cell_->leave();
    destroy(device_allocator, this);
  }
}


__device__ void Agent::prepare_move() {
  // Move to cell with the most sugar.
  assert(cell_ != nullptr);

  int turn = 0;
  Cell* target_cell = nullptr;
  int target_sugar = 0;

  int this_x = cell_->cell_id() % kSize;
  int this_y = cell_->cell_id() / kSize;

  for (int dx = -vision_; dx < vision_ + 1; ++dx) {
    for (int dy = -vision_; dy < vision_ + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;
        Cell* n_cell = cells[n_id];
        assert(n_cell != nullptr);

        if (n_cell->is_free()) {
          if (n_cell->sugar() > target_sugar) {
            target_cell = n_cell;
            target_sugar = n_cell->sugar();
            turn = 1;
          } else if (n_cell->sugar() == target_sugar) {
            // Select cell with probability 1/turn.
            if (random_float() <= 1.0f/(++turn)) {
              target_cell = n_cell;
            }
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
    assert(cell_request_ != nullptr);
    assert(cell_request_->is_free());
    cell_->leave();
    cell_request_->enter(this);
    cell_ = cell_request_;
  }

  harvest_sugar();

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
  return (sugar_ >= endowment_ * 2 / 3) && age_ >= kMinMatingAge;
}


__device__ Cell* Agent::cell_request() { return cell_request_; }


__device__ int Agent::sugar() { return sugar_; }


__device__ int Agent::vision() { return vision_; }


__device__ int Agent::max_age() { return max_age_; }


__device__ int Agent::endowment() { return endowment_; }


__device__ int Agent::metabolism() { return metabolism_; }


__device__ void Agent::take_sugar(int amount) { sugar_ -= amount; }


__device__ float Agent::random_float() {
  return curand_uniform(&random_state_);
}


__device__ void Cell::prepare_diffuse() {
  sugar_diffusion_ = kSugarDiffusionRate * sugar_;
  int max_diff = kMaxSugarDiffusion;
  if (sugar_diffusion_ > max_diff) {
    sugar_diffusion_ = max_diff;
  }

  sugar_ -= sugar_diffusion_;
}


__device__ void Cell::update_diffuse() {
  int new_sugar = 0;
  int this_x = cell_id_ % kSize;
  int this_y = cell_id_ / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;
        Cell* n_cell = cells[n_id];

        // Add sugar from neighboring 8 cells.
        new_sugar += 0.125f * n_cell->sugar_diffusion_;
      }
    }
  }

  sugar_ += new_sugar;
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
  int this_x = cell_id_ % kSize;
  int this_y = cell_id_ / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;
        Cell* n_cell = cells[n_id];
        Agent* n_agent = n_cell->agent_;

        if (n_agent != nullptr && n_agent->cell_request() == this) {
          ++turn;

          // Select cell with probability 1/turn.
          if (random_float() <= 1.0f/turn) {
            selected_agent = n_agent;
          } else {
            assert(turn > 1);
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
  assert(agent != nullptr);
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

    int this_x = cell_->cell_id() % kSize;
    int this_y = cell_->cell_id() / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
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


__device__ Female* Male::female_request() { return female_request_; }


__device__ void Male::propose_offspring_target() {
  if (proposal_accepted_) {
    assert(female_request_ != nullptr);

    // Select a random cell.
    Cell* target_cell = nullptr;
    int turn = 0;

    int this_x = cell_->cell_id() % kSize;
    int this_y = cell_->cell_id() / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
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

    assert((turn == 0) == (target_cell == nullptr));
    cell_request_ = target_cell;
  }
}


__device__ void Male::mate() {
  if (proposal_accepted_ && permission_) {
    assert(female_request_ != nullptr);
    assert(cell_request_ != nullptr);

    // Register birth.
    female_request_->increment_num_children();

    // Take sugar from endowment.
    int c_endowment = (endowment_ + female_request_->endowment()) / 2;
    sugar_ -= endowment_ / 2;
    female_request_->take_sugar(female_request_->endowment() / 2);

    // Calculate other properties.
    int c_vision = (vision_ + female_request_->vision()) / 2;
    int c_max_age = (max_age_ + female_request_->max_age()) / 2;
    int c_metabolism = (metabolism_ + female_request_->metabolism()) / 2;

    // Create agent.
    // TODO: Check why type cast is necessary here.
    // Otherwise: unspecified launch failure.
    Agent* child;
    if (random_float() <= 0.5f) {
      child = new(device_allocator) Male(
          (Cell*) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else {
      child = new(device_allocator) Female(
          (Cell*) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism, female_request_->max_children());
    }

    // Add agent to target cell.
    assert(cell_request_ != nullptr);
    assert(child != nullptr);
    assert(cell_request_->is_free());
    cell_request_->enter(child);
  }

  permission_ = false;
  proposal_accepted_ = false;
  female_request_ = nullptr;
  cell_request_ = nullptr;
}


__device__ void Female::decide_proposal() {
  if (num_children_ < max_children_) {
    Male* selected_agent = nullptr;
    int selected_sugar = -1;
    int this_x = cell_->cell_id() % kSize;
    int this_y = cell_->cell_id() / kSize;

    for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
      for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
          Cell* n_cell = cells[n_id];
          Male* n_male = n_cell->agent()->cast<Male>();

          if (n_male != nullptr) {
            if (n_male->female_request() == this
                && n_male->sugar() > selected_sugar) {
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
}


// Only for rendering purposes and checksum computation.
__device__ CellInfo cell_info[kSize * kSize];
CellInfo host_cell_info[kSize * kSize];

__device__ void Cell::add_to_draw_array() {
  cell_info[cell_id_].sugar = sugar_;

  if (agent_ == nullptr) {
    cell_info[cell_id_].agent_type = 0;
  } else if (agent_->cast<Male>() != nullptr) {
    cell_info[cell_id_].agent_type = 1;
  } else if (agent_->cast<Female>() != nullptr) {
    cell_info[cell_id_].agent_type = 2;
  }
}


void copy_data() {
  allocator_handle->parallel_do<Cell, &Cell::add_to_draw_array>();
  cudaMemcpyFromSymbol(host_cell_info, cell_info,
                       sizeof(CellInfo)*kSize*kSize, 0,
                      cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}


int checksum() {
  copy_data();
  int result = 0;
  for (int i = 0; i < kSize*kSize; ++i) {
    result += host_cell_info[i].agent_type;  //(host_cell_info[i].sugar * i) % 1234567;
  }
  return result;
}


void step() {
  allocator_handle->parallel_do<Cell, &Cell::grow_sugar>();
  allocator_handle->parallel_do<Cell, &Cell::prepare_diffuse>();
  allocator_handle->parallel_do<Cell, &Cell::update_diffuse>();

  allocator_handle->parallel_do<Agent, &Agent::age_and_metabolize>();
  allocator_handle->parallel_do<Agent, &Agent::prepare_move>();
  allocator_handle->parallel_do<Cell, &Cell::decide_permission>();
  allocator_handle->parallel_do<Agent, &Agent::update_move>();

  allocator_handle->parallel_do<Male, &Male::propose>();
  allocator_handle->parallel_do<Female, &Female::decide_proposal>();
  allocator_handle->parallel_do<Male, &Male::propose_offspring_target>();
  allocator_handle->parallel_do<Cell, &Cell::decide_permission>();
  allocator_handle->parallel_do<Male, &Male::mate>();

#ifdef OPTION_RENDER
  copy_data();
  draw(host_cell_info);
#endif  // OPTION_RENDER
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    cells[i] = new(device_allocator) Cell(
        kSeed, /*sugar=*/ 0, /*sugar_capacity=*/ kSugarCapacity,
        /*max_grow_rate=*/ 50, /*cell_id=*/ i);
  }
}


__global__ void create_agents() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    float r = cells[i]->random_float();
    int c_vision = kMaxVision/2 + cells[i]->random_int(0, kMaxVision/2);
    int c_max_age = kMaxAge*2/3 + cells[i]->random_int(0, kMaxAge/3);
    int c_endowment = kMaxEndowment/4
                      + cells[i]->random_int(0, kMaxEndowment*3/4);
    int c_metabolism = kMaxMetabolism/3
                       + cells[i]->random_int(0, kMaxMetabolism*2/3);
    int c_max_children = cells[i]->random_int(2, kMaxChildren);
    Agent* agent = nullptr;

    if (r < kProbMale) {
      // Create male agent.
      agent = new(device_allocator) Male(
          cells[i], c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else if (r < kProbMale + kProbFemale) {
      // Create female agent.
      agent = new(device_allocator) Female(
          cells[i], c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism, c_max_children);
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


int main(int /*argc*/, char** /*argv*/) {
#ifdef OPTION_RENDER
  init_renderer();
#endif  // OPTION_RENDER

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  initialize_simulation();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumIterations; ++i) {
#ifndef NDEBUG
    allocator_handle->DBG_print_state_stats();
#endif  // NDEBUG

    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

  printf("%lu, %lu\n", micros, allocator_handle->DBG_get_enumeration_time());

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
