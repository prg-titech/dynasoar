#include <chrono>

#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER

#include "sugarscape.h"

static const int kThreads = 256;
static const int kBlocks = (kSize*kSize + kThreads - 1)/kThreads;
__device__ static const int kNullptr = std::numeric_limits<int>::max();

__device__ curandState_t* dev_Cell_random_state;
__device__ Agent** dev_Cell_agent;
__device__ int* dev_Cell_sugar_diffusion;
__device__ int* dev_Cell_sugar;
__device__ int* dev_Cell_sugar_capacity;
__device__ int* dev_Cell_grow_rate;
// (No field for cell_id)


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ float Cell_random_float(int cell_id) {
  return curand_uniform(&dev_Cell_random_state[cell_id]);
}


__device__ int Cell_random_int(int cell_id, int a, int b) {
  return curand(&dev_Cell_random_state[cell_id]) % (b - a) + a;
}


__device__ bool Cell_is_free(int cell_id) {
  return dev_Cell_agent[cell_id] == nullptr;
}


__device__ int Cell_sugar(int cell_id) {
  return dev_Cell_sugar[cell_id];
}


__device__ void Cell_take_sugar(int cell_id, int amount) {
  dev_Cell_sugar[cell_id] -= amount;
}


__device__ void Cell_enter(int cell_id, Agent* agent) {
  assert(dev_Cell_agent[cell_id] == nullptr);
  assert(agent != nullptr);
  dev_Cell_agent[cell_id] = agent;
}


__device__ void Cell_leave(int cell_id) {
  assert(dev_Cell_agent[cell_id] != nullptr);
  dev_Cell_agent[cell_id] = nullptr;
}


__device__ void Cell_grow_sugar(int cell_id) {
  dev_Cell_sugar[cell_id] += min(
      dev_Cell_sugar_capacity[cell_id] - dev_Cell_sugar[cell_id],
      dev_Cell_grow_rate[cell_id]);
}


__device__ void new_Cell(int cell_id, int seed, int sugar, int sugar_capacity,
                         int max_grow_rate) {
  dev_Cell_sugar[cell_id] = sugar;
  dev_Cell_sugar_capacity[cell_id] = sugar_capacity;
  dev_Cell_agent[cell_id] = nullptr;

  curand_init(seed, cell_id, 0, &dev_Cell_random_state[cell_id]);

  // Set random grow rate.
  float r = curand_uniform(&dev_Cell_random_state[cell_id]);

  if (r <= 0.02) {
    dev_Cell_grow_rate[cell_id] = max_grow_rate;
  } else if (r <= 0.04) {
    dev_Cell_grow_rate[cell_id] = 0.5*max_grow_rate;
  } else if (r <= 0.08) {
    dev_Cell_grow_rate[cell_id] = 0.25*max_grow_rate;
  } else {
    dev_Cell_grow_rate[cell_id] = 0;
  }
}


__device__ Agent::Agent(int cell, int vision, int age, int max_age,
                        int endowment, int metabolism)
    : cell_(cell), cell_request_(kNullptr), vision_(vision), age_(age),
      max_age_(max_age), sugar_(endowment), endowment_(endowment),
      metabolism_(metabolism), permission_(false) {
  assert(cell != kNullptr);
  curand_init(Cell_random_int(cell, 0, kSize*kSize), 0, 0, &random_state_);
}


__device__ Male::Male(int cell, int vision, int age, int max_age,
                      int endowment, int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism),
      proposal_accepted_(false), female_request_(nullptr) {}


__device__ Female::Female(int cell, int vision, int age, int max_age,
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
    Cell_leave(cell_);
    destroy(device_allocator, this);
  }
}


__device__ void Agent::prepare_move() {
  // Move to cell with the most sugar.
  assert(cell_ != kNullptr);

  int turn = 0;
  int target_cell = kNullptr;
  int target_sugar = 0;

  int this_x = cell_ % kSize;
  int this_y = cell_ / kSize;

  for (int dx = -vision_; dx < vision_ + 1; ++dx) {
    for (int dy = -vision_; dy < vision_ + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        if (Cell_is_free(n_id)) {
          if (Cell_sugar(n_id) > target_sugar) {
            target_cell = n_id;
            target_sugar = Cell_sugar(n_id);
            turn = 1;
          } else if (Cell_sugar(n_id) == target_sugar) {
            // Select cell with probability 1/turn.
            if (random_float() <= 1.0f/(++turn)) {
              target_cell = n_id;
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
    assert(cell_request_ != kNullptr);
    assert(Cell_is_free(cell_request_));
    Cell_leave(cell_);
    Cell_enter(cell_request_, this);
    cell_ = cell_request_;
  }

  harvest_sugar();

  cell_request_ = kNullptr;
  permission_ = false;
}


__device__ void Agent::harvest_sugar() {
  // Harvest as much sugar as possible.
  int amount = Cell_sugar(cell_);
  Cell_take_sugar(cell_, amount);
  sugar_ += amount;
}


__device__ bool Agent::ready_to_mate() {
  // Half of endowment of sugar will go to the child. And the parent still
  // needs some sugar to survive.
  return (sugar_ >= endowment_ * 2 / 3) && age_ >= kMinMatingAge;
}


__device__ int Agent::cell_request() { return cell_request_; }


__device__ int Agent::sugar() { return sugar_; }


__device__ int Agent::vision() { return vision_; }


__device__ int Agent::max_age() { return max_age_; }


__device__ int Agent::endowment() { return endowment_; }


__device__ int Agent::metabolism() { return metabolism_; }


__device__ void Agent::take_sugar(int amount) { sugar_ -= amount; }


__device__ float Agent::random_float() {
  return curand_uniform(&random_state_);
}


__device__ void Cell_prepare_diffuse(int cell_id) {
  dev_Cell_sugar_diffusion[cell_id] =
      kSugarDiffusionRate * dev_Cell_sugar[cell_id];
  int max_diff = kMaxSugarDiffusion;
  if (dev_Cell_sugar_diffusion[cell_id] > max_diff) {
    dev_Cell_sugar_diffusion[cell_id] = max_diff;
  }

  dev_Cell_sugar[cell_id] -= dev_Cell_sugar_diffusion[cell_id];
}


__device__ void Cell_update_diffuse(int cell_id) {
  int new_sugar = 0;
  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        // Add sugar from neighboring 8 cells.
        new_sugar += 0.125f * dev_Cell_sugar_diffusion[n_id];
      }
    }
  }

  dev_Cell_sugar[cell_id] += new_sugar;
}


__device__ void Cell_decide_permission(int cell_id) {
  Agent* selected_agent = nullptr;
  int turn = 0;
  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;
        Agent* n_agent = dev_Cell_agent[n_id];

        if (n_agent != nullptr && n_agent->cell_request() == cell_id) {
          ++turn;

          // Select cell with probability 1/turn.
          if (Cell_random_float(cell_id) <= 1.0f/turn) {
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


__device__ void Male::propose() {
  if (ready_to_mate()) {
    // Propose to female with highest endowment.
    Female* target_agent = nullptr;
    int target_sugar = -1;

    int this_x = cell_ % kSize;
    int this_y = cell_ / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
          Female* n_female = dev_Cell_agent[n_id]->cast<Female>();

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
    int target_cell = kNullptr;
    int turn = 0;

    int this_x = cell_ % kSize;
    int this_y = cell_ / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;

          if (Cell_is_free(n_id)) {
            ++turn;

            // Select cell with probability 1/turn.
            if (random_float() <= 1.0f/turn) {
              target_cell = n_id;
            }
          }
        }
      }
    }

    assert((turn == 0) == (target_cell == kNullptr));
    cell_request_ = target_cell;
  }
}


__device__ void Male::mate() {
  if (proposal_accepted_ && permission_) {
    assert(female_request_ != nullptr);
    assert(cell_request_ != kNullptr);

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
          (int) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else {
      child = new(device_allocator) Female(
          (int) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism, female_request_->max_children());
    }

    // Add agent to target cell.
    assert(cell_request_ != kNullptr);
    assert(child != nullptr);
    assert(Cell_is_free(cell_request_));
    Cell_enter(cell_request_, child);
  }

  permission_ = false;
  proposal_accepted_ = false;
  female_request_ = nullptr;
  cell_request_ = kNullptr;
}


__device__ void Female::decide_proposal() {
  if (num_children_ < max_children_) {
    Male* selected_agent = nullptr;
    int selected_sugar = -1;
    int this_x = cell_ % kSize;
    int this_y = cell_ / kSize;

    for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
      for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
          Male* n_male = dev_Cell_agent[n_id]->cast<Male>();

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

__device__ void Cell_add_to_draw_array(int cell_id) {
  cell_info[cell_id].sugar = dev_Cell_sugar[cell_id];

  if (dev_Cell_agent[cell_id] == nullptr) {
    cell_info[cell_id].agent_type = 0;
  } else if (dev_Cell_agent[cell_id]->cast<Male>() != nullptr) {
    cell_info[cell_id].agent_type = 1;
  } else if (dev_Cell_agent[cell_id]->cast<Female>() != nullptr) {
    cell_info[cell_id].agent_type = 2;
  }
}


__global__ void kernel_Cell_add_to_draw_array() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_add_to_draw_array(i);
  }
}


__global__ void kernel_Cell_grow_sugar() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_grow_sugar(i);
  }
}


__global__ void kernel_Cell_prepare_diffuse() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_prepare_diffuse(i);
  }
}


__global__ void kernel_Cell_update_diffuse() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_update_diffuse(i);
  }
}


__global__ void kernel_Cell_decide_permission() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_decide_permission(i);
  }
}


void copy_data() {
  kernel_Cell_add_to_draw_array<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(host_cell_info, cell_info,
                       sizeof(CellInfo)*kSize*kSize, 0,
                      cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}


int checksum() {
  copy_data();
  int result = 0;
  for (int i = 0; i < kSize*kSize; ++i) {
    result += host_cell_info[i].agent_type; //(host_cell_info[i].sugar * i) % 1234567;
  }
  return result;
}


void step() {
  kernel_Cell_grow_sugar<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_prepare_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_update_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Agent, &Agent::age_and_metabolize>();
  allocator_handle->parallel_do<Agent, &Agent::prepare_move>();

  kernel_Cell_decide_permission<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Agent, &Agent::update_move>();

  allocator_handle->parallel_do<Male, &Male::propose>();
  allocator_handle->parallel_do<Female, &Female::decide_proposal>();
  allocator_handle->parallel_do<Male, &Male::propose_offspring_target>();

  kernel_Cell_decide_permission<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Male, &Male::mate>();

#ifdef OPTION_RENDER
  copy_data();
  draw(host_cell_info);
#endif  // OPTION_RENDER
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    new_Cell(i, kSeed, /*sugar=*/ 0, kSugarCapacity, /*max_grow_rate=*/ 50);
  }
}


__global__ void create_agents() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    float r = Cell_random_float(i);
    int c_vision = kMaxVision/2 + Cell_random_int(i, 0, kMaxVision/2);
    int c_max_age = kMaxAge*2/3 + Cell_random_int(i, 0, kMaxAge/3);
    int c_endowment = kMaxEndowment/4
                      + Cell_random_int(i, 0, kMaxEndowment*3/4);
    int c_metabolism = kMaxMetabolism/3
                       + Cell_random_int(i, 0, kMaxMetabolism*2/3);
    int c_max_children = Cell_random_int(i, 2, kMaxChildren);
    Agent* agent = nullptr;

    if (r < kProbMale) {
      // Create male agent.
      agent = new(device_allocator) Male(
          i, c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    } else if (r < kProbMale + kProbFemale) {
      // Create female agent.
      agent = new(device_allocator) Female(
          i, c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism,
          c_max_children);
    }   // else: Do not create agent.

    if (agent != nullptr) {
      Cell_enter(i, agent);
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

  // Allocate Cell memory.
  curandState_t* host_Cell_random_state;
  cudaMalloc(&host_Cell_random_state, sizeof(curandState_t)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_random_state, &host_Cell_random_state,
                     sizeof(curandState_t*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_sugar_diffusion;
  cudaMalloc(&host_Cell_sugar_diffusion, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_sugar_diffusion, &host_Cell_sugar_diffusion,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_sugar;
  cudaMalloc(&host_Cell_sugar, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_sugar, &host_Cell_sugar,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_sugar_capacity;
  cudaMalloc(&host_Cell_sugar_capacity, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_sugar_capacity, &host_Cell_sugar_capacity,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_grow_rate;
  cudaMalloc(&host_Cell_grow_rate, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_grow_rate, &host_Cell_grow_rate,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  Agent** host_Cell_agent;
  cudaMalloc(&host_Cell_agent, sizeof(Agent*)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_agent, &host_Cell_agent,
                     sizeof(Agent**), 0, cudaMemcpyHostToDevice);

  initialize_simulation();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumIterations; ++i) {
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
