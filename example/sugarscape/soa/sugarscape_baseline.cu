#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>

#include "configuration.h"
#include "rendering.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using IndexT = int;

static const int kThreads = 256;
static const int kBlocks = (kSize*kSize + kThreads - 1)/kThreads;
static const IndexT kNullptr = std::numeric_limits<IndexT>::max();

static const char kNoType = 0;
static const char kClassMale = 1;
static const char kClassFemale = 2;

__device__ curandState_t* dev_Cell_random_state;
// (No field for agent)
__device__ int* dev_Cell_sugar_diffusion;
__device__ int* dev_Cell_sugar;
__device__ int* dev_Cell_sugar_capacity;
__device__ int* dev_Cell_grow_rate;
// (No field for cell_id)
__device__ char* dev_Cell_Agent_type;
__device__ curandState_t* dev_Cell_Agent_random_state;
// (No field for cell)
__device__ IndexT* dev_Cell_Agent_cell_request;
__device__ int* dev_Cell_Agent_vision;
__device__ int* dev_Cell_Agent_age;
__device__ int* dev_Cell_Agent_max_age;
__device__ int* dev_Cell_Agent_sugar;
__device__ int* dev_Cell_Agent_metabolism;
__device__ int* dev_Cell_Agent_endowment;
__device__ bool* dev_Cell_Agent_permission;
__device__ IndexT* dev_Cell_Male_female_request;
__device__ bool* dev_Cell_Male_proposal_accepted;


__device__ float Cell_random_float(IndexT cell_id) {
  return curand_uniform(&dev_Cell_random_state[cell_id]);
}


__device__ int Cell_random_int(IndexT cell_id, int a, int b) {
  return curand(&dev_Cell_random_state[cell_id]) % (b - a) + a;
}


__device__ float Agent_random_float(IndexT cell_id) {
  return curand_uniform(&dev_Cell_Agent_random_state[cell_id]);
}


__device__ bool Cell_is_free(IndexT cell_id) {
  return dev_Cell_Agent_type[cell_id] == kNoType;
}


__device__ void Cell_enter(IndexT cell_id, int agent) {
  assert(cell_id >= 0 && cell_id < kSize*kSize);
  assert(agent >= 0 && agent < kSize*kSize);
  assert(dev_Cell_Agent_type[cell_id] == kNoType);
  assert(dev_Cell_Agent_type[agent] != kNoType);

  dev_Cell_Agent_type[cell_id] = dev_Cell_Agent_type[agent];
  dev_Cell_Agent_random_state[cell_id] = dev_Cell_Agent_random_state[agent];
  dev_Cell_Agent_vision[cell_id] = dev_Cell_Agent_vision[agent];
  dev_Cell_Agent_age[cell_id] = dev_Cell_Agent_age[agent];
  dev_Cell_Agent_max_age[cell_id] = dev_Cell_Agent_max_age[agent];
  dev_Cell_Agent_sugar[cell_id] = dev_Cell_Agent_sugar[agent];
  dev_Cell_Agent_metabolism[cell_id] = dev_Cell_Agent_metabolism[agent];
  dev_Cell_Agent_endowment[cell_id] = dev_Cell_Agent_endowment[agent];
  // No need to copy the other fields.
  dev_Cell_Male_female_request[cell_id] = kNullptr;
  dev_Cell_Male_proposal_accepted[cell_id] = false;
}


__device__ void Cell_leave(IndexT cell_id) {
  assert(dev_Cell_Agent_type[cell_id] != kNoType);
  dev_Cell_Agent_type[cell_id] = kNoType;
  dev_Cell_Agent_permission[cell_id] = false;
  dev_Cell_Agent_cell_request[cell_id] = kNullptr;
  dev_Cell_Male_female_request[cell_id] = kNullptr;
  dev_Cell_Male_proposal_accepted[cell_id] = false;
}


__device__ int xxx69;
__device__ void Agent_harvest_sugar(IndexT cell_id) {
  // Harvest as much sugar as possible.
  // TODO: Do we need two sugar fields here?
  int a = dev_Cell_Agent_sugar[cell_id];
  assert(dev_Cell_Agent_sugar[cell_id] < 9999999);
  //dev_Cell_Agent_sugar[cell_id] += dev_Cell_sugar[cell_id];
  //dev_Cell_sugar[cell_id] = 0;

  int b =dev_Cell_sugar[cell_id];
  assert(b < 9999999);
  xxx69 = b;

  if (a != dev_Cell_Agent_sugar[cell_id]) {
    printf("[%i]   %i -- %i -- %i\n", cell_id, a, dev_Cell_Agent_sugar[cell_id], dev_Cell_sugar[cell_id]);
  }
  //assert(a == dev_Cell_Agent_sugar[cell_id]);
}


__device__ bool Agent_ready_to_mate(IndexT cell_id) {
  // Half of endowment of sugar will go to the child. And the parent still
  // needs some sugar to survive.
  return (dev_Cell_Agent_sugar[cell_id]
          >= dev_Cell_Agent_endowment[cell_id] * 2 / 3)
      && dev_Cell_Agent_age[cell_id] >= kMinMatingAge;
}


__device__ void new_Cell(IndexT cell_id, int seed, int sugar,
                         int sugar_capacity, int max_grow_rate) {
  dev_Cell_sugar[cell_id] = sugar;
  dev_Cell_sugar_capacity[cell_id] = sugar_capacity;
  dev_Cell_Agent_type[cell_id] = kNoType;

  curand_init(seed, cell_id, 0, &dev_Cell_random_state[cell_id]);

  // Set random grow rate.
  float r = curand_uniform(&dev_Cell_random_state[cell_id]);

  if (r <= 0.01) {
    dev_Cell_grow_rate[cell_id] = max_grow_rate;
  } else if (r <= 0.05) {
    dev_Cell_grow_rate[cell_id] = 0.5*max_grow_rate;
  } else if (r <= 0.07) {
    dev_Cell_grow_rate[cell_id] = 0.25*max_grow_rate;
  } else {
    dev_Cell_grow_rate[cell_id] = 0;
  }
}


__device__ void new_Agent(IndexT cell_id, int vision, int age, int max_age,
                          int endowment, int metabolism) {
  assert(cell_id != kNullptr);
  assert(dev_Cell_Agent_type[cell_id] == kNoType);

  dev_Cell_Agent_cell_request[cell_id] = kNullptr;
  dev_Cell_Agent_vision[cell_id] = vision;
  dev_Cell_Agent_age[cell_id] = age;
  dev_Cell_Agent_max_age[cell_id] = max_age;
  dev_Cell_Agent_sugar[cell_id] = endowment;
  dev_Cell_Agent_endowment[cell_id] = endowment;
  dev_Cell_Agent_metabolism[cell_id] = metabolism;
  dev_Cell_Agent_permission[cell_id] = false;

  curand_init(Cell_random_int(cell_id, 0, kSize*kSize), 0, 0,
  	          &dev_Cell_Agent_random_state[cell_id]);
}


__device__ void new_Male(IndexT cell_id, int vision, int age, int max_age,
                         int endowment, int metabolism) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_Cell_Male_proposal_accepted[cell_id] = false;
  dev_Cell_Male_female_request[cell_id] = kNullptr;
  dev_Cell_Agent_type[cell_id] = kClassMale;
}


__device__ void new_Female(IndexT cell_id, int vision, int age, int max_age,
                           int endowment, int metabolism) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_Cell_Agent_type[cell_id] = kClassFemale;
}


__device__ void Agent_age_and_metabolize(IndexT cell_id) {
  assert(dev_Cell_Agent_type[cell_id] != kNoType);
  bool dead = false;

  dev_Cell_Agent_age[cell_id] = dev_Cell_Agent_age[cell_id] + 1;
  dead = dev_Cell_Agent_age[cell_id] > dev_Cell_Agent_max_age[cell_id];

  dev_Cell_Agent_sugar[cell_id] -= dev_Cell_Agent_metabolism[cell_id];
  dead = dead || (dev_Cell_Agent_sugar[cell_id] <= 0);

  if (dead) {
    Cell_leave(cell_id);
    // No delete in baseline implementation.
  }
}


__device__ void Agent_prepare_move(IndexT cell_id) {
  // Move to cell with the most sugar.
  assert(dev_Cell_Agent_type[cell_id] != kNoType);
  assert(dev_Cell_Agent_cell_request[cell_id] == kNullptr);
  int turn = 0;
  IndexT target_cell = kNullptr;
  int target_sugar = 0;

  IndexT this_x = cell_id % kSize;
  IndexT this_y = cell_id / kSize;

  for (int dx = -dev_Cell_Agent_vision[cell_id];
       dx < dev_Cell_Agent_vision[cell_id] + 1; ++dx) {
    for (int dy = -dev_Cell_Agent_vision[cell_id];
         dy < dev_Cell_Agent_vision[cell_id] + 1; ++dy) {
      IndexT nx = this_x + dx;
      IndexT ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        IndexT n_id = nx + ny*kSize;

        if (Cell_is_free(n_id)) {
          if (dev_Cell_sugar[n_id] > target_sugar) {
            target_cell = n_id;
            target_sugar = dev_Cell_sugar[n_id];
            turn = 1;
          } else if (dev_Cell_sugar[n_id] == target_sugar) {
            // Select cell with probability 1/turn.
            if (Agent_random_float(cell_id) <= 1.0f/(++turn)) {
              target_cell = n_id;
            }
          }
        }
      }
    }
  }

  dev_Cell_Agent_cell_request[cell_id] = target_cell;
}


__device__ void Agent_update_move(IndexT cell_id) {
  assert(cell_id < kSize*kSize);
  assert(!Cell_is_free(cell_id));
  Agent_harvest_sugar(cell_id);

  int new_cell = cell_id;

  if (dev_Cell_Agent_permission[cell_id]) {
    // Have permission to enter the cell.
    new_cell = dev_Cell_Agent_cell_request[cell_id];
    assert(new_cell != kNullptr);
    assert(new_cell >= 0);
    assert(new_cell < kSize*kSize);
    assert(Cell_is_free(new_cell));
    Cell_enter(new_cell, cell_id);
    Cell_leave(cell_id);
  }

  dev_Cell_Agent_permission[new_cell] = false;
  dev_Cell_Agent_cell_request[new_cell] = kNullptr;
}


__device__ void Cell_prepare_diffuse(IndexT cell_id) {
  dev_Cell_sugar_diffusion[cell_id] =
      kSugarDiffusionRate * dev_Cell_sugar[cell_id];
  int max_diff = kMaxSugarDiffusion;
  if (dev_Cell_sugar_diffusion[cell_id] > max_diff) {
    dev_Cell_sugar_diffusion[cell_id] = max_diff;
  }

  dev_Cell_sugar[cell_id] -= dev_Cell_sugar_diffusion[cell_id];
}


__device__ void Cell_update_diffuse(IndexT cell_id) {
  int new_sugar = 0;
  IndexT this_x = cell_id % kSize;
  IndexT this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      IndexT nx = this_x + dx;
      IndexT ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        IndexT n_id = nx + ny*kSize;

        // Add sugar from neighboring 8 cells.
        new_sugar += 0.125f * dev_Cell_sugar_diffusion[n_id];
      }
    }
  }

  dev_Cell_sugar[cell_id] += new_sugar;
}


__device__ void Cell_decide_permission(IndexT cell_id) {
  IndexT selected_agent = kNullptr;
  int turn = 0;
  IndexT this_x = cell_id % kSize;
  IndexT this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      IndexT nx = this_x + dx;
      IndexT ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        IndexT n_id = nx + ny*kSize;

        if (dev_Cell_Agent_type[n_id] != kNoType
            && dev_Cell_Agent_cell_request[n_id] == cell_id) {
          ++turn;

          // Select cell with probability 1/turn.
          if (Cell_random_float(cell_id) <= 1.0f/turn) {
            selected_agent = n_id;
          } else {
            assert(turn > 1);
          }
        }
      }
    }
  }

  assert((turn == 0) == (selected_agent == kNullptr));

  if (selected_agent != kNullptr) {
    assert(Cell_is_free(cell_id));
    assert(dev_Cell_Agent_cell_request[selected_agent] == cell_id);
    dev_Cell_Agent_permission[selected_agent] = true;
  }
}


__device__ void Cell_grow_sugar(IndexT cell_id) {
  dev_Cell_sugar[cell_id] += min(
      dev_Cell_sugar_capacity[cell_id] - dev_Cell_sugar[cell_id],
      dev_Cell_grow_rate[cell_id]);
}


__device__ void Male_propose(IndexT cell_id) {
  if (Agent_ready_to_mate(cell_id)) {
    // Propose to female with highest endowment.
    IndexT target_agent = kNullptr;
    int target_sugar = -1;

    IndexT this_x = cell_id % kSize;
    IndexT this_y = cell_id / kSize;

    for (int dx = -dev_Cell_Agent_vision[cell_id];
         dx < dev_Cell_Agent_vision[cell_id] + 1; ++dx) {
      for (int dy = -dev_Cell_Agent_vision[cell_id];
           dy < dev_Cell_Agent_vision[cell_id] + 1; ++dy) {
        IndexT nx = this_x + dx;
        IndexT ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          IndexT n_id = nx + ny*kSize;

          if (dev_Cell_Agent_type[n_id] == kClassFemale
              && Agent_ready_to_mate(n_id)) {
            if (dev_Cell_Agent_sugar[n_id] > target_sugar) {
              target_agent = n_id;
              target_sugar = dev_Cell_Agent_sugar[n_id];
            }
          }
        }
      }
    }

    assert((target_sugar == -1) == (target_agent == kNullptr));
    dev_Cell_Male_female_request[cell_id] = target_agent;
  }
}


__device__ void Male_propose_offspring_target(IndexT cell_id) {
  if (dev_Cell_Male_proposal_accepted[cell_id]) {
    assert(dev_Cell_Male_female_request[cell_id] != kNullptr);

    // Select a random cell.
    IndexT target_cell = kNullptr;
    int turn = 0;

    IndexT this_x = cell_id % kSize;
    IndexT this_y = cell_id / kSize;

    for (int dx = -dev_Cell_Agent_vision[cell_id];
         dx < dev_Cell_Agent_vision[cell_id] + 1; ++dx) {
      for (int dy = -dev_Cell_Agent_vision[cell_id];
           dy < dev_Cell_Agent_vision[cell_id] + 1; ++dy) {
        IndexT nx = this_x + dx;
        IndexT ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          IndexT n_id = nx + ny*kSize;

          if (Cell_is_free(n_id)) {
            ++turn;

            // Select cell with probability 1/turn.
            if (Agent_random_float(cell_id) <= 1.0f/turn) {
              target_cell = n_id;
            }
          }
        }
      }
    }

    assert((turn == 0) == (target_cell == kNullptr));
    dev_Cell_Agent_cell_request[cell_id] = target_cell;
  }
}


__device__ void Male_mate(IndexT cell_id) {
  if (dev_Cell_Male_proposal_accepted[cell_id]
      && dev_Cell_Agent_permission[cell_id]) {
    assert(dev_Cell_Male_female_request[cell_id] != kNullptr);
    assert(dev_Cell_Agent_cell_request[cell_id] != kNullptr);

    // Take sugar from endowment.
    int c_endowment = (dev_Cell_Agent_endowment[cell_id]
        + dev_Cell_Agent_endowment[dev_Cell_Male_female_request[cell_id]]) / 2;
    dev_Cell_Agent_sugar[cell_id] -= dev_Cell_Agent_endowment[cell_id] / 2;
    dev_Cell_Agent_sugar[dev_Cell_Male_female_request[cell_id]]
        -= dev_Cell_Agent_endowment[dev_Cell_Male_female_request[cell_id]] / 2;

    // Calculate other properties.
    int c_vision = (dev_Cell_Agent_vision[cell_id]
        + dev_Cell_Agent_vision[dev_Cell_Male_female_request[cell_id]]) / 2;
    int c_max_age = (dev_Cell_Agent_max_age[cell_id]
        + dev_Cell_Agent_max_age[dev_Cell_Male_female_request[cell_id]]) / 2;
    int c_metabolism = (dev_Cell_Agent_metabolism[cell_id]
        + dev_Cell_Agent_metabolism[dev_Cell_Male_female_request[cell_id]]) / 2;


    // Create agent.
    assert(dev_Cell_Agent_cell_request[cell_id] != kNullptr);
    assert(dev_Cell_Agent_type[dev_Cell_Agent_cell_request[cell_id]] == kNoType);

    if (Agent_random_float(cell_id) <= 0.5f) {
      new_Male(dev_Cell_Agent_cell_request[cell_id],
               2*c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    } else {
      new_Female(dev_Cell_Agent_cell_request[cell_id],
                 c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    }
  }

  dev_Cell_Agent_permission[cell_id] = false;
  dev_Cell_Male_proposal_accepted[cell_id] = false;
  dev_Cell_Male_female_request[cell_id] = kNullptr;
  dev_Cell_Agent_cell_request[cell_id] = kNullptr;
}


__device__ void Female_decide_proposal(IndexT cell_id) {
  IndexT selected_agent = kNullptr;
  int selected_sugar = -1;
  IndexT this_x = cell_id % kSize;
  IndexT this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      IndexT nx = this_x + dx;
      IndexT ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        IndexT n_id = nx + ny*kSize;

        if (dev_Cell_Agent_type[n_id] == kClassMale) {
          if (dev_Cell_Male_female_request[n_id] == cell_id
              && dev_Cell_Agent_sugar[n_id] > selected_sugar) {
            selected_agent = n_id;
            selected_sugar = dev_Cell_Agent_sugar[n_id];
          }
        }
      }
    }
  }

  assert((selected_sugar == -1) == (selected_agent == kNullptr));

  if (selected_agent != kNullptr) {
    dev_Cell_Male_proposal_accepted[selected_agent] = true;
  }
}


__global__ void kernel_Cell_grow_sugar() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_grow_sugar(i);
  }
}


__global__ void kernel_Cell_prepare_diffuse() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_prepare_diffuse(i);
  }
}


__global__ void kernel_Cell_update_diffuse() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_update_diffuse(i);
  }
}


__global__ void kernel_Agent_age_and_metabolize() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] != kNoType) {
      Agent_age_and_metabolize(i);
    }
  }
}


__global__ void kernel_Agent_prepare_move() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] != kNoType) {
      Agent_prepare_move(i);
    }
  }
}


__global__ void kernel_Cell_decide_permission() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_decide_permission(i);
  }
}


__global__ void kernel_Agent_update_move() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] != kNoType) {
      Agent_update_move(i);
    }
  }
}


__global__ void kernel_Male_propose() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] == kClassMale) {
      Male_propose(i);
    }
  }
}


__global__ void kernel_Female_decide_proposal() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] == kClassFemale) {
      Female_decide_proposal(i);
    }
  }
}


__global__ void kernel_Male_propose_offspring_target() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] == kClassMale) {
      Male_propose_offspring_target(i);
    }
  }
}


__global__ void kernel_Male_mate() {
  for (IndexT i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_Cell_Agent_type[i] == kClassMale) {
      Male_mate(i);
    }
  }
}


void step() {
  /*
  kernel_Cell_grow_sugar<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_prepare_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_update_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
  */

  kernel_Agent_age_and_metabolize<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_prepare_move<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_decide_permission<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_update_move<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

/*
  kernel_Male_propose<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Female_decide_proposal<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Male_propose_offspring_target<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_decide_permission<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Male_mate<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
*/
}


__global__ void create_cells() {
  for (IndexT i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    new_Cell(i, kSeed, /*sugar=*/ 0, kSugarCapacity, /*max_grow_rate=*/ 50);
  }
}


__global__ void create_agents() {
  for (IndexT i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    float r = Cell_random_float(i);
    int c_vision = kMaxVision/2 + Cell_random_int(i, 0, kMaxVision/2);
    int c_max_age = kMaxAge*2/3 + Cell_random_int(i, 0, kMaxAge/3);
    int c_endowment = kMaxEndowment/4
                      + Cell_random_int(i, 0, kMaxEndowment*3/4);
    int c_metabolism = kMaxMetabolism/3
                       + Cell_random_int(i, 0, kMaxMetabolism*2/3);

    if (r < kProbMale) {
      // Create male agent.
      new_Male(i, 2*c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    } else if (r < kProbMale + kProbFemale) {
      // Create female agent.
      new_Female(i, c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    }   // else: Do not create agent.
  }
}


void initialize_simulation() {
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  create_agents<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


int data_Cell_sugar[kSize*kSize];
char data_Cell_type[kSize*kSize];
CellInfo host_cell_info[kSize*kSize];

void transfer_data(int* host_Cell_sugar, char* host_Cell_Agent_type) {
  cudaMemcpy(data_Cell_sugar, host_Cell_sugar, sizeof(int)*kSize*kSize,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data_Cell_type, host_Cell_Agent_type, sizeof(char)*kSize*kSize,
             cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kSize*kSize; ++i) {
    host_cell_info[i].sugar = data_Cell_sugar[i];
    host_cell_info[i].agent_type = data_Cell_type[i];
  }
}


int checksum(int* host_Cell_sugar, char* host_Cell_Agent_type) {
  transfer_data(host_Cell_sugar, host_Cell_Agent_type);
  int result = 0;
  for (int i = 0; i < kSize*kSize; ++i) {
    result += data_Cell_type[i];
    //result %= 12456789;
  }
  return result;
}


int main(int /*argc*/, char** /*argv*/) {
  if (kOptionRender) {
    init_renderer();
  }

  // Allocate memory.
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

  char* host_Cell_Agent_type;
  cudaMalloc(&host_Cell_Agent_type, sizeof(char)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_type, &host_Cell_Agent_type,
                     sizeof(char*), 0, cudaMemcpyHostToDevice);

  curandState_t* host_Cell_Agent_random_state;
  cudaMalloc(&host_Cell_Agent_random_state, sizeof(curandState_t)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_random_state, &host_Cell_Agent_random_state,
                     sizeof(curandState_t*), 0, cudaMemcpyHostToDevice);

  IndexT* host_Cell_Agent_cell_request;
  cudaMalloc(&host_Cell_Agent_cell_request, sizeof(IndexT)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_cell_request, &host_Cell_Agent_cell_request,
                     sizeof(IndexT*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_vision;
  cudaMalloc(&host_Cell_Agent_vision, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_vision, &host_Cell_Agent_vision,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_age;
  cudaMalloc(&host_Cell_Agent_age, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_age, &host_Cell_Agent_age,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_max_age;
  cudaMalloc(&host_Cell_Agent_max_age, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_max_age, &host_Cell_Agent_max_age,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_sugar;
  cudaMalloc(&host_Cell_Agent_sugar, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_sugar, &host_Cell_Agent_sugar,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_metabolism;
  cudaMalloc(&host_Cell_Agent_metabolism, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_metabolism, &host_Cell_Agent_metabolism,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  int* host_Cell_Agent_endowment;
  cudaMalloc(&host_Cell_Agent_endowment, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_endowment, &host_Cell_Agent_endowment,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  bool* host_Cell_Agent_permission;
  cudaMalloc(&host_Cell_Agent_permission, sizeof(bool)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_permission, &host_Cell_Agent_permission,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);

  IndexT* host_Cell_Male_female_request;
  cudaMalloc(&host_Cell_Male_female_request, sizeof(IndexT)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Male_female_request,
                     &host_Cell_Male_female_request,
                     sizeof(IndexT*), 0, cudaMemcpyHostToDevice);

  bool* host_Cell_Male_proposal_accepted;
  cudaMalloc(&host_Cell_Male_proposal_accepted, sizeof(bool)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Male_proposal_accepted,
                     &host_Cell_Male_proposal_accepted,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);
  assert(host_Cell_Male_proposal_accepted != nullptr);

  gpuErrchk(cudaDeviceSynchronize());

  initialize_simulation();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumIterations; ++i) {
    //printf("%i\n", i);
    //printf("Checksum: %i\n", checksum(host_Cell_sugar, host_Cell_Agent_type));
    step();

    if (kOptionRender) {
      transfer_data(host_Cell_sugar, host_Cell_Agent_type);
      draw(host_cell_info);
    }
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("Time: %lu ms\n", millis);

  if (kOptionRender) {
    close_renderer();
  }

  printf("Checksum: %i\n", checksum(host_Cell_sugar, host_Cell_Agent_type));
  return 0;

  // TODO: Free CUDA memory.
}
