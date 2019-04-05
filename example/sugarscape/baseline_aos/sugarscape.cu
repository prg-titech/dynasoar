#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>

#include "../configuration.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const int kThreads = 256;
static const int kBlocks = (kSize*kSize + kThreads - 1)/kThreads;
static const int kNullptr = std::numeric_limits<int>::max();

static const char kNoType = 0;
static const char kClassMale = 1;
static const char kClassFemale = 2;

struct Cell {
  curandState_t random_state;
  curandState_t Agent_random_state;
  int sugar_diffusion;
  int sugar;
  int sugar_capacity;
  int grow_rate;
  int Agent_cell_request;
  int Agent_vision;
  int Agent_age;
  int Agent_max_age;
  int Agent_sugar;
  int Agent_metabolism;
  int Agent_endowment;
  int Male_female_request;
  int Female_num_children;
  int Female_max_children;
  bool Male_proposal_accepted;
  bool Agent_permission;
  char Agent_type;
};

__device__ Cell* dev_cells;

// For computing the checksum.
__device__ int* dev_Cell_sugar;
__device__ char* dev_Cell_Agent_type;


__device__ float Cell_random_float(int cell_id) {
  return curand_uniform(&dev_cells[cell_id].random_state);
}


__device__ int Cell_random_int(int cell_id, int a, int b) {
  return curand(&dev_cells[cell_id].random_state) % (b - a) + a;
}


__device__ float Agent_random_float(int cell_id) {
  return curand_uniform(&dev_cells[cell_id].Agent_random_state);
}


__device__ bool Cell_is_free(int cell_id) {
  return dev_cells[cell_id].Agent_type == kNoType;
}


__device__ void Cell_enter(int cell_id, int agent) {
  assert(dev_cells[cell_id].Agent_type == kNoType);
  assert(dev_cells[agent].Agent_type != kNoType);

  dev_cells[cell_id].Agent_permission = false;
  dev_cells[cell_id].Male_proposal_accepted = false;
  dev_cells[cell_id].Male_female_request = kNullptr;
  dev_cells[cell_id].Agent_cell_request = kNullptr;

  // Threadfence to make sure that cell will not be processed by accident.
  // E.g.: permission set to false first before setting new type.
  __threadfence();

  dev_cells[cell_id].Agent_type = dev_cells[agent].Agent_type;

  __threadfence();

  dev_cells[cell_id].Agent_random_state = dev_cells[agent].Agent_random_state;
  dev_cells[cell_id].Agent_vision = dev_cells[agent].Agent_vision;
  dev_cells[cell_id].Agent_age = dev_cells[agent].Agent_age;
  dev_cells[cell_id].Agent_max_age = dev_cells[agent].Agent_max_age;
  dev_cells[cell_id].Agent_sugar = dev_cells[agent].Agent_sugar;
  dev_cells[cell_id].Agent_metabolism = dev_cells[agent].Agent_metabolism;
  dev_cells[cell_id].Agent_endowment = dev_cells[agent].Agent_endowment;
  dev_cells[cell_id].Female_max_children = dev_cells[agent].Female_max_children;
  dev_cells[cell_id].Female_num_children = dev_cells[agent].Female_num_children;
}


__device__ void Cell_leave(int cell_id) {
  assert(dev_cells[cell_id].Agent_type != kNoType);
  dev_cells[cell_id].Agent_type = kNoType;
  dev_cells[cell_id].Agent_permission = false;
  dev_cells[cell_id].Male_proposal_accepted = false;
  dev_cells[cell_id].Male_female_request = kNullptr;
  dev_cells[cell_id].Agent_cell_request = kNullptr;
}


__device__ void Agent_harvest_sugar(int cell_id) {
  // Harvest as much sugar as possible.
  // TODO: Do we need two sugar fields here?
  dev_cells[cell_id].Agent_sugar += dev_cells[cell_id].sugar;
  dev_cells[cell_id].sugar = 0;
}


__device__ bool Agent_ready_to_mate(int cell_id) {
  // Half of endowment of sugar will go to the child. And the parent still
  // needs some sugar to survive.
  return (dev_cells[cell_id].Agent_sugar
          >= dev_cells[cell_id].Agent_endowment * 2 / 3)
      && dev_cells[cell_id].Agent_age >= kMinMatingAge;
}


__device__ void new_Cell(int cell_id, int seed, int sugar, int sugar_capacity,
                         int max_grow_rate) {
  dev_cells[cell_id].sugar = sugar;
  dev_cells[cell_id].sugar_capacity = sugar_capacity;

  curand_init(seed, cell_id, 0, &dev_cells[cell_id].random_state);

  // Set random grow rate.
  float r = curand_uniform(&dev_cells[cell_id].random_state);

  if (r <= 0.02) {
    dev_cells[cell_id].grow_rate = max_grow_rate;
  } else if (r <= 0.04) {
    dev_cells[cell_id].grow_rate = 0.5*max_grow_rate;
  } else if (r <= 0.08) {
    dev_cells[cell_id].grow_rate = 0.25*max_grow_rate;
  } else {
    dev_cells[cell_id].grow_rate = 0;
  }
}


__device__ void new_Agent(int cell_id, int vision, int age, int max_age,
                          int endowment, int metabolism) {
  assert(cell_id != kNullptr);
  dev_cells[cell_id].Agent_cell_request = kNullptr;
  dev_cells[cell_id].Agent_vision = vision;
  dev_cells[cell_id].Agent_age = age;
  dev_cells[cell_id].Agent_max_age = max_age;
  dev_cells[cell_id].Agent_sugar = endowment;
  dev_cells[cell_id].Agent_endowment = endowment;
  dev_cells[cell_id].Agent_metabolism = metabolism;
  dev_cells[cell_id].Agent_permission = false;

  curand_init(Cell_random_int(cell_id, 0, kSize*kSize), 0, 0,
              &dev_cells[cell_id].Agent_random_state);
}


__device__ void new_Male(int cell_id, int vision, int age, int max_age,
                         int endowment, int metabolism) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_cells[cell_id].Male_proposal_accepted = false;
  dev_cells[cell_id].Male_female_request = kNullptr;

  __threadfence();

  dev_cells[cell_id].Agent_type = kClassMale;
}


__device__ void new_Female(int cell_id, int vision, int age, int max_age,
                           int endowment, int metabolism, int max_children) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_cells[cell_id].Female_num_children = 0;
  dev_cells[cell_id].Female_max_children = max_children;

  __threadfence();

  dev_cells[cell_id].Agent_type = kClassFemale;
}


__device__ void Agent_give_permission(int cell_id) {
  dev_cells[cell_id].Agent_permission = true;
}


__device__ void Agent_age_and_metabolize(int cell_id) {
  bool dead = false;

  dev_cells[cell_id].Agent_age = dev_cells[cell_id].Agent_age + 1;
  dead = dev_cells[cell_id].Agent_age > dev_cells[cell_id].Agent_max_age;

  dev_cells[cell_id].Agent_sugar -= dev_cells[cell_id].Agent_metabolism;
  dead = dead || dev_cells[cell_id].Agent_sugar <= 0;

  if (dead) {
    Cell_leave(cell_id);
    // No delete in baseline implementation.
  }
}


__device__ void Agent_prepare_move(int cell_id) {
  // Move to cell with the most sugar.
  int turn = 0;
  int target_cell = kNullptr;
  int target_sugar = 0;

  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -dev_cells[cell_id].Agent_vision;
       dx < dev_cells[cell_id].Agent_vision + 1; ++dx) {
    for (int dy = -dev_cells[cell_id].Agent_vision;
         dy < dev_cells[cell_id].Agent_vision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        if (Cell_is_free(n_id)) {
          if (dev_cells[n_id].sugar > target_sugar) {
            target_cell = n_id;
            target_sugar = dev_cells[n_id].sugar;
            turn = 1;
          } else if (dev_cells[n_id].sugar == target_sugar) {
            // Select cell with probability 1/turn.
            if (Agent_random_float(cell_id) <= 1.0f/(++turn)) {
              target_cell = n_id;
            }
          }
        }
      }
    }
  }

  dev_cells[cell_id].Agent_cell_request = target_cell;
}


__device__ void Agent_update_move(int cell_id) {
  if (dev_cells[cell_id].Agent_permission) {
    // Have permission to enter the cell.
    assert(dev_cells[cell_id].Agent_cell_request != kNullptr);
    assert(Cell_is_free(dev_cells[cell_id].Agent_cell_request));
    Cell_enter(dev_cells[cell_id].Agent_cell_request, cell_id);
    Cell_leave(cell_id);
  }

  dev_cells[cell_id].Agent_cell_request = kNullptr;
  dev_cells[cell_id].Agent_permission = false;
}


__device__ void Cell_prepare_diffuse(int cell_id) {
  dev_cells[cell_id].sugar_diffusion =
      kSugarDiffusionRate * dev_cells[cell_id].sugar;
  int max_diff = kMaxSugarDiffusion;
  if (dev_cells[cell_id].sugar_diffusion > max_diff) {
    dev_cells[cell_id].sugar_diffusion = max_diff;
  }

  dev_cells[cell_id].sugar -= dev_cells[cell_id].sugar_diffusion;
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
        new_sugar += 0.125f * dev_cells[n_id].sugar_diffusion;
      }
    }
  }

  dev_cells[cell_id].sugar += new_sugar;
}


__device__ void Cell_decide_permission(int cell_id) {
  int selected_agent = kNullptr;
  int turn = 0;
  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        if (dev_cells[n_id].Agent_type != kNoType
            && dev_cells[n_id].Agent_cell_request == cell_id) {
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
    Agent_give_permission(selected_agent);
  }
}


__device__ void Cell_grow_sugar(int cell_id) {
  dev_cells[cell_id].sugar += min(
      dev_cells[cell_id].sugar_capacity - dev_cells[cell_id].sugar,
      dev_cells[cell_id].grow_rate);
}


__device__ void Male_propose(int cell_id) {
  if (Agent_ready_to_mate(cell_id)) {
    // Propose to female with highest endowment.
    int target_agent = kNullptr;
    int target_sugar = -1;

    int this_x = cell_id % kSize;
    int this_y = cell_id / kSize;

    for (int dx = -dev_cells[cell_id].Agent_vision;
         dx < dev_cells[cell_id].Agent_vision + 1; ++dx) {
      for (int dy = -dev_cells[cell_id].Agent_vision;
           dy < dev_cells[cell_id].Agent_vision + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;

          if (dev_cells[n_id].Agent_type == kClassFemale
              && Agent_ready_to_mate(n_id)) {
            if (dev_cells[n_id].Agent_sugar > target_sugar) {
              target_agent = n_id;
              target_sugar = dev_cells[n_id].Agent_sugar;
            }
          }
        }
      }
    }

    assert((target_sugar == -1) == (target_agent == kNullptr));
    dev_cells[cell_id].Male_female_request = target_agent;
  }
}


__device__ void Male_propose_offspring_target(int cell_id) {
  if (dev_cells[cell_id].Male_proposal_accepted) {
    assert(dev_cells[cell_id].Male_female_request != kNullptr);

    // Select a random cell.
    int target_cell = kNullptr;
    int turn = 0;

    int this_x = cell_id % kSize;
    int this_y = cell_id / kSize;

    for (int dx = -dev_cells[cell_id].Agent_vision;
         dx < dev_cells[cell_id].Agent_vision + 1; ++dx) {
      for (int dy = -dev_cells[cell_id].Agent_vision;
           dy < dev_cells[cell_id].Agent_vision + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;

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
    dev_cells[cell_id].Agent_cell_request = target_cell;
  }
}


__device__ void Male_mate(int cell_id) {
  if (dev_cells[cell_id].Male_proposal_accepted
      && dev_cells[cell_id].Agent_permission) {
    assert(dev_cells[cell_id].Male_female_request != kNullptr);
    assert(dev_cells[cell_id].Agent_cell_request != kNullptr);

    // Register birth.
    ++dev_cells[dev_cells[cell_id].Male_female_request].Female_num_children;

    // Take sugar from endowment.
    int c_endowment = (dev_cells[cell_id].Agent_endowment
        + dev_cells[dev_cells[cell_id].Male_female_request].Agent_endowment) / 2;
    dev_cells[cell_id].Agent_sugar -= dev_cells[cell_id].Agent_endowment / 2;
    dev_cells[dev_cells[cell_id].Male_female_request].Agent_sugar
        -= dev_cells[dev_cells[cell_id].Male_female_request].Agent_endowment / 2;

    // Calculate other properties.
    int c_vision = (dev_cells[cell_id].Agent_vision
        + dev_cells[dev_cells[cell_id].Male_female_request].Agent_vision) / 2;
    int c_max_age = (dev_cells[cell_id].Agent_max_age
        + dev_cells[dev_cells[cell_id].Male_female_request].Agent_max_age) / 2;
    int c_metabolism = (dev_cells[cell_id].Agent_metabolism
        + dev_cells[dev_cells[cell_id].Male_female_request].Agent_metabolism) / 2;
    int c_max_children =
        dev_cells[dev_cells[cell_id].Male_female_request].Female_max_children;

    // Create agent.
    assert(dev_cells[cell_id].Agent_cell_request != kNullptr);
    assert(dev_cells[dev_cells[cell_id].Agent_cell_request].Agent_type == kNoType);

    if (Agent_random_float(cell_id) <= 0.5f) {
      new_Male(dev_cells[cell_id].Agent_cell_request,
               c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    } else {
      new_Female(dev_cells[cell_id].Agent_cell_request,
                 c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism,
                 c_max_children);
    }

    // No Cell::enter necessary.
  }

  dev_cells[cell_id].Agent_permission = false;
  dev_cells[cell_id].Male_proposal_accepted = false;
  dev_cells[cell_id].Male_female_request = kNullptr;
  dev_cells[cell_id].Agent_cell_request = kNullptr;
}


__device__ void Female_decide_proposal(int cell_id) {
  if (dev_cells[cell_id].Female_num_children
      < dev_cells[cell_id].Female_max_children) {
    int selected_agent = kNullptr;
    int selected_sugar = -1;
    int this_x = cell_id % kSize;
    int this_y = cell_id / kSize;

    for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
      for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;

          if (dev_cells[n_id].Agent_type == kClassMale) {
            if (dev_cells[n_id].Male_female_request == cell_id
                && dev_cells[n_id].Agent_sugar > selected_sugar) {
              selected_agent = n_id;
              selected_sugar = dev_cells[n_id].Agent_sugar;
            }
          }
        }
      }
    }

    assert((selected_sugar == -1) == (selected_agent == kNullptr));

    if (selected_agent != kNullptr) {
      dev_cells[selected_agent].Male_proposal_accepted = true;
    }
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


__global__ void kernel_Agent_age_and_metabolize() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type != kNoType) {
      Agent_age_and_metabolize(i);
    }
  }
}


__global__ void kernel_Agent_prepare_move() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type != kNoType) {
      Agent_prepare_move(i);
    }
  }
}


__global__ void kernel_Cell_decide_permission() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    Cell_decide_permission(i);
  }
}


__global__ void kernel_Agent_update_move() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type != kNoType) {
      Agent_update_move(i);
    }
  }
}


__global__ void kernel_Agent_harvest_sugar() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type != kNoType) {
      // Must be in a separate kernel to avoid race condition.
      // (Old and new cell could both be processed.)
      Agent_harvest_sugar(i);
    }
  }
}


__global__ void kernel_Male_propose() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type == kClassMale) {
      Male_propose(i);
    }
  }
}


__global__ void kernel_Female_decide_proposal() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type == kClassFemale) {
      Female_decide_proposal(i);
    }
  }
}


__global__ void kernel_Male_propose_offspring_target() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type == kClassMale) {
      Male_propose_offspring_target(i);
    }
  }
}


__global__ void kernel_Male_mate() {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].Agent_type == kClassMale) {
      Male_mate(i);
    }
  }
}


void step() {
  kernel_Cell_grow_sugar<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_prepare_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_update_diffuse<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_age_and_metabolize<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_prepare_move<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_decide_permission<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_update_move<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_harvest_sugar<<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

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

    if (r < kProbMale) {
      // Create male agent.
      new_Male(i, c_vision, /*age=*/ 0, c_max_age, c_endowment, c_metabolism);
    } else if (r < kProbMale + kProbFemale) {
      // Create female agent.
      new_Female(i, c_vision, /*age=*/ 0, c_max_age, c_endowment,
                 c_metabolism, c_max_children);
    }   // else: Do not create agent.
  }
}


void initialize_simulation() {
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  create_agents<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


__global__ void kernel_copy_checksum_data() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kSize*kSize; i += blockDim.x * gridDim.x) {
    dev_Cell_sugar[i] = dev_cells[i].sugar;
    dev_Cell_Agent_type[i] = dev_cells[i].Agent_type;
  }
}


int data_Cell_sugar[kSize*kSize];
char data_Cell_agent_types[kSize*kSize];
int checksum(int* host_Cell_sugar, char* host_Cell_Agent_type) {
  kernel_copy_checksum_data<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(data_Cell_sugar, host_Cell_sugar, sizeof(int)*kSize*kSize,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data_Cell_agent_types, host_Cell_Agent_type,
             sizeof(char)*kSize*kSize, cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  int result = 0;
  for (int i = 0; i < kSize*kSize; ++i) {
    result += data_Cell_agent_types[i]; //(data_Cell_sugar[i] * i) % 1234567;
  }
  return result;
}


int main(int /*argc*/, char** /*argv*/) {
  // Allocate memory.
  int* host_Cell_sugar;
  cudaMalloc(&host_Cell_sugar, sizeof(int)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_sugar, &host_Cell_sugar,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  char* host_Cell_Agent_type;
  cudaMalloc(&host_Cell_Agent_type, sizeof(char)*kSize*kSize);
  cudaMemcpyToSymbol(dev_Cell_Agent_type, &host_Cell_Agent_type,
                     sizeof(char*), 0, cudaMemcpyHostToDevice);

  Cell* host_cells;
  cudaMalloc(&host_cells, sizeof(Cell)*kSize*kSize);
  cudaMemcpyToSymbol(dev_cells, &host_cells,
                     sizeof(Cell*), 0, cudaMemcpyHostToDevice);

  gpuErrchk(cudaDeviceSynchronize());

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
  printf("Checksum: %i\n", checksum(host_Cell_sugar, host_Cell_Agent_type));
#endif  // NDEBUG

  printf("%lu\n", micros);

  return 0;

  // TODO: Free CUDA memory.
}
