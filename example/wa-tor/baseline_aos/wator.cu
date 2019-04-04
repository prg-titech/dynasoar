#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <limits>

#include "../configuration.h"
#include "../dynasoar_no_cell/wator.h"


static const int kNumBlockSize = 256;
static const char kAgentTypeNone = 0;
static const char kAgentTypeFish = 1;
static const char kAgentTypeShark = 2;

using IndexT = int;

struct Cell {
  curandState_t random_state;
  curandState_t agent_random_state;
  DeviceArray<bool, 5> neighbor_request;
  IndexT agent_new_position;
  uint32_t agent_egg_counter;
  uint32_t agent_energy;
  bool agent_active;
  char agent_type;
};

// Array of structure.
__device__ Cell* dev_cells;


__device__ void Cell_prepare(IndexT cell_id) {
  for (int i = 0; i < 5; ++i) {
    dev_cells[cell_id].neighbor_request[i] = false;
  }
}


__device__ IndexT Cell_neighbor(IndexT cell_id, uint8_t nid) {
  int x, y;
  int self_x = cell_id % kSizeX;
  int self_y = cell_id / kSizeX;

  if (nid == 0) {
    // left
    x = self_x == 0 ? kSizeX - 1 : self_x - 1;
    y = self_y;
  } else if (nid == 1) {
    // top
    x = self_x;
    y = self_y == 0 ? kSizeY - 1 : self_y - 1;
  } else if (nid == 2) {
    // right
    x = self_x == kSizeX - 1 ? 0 : self_x + 1;
    y = self_y;
  } else if (nid == 3) {
    // bottom
    x = self_x;
    y = self_y == kSizeY - 1 ? 0 : self_y + 1;
  } else {
    assert(false);
  }

  return y*kSizeX + x;
}


__device__ void new_Cell(IndexT cell_id) {
  dev_cells[cell_id].agent_type = kAgentTypeNone;
  dev_cells[cell_id].agent_active = false;
  curand_init(kSeed, cell_id, 0, &dev_cells[cell_id].random_state);
  Cell_prepare(cell_id);
}


template<bool(*predicate)(IndexT)>
__device__ bool Cell_request_random_neighbor(
    IndexT cell_id, curandState_t& random_state) {
  uint8_t candidates[4];
  uint8_t num_candidates = 0;

  for (int i = 0; i < 4; ++i) {
    if (predicate(Cell_neighbor(cell_id, i))) {
      candidates[num_candidates++] = i;
    }
  }

  if (num_candidates == 0) {
    return false;
  } else {
    uint32_t selected_index = curand(&random_state) % num_candidates;
    uint8_t selected = candidates[selected_index];
    uint8_t neighbor_index = (selected + 2) % 4;
    dev_cells[Cell_neighbor(cell_id, selected)].neighbor_request[neighbor_index] = true;

    // Check correctness of neighbor calculation.
    assert(Cell_neighbor(Cell_neighbor(cell_id, selected), neighbor_index) == cell_id);

    return true;
  }
}


__device__ void Cell_decide(IndexT cell_id) {
  if (dev_cells[cell_id].neighbor_request[4]) {
    // This cell has priority.
    dev_cells[cell_id].agent_new_position = cell_id;
  } else {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if (dev_cells[cell_id].neighbor_request[i]) {
        candidates[num_candidates++] = i;
      }
    }

    if (num_candidates > 0) {
      assert(dev_cells[cell_id].agent_type != kAgentTypeShark);
      uint32_t selected_index = curand(&dev_cells[cell_id].random_state) % num_candidates;
      dev_cells[Cell_neighbor(cell_id, candidates[selected_index])].agent_new_position = cell_id;
    }
  }
}


__device__ void Cell_enter(IndexT cell_id, IndexT agent) {
  assert(dev_cells[cell_id].agent_type == kAgentTypeNone);
  assert(dev_cells[agent].agent_type != kAgentTypeNone);

  // TODO: Assign agent but do not commit yet!
  dev_cells[cell_id].agent_random_state = dev_cells[agent].agent_random_state;
  dev_cells[cell_id].agent_type = dev_cells[agent].agent_type;
  dev_cells[cell_id].agent_energy = dev_cells[agent].agent_energy;
  dev_cells[cell_id].agent_egg_counter = dev_cells[agent].agent_egg_counter;
  dev_cells[cell_id].agent_new_position = dev_cells[agent].agent_new_position;
}


__device__ void Cell_kill(IndexT cell_id) {
  assert(dev_cells[cell_id].agent_type != kAgentTypeNone);
  dev_cells[cell_id].agent_type = kAgentTypeNone;
  dev_cells[cell_id].agent_active = false;
}


__device__ bool Cell_has_fish(IndexT cell_id) {
  return dev_cells[cell_id].agent_type == kAgentTypeFish;
}


__device__ bool Cell_has_shark(IndexT cell_id) {
  return dev_cells[cell_id].agent_type == kAgentTypeShark;
}


__device__ bool Cell_is_free(IndexT cell_id) {
  return dev_cells[cell_id].agent_type == kAgentTypeNone;
}


__device__ void Cell_leave(IndexT cell_id) {
  assert(dev_cells[cell_id].agent_type != kAgentTypeNone);
  dev_cells[cell_id].agent_type = kAgentTypeNone;
  dev_cells[cell_id].agent_active = false;
}


__device__ void Cell_request_random_fish_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_has_fish>(
      cell_id, dev_cells[cell_id].agent_random_state)) {
    // No fish found. Look for free cell.
    if (!Cell_request_random_neighbor<&Cell_is_free>(
        cell_id, dev_cells[cell_id].agent_random_state)) {
      dev_cells[cell_id].neighbor_request[4] = true;
    }
  }
}


__device__ void Cell_request_random_free_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_is_free>(
      cell_id, dev_cells[cell_id].agent_random_state)) {
    dev_cells[cell_id].neighbor_request[4] = true;
  }
}


__device__ void new_Agent(int cell_id, int seed) {
  curand_init(seed, 0, 0, &dev_cells[cell_id].agent_random_state);
  dev_cells[cell_id].agent_active = false;
}


__device__ void new_Fish(int cell_id, int seed) {
  new_Agent(cell_id, seed);
  dev_cells[cell_id].agent_type = kAgentTypeFish;
  dev_cells[cell_id].agent_egg_counter = seed % kSpawnThreshold;
}


__device__ void Fish_prepare(int cell_id) {
  dev_cells[cell_id].agent_egg_counter++;

  // Fallback: Stay on current cell.
  dev_cells[cell_id].agent_new_position = cell_id;

  Cell_request_random_free_neighbor(cell_id);
}


__device__ void Fish_update(int cell_id) {
  auto new_pos = dev_cells[cell_id].agent_new_position;
  if (cell_id != new_pos) {
    Cell_enter(new_pos, cell_id);
    Cell_leave(cell_id);

    if (kOptionFishSpawn && dev_cells[new_pos].agent_egg_counter > kSpawnThreshold) {
      new_Fish(cell_id, curand(&dev_cells[new_pos].agent_random_state));
      dev_cells[new_pos].agent_egg_counter = 0;
    }
  }
}


__device__ void new_Shark(int cell_id, int seed) {
  new_Agent(cell_id, seed);
  dev_cells[cell_id].agent_type = kAgentTypeShark;
  dev_cells[cell_id].agent_energy = kEngeryStart;
  dev_cells[cell_id].agent_egg_counter = seed % kSpawnThreshold;
}


__device__ void Shark_prepare(int cell_id) {
  dev_cells[cell_id].agent_egg_counter++;
  dev_cells[cell_id].agent_energy--;

  if (kOptionSharkDie && dev_cells[cell_id].agent_energy == 0) {
    // Do nothing. Shark will die.
  } else {
    // Fallback: Stay on current cell.
    dev_cells[cell_id].agent_new_position = cell_id;
    Cell_request_random_fish_neighbor(cell_id);
  }
}


__device__ void Shark_update(int cell_id) {
  auto new_pos = dev_cells[cell_id].agent_new_position;

  if (kOptionSharkDie && dev_cells[cell_id].agent_energy == 0) {
    Cell_kill(cell_id);
  } else {
    if (cell_id != new_pos) {
      if (Cell_has_fish(new_pos)) {
        dev_cells[cell_id].agent_energy += kEngeryBoost;
        Cell_kill(new_pos);
      }

      assert(dev_cells[new_pos].agent_type != kAgentTypeFish);
      assert(dev_cells[new_pos].agent_type == kAgentTypeNone);
      Cell_enter(new_pos, cell_id);
      Cell_leave(cell_id);

      if (kOptionSharkSpawn && dev_cells[new_pos].agent_egg_counter > kSpawnThreshold) {
        new_Shark(cell_id, curand(&dev_cells[new_pos].agent_random_state));
        dev_cells[new_pos].agent_egg_counter = 0;
      }
    }
  }
}


// ----- KERNELS -----

__device__ int d_checksum;


__device__ void Cell_add_to_checksum(IndexT cell_id) {
  if (Cell_has_fish(cell_id)) {
    atomicAdd(&d_checksum, 3);
  } else if (Cell_has_shark(cell_id)) {
    atomicAdd(&d_checksum, 7);
  }
}


__global__ void reset_checksum() {
  d_checksum = 0;
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    new_Cell(i);
  }
}


__global__ void setup_cells() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    // Initialize with random agent.
    auto& rand_state = dev_cells[i].random_state;
    uint32_t agent_type = curand(&rand_state) % 4;
    if (agent_type == 0) {
      new_Fish(i, curand(&rand_state));
    } else if (agent_type == 1) {
      new_Shark(i, curand(&rand_state));
    } else {
      // Free cell.
    }
  }
}


__global__ void kernel_Cell_add_to_checksum() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    Cell_add_to_checksum(i);
  }
}


__global__ void kernel_Cell_prepare() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    Cell_prepare(i);
  }
}


__global__ void kernel_Cell_decide() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    Cell_decide(i);
  }
}


__global__ void kernel_Agent_set_active() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].agent_type != kAgentTypeNone) {
      dev_cells[i].agent_active = true;
    }
  }
}


__global__ void kernel_Fish_prepare() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].agent_type == kAgentTypeFish && dev_cells[i].agent_active) {
      Fish_prepare(i);
    }
  }
}


__global__ void kernel_Fish_update() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].agent_type == kAgentTypeFish && dev_cells[i].agent_active) {
      Fish_update(i);
    }
  }
}


__global__ void kernel_Shark_prepare() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].agent_type == kAgentTypeShark && dev_cells[i].agent_active) {
      Shark_prepare(i);
    }
  }
}


__global__ void kernel_Shark_update() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].agent_type == kAgentTypeShark && dev_cells[i].agent_active) {
      Shark_update(i);
    }
  }
}


__global__ void print_checksum() {
  printf("%i,%u,%u,%u,%u\n",
         d_checksum, 0, 0, 0, 0);
}


void print_stats() {
  reset_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_add_to_checksum<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());;

  print_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void step() {
  // --- FISH ---
  kernel_Cell_prepare<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Fish_prepare<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_decide<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Fish_update<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_set_active<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // --- SHARKS ---
  kernel_Cell_prepare<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Shark_prepare<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Cell_decide<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Shark_update<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_set_active<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void initialize() {
  Cell* h_cells;
  cudaMalloc(&h_cells, sizeof(Cell)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_cells, &h_cells,
                     sizeof(Cell*), 0, cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
  setup_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Agent_set_active<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


int main(int /*argc*/, char*[] /*arvg[]*/) {
  initialize();

  int total_time = 0;

  for (int i = 0; i < kNumIterations; ++i) {
#ifndef NDEBUG
    printf("%i\n", i);
    print_stats();
#endif  // NDEBUG

    auto time_before = std::chrono::high_resolution_clock::now();
    step();

    auto time_after = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::microseconds>(
        time_after - time_before).count();
  }

#ifndef NDEBUG
  print_stats();
#endif  // NDEBUG

  // Print total running time
  printf("%i\n", total_time);

  return 0;
}
