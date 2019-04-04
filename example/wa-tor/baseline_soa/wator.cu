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

__device__ curandState_t* dev_Cell_random_state;
__device__ curandState_t* dev_Agent_random_state;
__device__ DeviceArray<bool, 5>* dev_Cell_neighbor_request;
__device__ IndexT* dev_Agent_new_position;
__device__ uint32_t* dev_Agent_egg_counter;
__device__ uint32_t* dev_Agent_energy;
__device__ bool* dev_Agent_active;
__device__ char* dev_Agent_type;


__device__ void Cell_prepare(IndexT cell_id) {
  for (int i = 0; i < 5; ++i) {
    dev_Cell_neighbor_request[cell_id][i] = false;
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
  dev_Agent_type[cell_id] = kAgentTypeNone;
  dev_Agent_active[cell_id] = false;
  curand_init(kSeed, cell_id, 0, &dev_Cell_random_state[cell_id]);
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
    dev_Cell_neighbor_request[Cell_neighbor(cell_id, selected)][neighbor_index] = true;

    // Check correctness of neighbor calculation.
    assert(Cell_neighbor(Cell_neighbor(cell_id, selected), neighbor_index) == cell_id);

    return true;
  }
}


__device__ void Cell_decide(IndexT cell_id) {
  if (dev_Cell_neighbor_request[cell_id][4]) {
    // This cell has priority.
    dev_Agent_new_position[cell_id] = cell_id;
  } else {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if (dev_Cell_neighbor_request[cell_id][i]) {
        candidates[num_candidates++] = i;
      }
    }

    if (num_candidates > 0) {
      assert(dev_Agent_type[cell_id] != kAgentTypeShark);
      uint32_t selected_index = curand(&dev_Cell_random_state[cell_id]) % num_candidates;
      dev_Agent_new_position[Cell_neighbor(cell_id, candidates[selected_index])] = cell_id;
    }
  }
}


__device__ void Cell_enter(IndexT cell_id, IndexT agent) {
  assert(dev_Agent_type[cell_id] == kAgentTypeNone);
  assert(dev_Agent_type[agent] != kAgentTypeNone);

  // TODO: Assign agent but do not commit yet!
  dev_Agent_random_state[cell_id] = dev_Agent_random_state[agent];
  dev_Agent_type[cell_id] = dev_Agent_type[agent];
  dev_Agent_energy[cell_id] = dev_Agent_energy[agent];
  dev_Agent_egg_counter[cell_id] = dev_Agent_egg_counter[agent];
  dev_Agent_new_position[cell_id] = dev_Agent_new_position[agent];
}


__device__ void Cell_kill(IndexT cell_id) {
  assert(dev_Agent_type[cell_id] != kAgentTypeNone);
  dev_Agent_type[cell_id] = kAgentTypeNone;
  dev_Agent_active[cell_id] = false;
}


__device__ bool Cell_has_fish(IndexT cell_id) {
  return dev_Agent_type[cell_id] == kAgentTypeFish;
}


__device__ bool Cell_has_shark(IndexT cell_id) {
  return dev_Agent_type[cell_id] == kAgentTypeShark;
}


__device__ bool Cell_is_free(IndexT cell_id) {
  return dev_Agent_type[cell_id] == kAgentTypeNone;
}


__device__ void Cell_leave(IndexT cell_id) {
  assert(dev_Agent_type[cell_id] != kAgentTypeNone);
  dev_Agent_type[cell_id] = kAgentTypeNone;
  dev_Agent_active[cell_id] = false;
}


__device__ void Cell_request_random_fish_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_has_fish>(
      cell_id, dev_Agent_random_state[cell_id])) {
    // No fish found. Look for free cell.
    if (!Cell_request_random_neighbor<&Cell_is_free>(
        cell_id, dev_Agent_random_state[cell_id])) {
      dev_Cell_neighbor_request[cell_id][4] = true;
    }
  }
}


__device__ void Cell_request_random_free_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_is_free>(
      cell_id, dev_Agent_random_state[cell_id])) {
    dev_Cell_neighbor_request[cell_id][4] = true;
  }
}


__device__ void new_Agent(int cell_id, int seed) {
  curand_init(seed, 0, 0, &dev_Agent_random_state[cell_id]);
  dev_Agent_active[cell_id] = false;
}


__device__ void new_Fish(int cell_id, int seed) {
  new_Agent(cell_id, seed);
  dev_Agent_type[cell_id] = kAgentTypeFish;
  dev_Agent_egg_counter[cell_id] = seed % kSpawnThreshold;
}


__device__ void Fish_prepare(int cell_id) {
  dev_Agent_egg_counter[cell_id]++;

  // Fallback: Stay on current cell.
  dev_Agent_new_position[cell_id] = cell_id;

  Cell_request_random_free_neighbor(cell_id);
}


__device__ void Fish_update(int cell_id) {
  auto new_pos = dev_Agent_new_position[cell_id];
  if (cell_id != new_pos) {
    Cell_enter(new_pos, cell_id);
    Cell_leave(cell_id);

    if (kOptionFishSpawn && dev_Agent_egg_counter[new_pos] > kSpawnThreshold) {
      new_Fish(cell_id, curand(&dev_Agent_random_state[new_pos]));
      dev_Agent_egg_counter[new_pos] = 0;
    }
  }
}


__device__ void new_Shark(int cell_id, int seed) {
  new_Agent(cell_id, seed);
  dev_Agent_type[cell_id] = kAgentTypeShark;
  dev_Agent_energy[cell_id] = kEngeryStart;
  dev_Agent_egg_counter[cell_id] = seed % kSpawnThreshold;
}


__device__ void Shark_prepare(int cell_id) {
  dev_Agent_egg_counter[cell_id]++;
  dev_Agent_energy[cell_id]--;

  if (kOptionSharkDie && dev_Agent_energy[cell_id] == 0) {
    // Do nothing. Shark will die.
  } else {
    // Fallback: Stay on current cell.
    dev_Agent_new_position[cell_id] = cell_id;
    Cell_request_random_fish_neighbor(cell_id);
  }
}


__device__ void Shark_update(int cell_id) {
  auto new_pos = dev_Agent_new_position[cell_id];

  if (kOptionSharkDie && dev_Agent_energy[cell_id] == 0) {
    Cell_kill(cell_id);
  } else {
    if (cell_id != new_pos) {
      if (Cell_has_fish(new_pos)) {
        dev_Agent_energy[cell_id] += kEngeryBoost;
        Cell_kill(new_pos);
      }

      assert(dev_Agent_type[new_pos] != kAgentTypeFish);
      assert(dev_Agent_type[new_pos] == kAgentTypeNone);
      Cell_enter(new_pos, cell_id);
      Cell_leave(cell_id);

      if (kOptionSharkSpawn && dev_Agent_egg_counter[new_pos] > kSpawnThreshold) {
        new_Shark(cell_id, curand(&dev_Agent_random_state[new_pos]));
        dev_Agent_egg_counter[new_pos] = 0;
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
    auto& rand_state = dev_Cell_random_state[i];
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
    if (dev_Agent_type[i] != kAgentTypeNone) {
      dev_Agent_active[i] = true;
    }
  }
}


__global__ void kernel_Fish_prepare() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_Agent_type[i] == kAgentTypeFish && dev_Agent_active[i]) {
      Fish_prepare(i);
    }
  }
}


__global__ void kernel_Fish_update() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_Agent_type[i] == kAgentTypeFish && dev_Agent_active[i]) {
      Fish_update(i);
    }
  }
}


__global__ void kernel_Shark_prepare() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_Agent_type[i] == kAgentTypeShark && dev_Agent_active[i]) {
      Shark_prepare(i);
    }
  }
}


__global__ void kernel_Shark_update() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    if (dev_Agent_type[i] == kAgentTypeShark && dev_Agent_active[i]) {
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
  curandState_t* h_Cell_random_state;
  cudaMalloc(&h_Cell_random_state, sizeof(curandState_t)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Cell_random_state, &h_Cell_random_state,
                     sizeof(curandState_t*), 0, cudaMemcpyHostToDevice);

  curandState_t* h_Agent_random_state;
  cudaMalloc(&h_Agent_random_state, sizeof(curandState_t)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_random_state, &h_Agent_random_state,
                     sizeof(curandState_t*), 0, cudaMemcpyHostToDevice);

  DeviceArray<bool, 5>* h_Cell_neighbor_request;
  cudaMalloc(&h_Cell_neighbor_request, sizeof(DeviceArray<bool, 5>)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Cell_neighbor_request, &h_Cell_neighbor_request,
                     sizeof(DeviceArray<bool, 5>*), 0, cudaMemcpyHostToDevice);

  IndexT* h_Agent_new_position;
  cudaMalloc(&h_Agent_new_position, sizeof(IndexT)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_new_position, &h_Agent_new_position,
                     sizeof(IndexT*), 0, cudaMemcpyHostToDevice);

  uint32_t* h_Agent_egg_counter;
  cudaMalloc(&h_Agent_egg_counter, sizeof(uint32_t)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_egg_counter, &h_Agent_egg_counter,
                     sizeof(uint32_t*), 0, cudaMemcpyHostToDevice);

  uint32_t* h_Agent_energy;
  cudaMalloc(&h_Agent_energy, sizeof(uint32_t)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_energy, &h_Agent_energy,
                     sizeof(uint32_t*), 0, cudaMemcpyHostToDevice);

  bool* h_Agent_active;
  cudaMalloc(&h_Agent_active, sizeof(bool)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_active, &h_Agent_active,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);

  char* h_Agent_type;
  cudaMalloc(&h_Agent_type, sizeof(char)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Agent_type, &h_Agent_type,
                     sizeof(char*), 0, cudaMemcpyHostToDevice);

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
