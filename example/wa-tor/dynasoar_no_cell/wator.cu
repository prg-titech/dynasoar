#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <limits>

#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER

#include "wator.h"


static const int kNumBlockSize = 256;
static const int kNullptr = std::numeric_limits<IndexT>::max();

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;

__device__ curandState_t* dev_Cell_random_state;
__device__ Agent** dev_Cell_agent;
__device__ DeviceArray<bool, 5>* dev_Cell_neighbor_request;


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
  dev_Cell_agent[cell_id] = nullptr;
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
    dev_Cell_agent[cell_id]->set_new_position(cell_id);
  } else {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if (dev_Cell_neighbor_request[cell_id][i]) {
        candidates[num_candidates++] = i;
      }
    }

    if (num_candidates > 0) {
      uint32_t selected_index = curand(&dev_Cell_random_state[cell_id]) % num_candidates;
      dev_Cell_agent[Cell_neighbor(cell_id, candidates[selected_index])]
          ->set_new_position(cell_id);
    }
  }
}


__device__ void Cell_enter(IndexT cell_id, Agent* agent) {
  assert(dev_Cell_agent[cell_id] == nullptr);
  assert(agent != nullptr);

  dev_Cell_agent[cell_id] = agent;
  agent->set_position(cell_id);
}


__device__ void Cell_kill(IndexT cell_id) {
  assert(dev_Cell_agent[cell_id] != nullptr);
  destroy(device_allocator, dev_Cell_agent[cell_id]);
  dev_Cell_agent[cell_id] = nullptr;
}


__device__ bool Cell_has_fish(IndexT cell_id) {
  return dev_Cell_agent[cell_id]->cast<Fish>() != nullptr;
}


__device__ bool Cell_has_shark(IndexT cell_id) {
  return dev_Cell_agent[cell_id]->cast<Shark>() != nullptr;
}


__device__ bool Cell_is_free(IndexT cell_id) {
  return dev_Cell_agent[cell_id] == nullptr;
}


__device__ void Cell_leave(IndexT cell_id) {
  assert(dev_Cell_agent[cell_id] != nullptr);
  dev_Cell_agent[cell_id] = nullptr;
}


__device__ void Cell_request_random_fish_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_has_fish>(
      cell_id, dev_Cell_agent[cell_id]->random_state())) {
    // No fish found. Look for free cell.
    if (!Cell_request_random_neighbor<&Cell_is_free>(
        cell_id, dev_Cell_agent[cell_id]->random_state())) {
      dev_Cell_neighbor_request[cell_id][4] = true;
    }
  }
}


__device__ void Cell_request_random_free_neighbor(IndexT cell_id) {
  if (!Cell_request_random_neighbor<&Cell_is_free>(
      cell_id, dev_Cell_agent[cell_id]->random_state())) {
    dev_Cell_neighbor_request[cell_id][4] = true;
  }
}


__device__ Agent::Agent(int seed) { curand_init(seed, 0, 0, &random_state_); }


__device__ curandState_t& Agent::random_state() { return random_state_; }


__device__ void Agent::set_new_position(IndexT new_pos) {
  // Check for race condition. (This is not bullet proof.)
  assert(new_position_ == position_);

  new_position_ = new_pos;
}


__device__ IndexT Agent::position() const { return position_; }


__device__ void Agent::set_position(IndexT cell) { position_ = cell; }


__device__ Fish::Fish(int seed)
    : Agent(seed), egg_timer_(seed % kSpawnThreshold) {}


__device__ void Fish::prepare() {
  egg_timer_++;
  // Fallback: Stay on current cell.
  new_position_ = position_;

  assert(position_ != kNullptr);
  Cell_request_random_free_neighbor(position_);
}


__device__ void Fish::update() {
  IndexT old_position = position_;

  if (old_position != new_position_) {
    Cell_leave(old_position);
    Cell_enter(new_position_, this);

    if (kOptionFishSpawn && egg_timer_ > kSpawnThreshold) {
      auto* new_fish = new(device_allocator) Fish(curand(&random_state_));
      assert(new_fish != nullptr);
      Cell_enter(old_position, new_fish);
      egg_timer_ = (uint32_t) 0;
    }
  }
}


__device__ Shark::Shark(int seed)
    : Agent(seed), energy_(kEngeryStart), egg_timer_(seed % kSpawnThreshold) {}

__device__ void Shark::prepare() {
  egg_timer_++;
  energy_--;

  assert(position_ != kNullptr);
  if (kOptionSharkDie && energy_ == 0) {
    // Do nothing. Shark will die.
  } else {
    // Fallback: Stay on current cell.
    new_position_ = position_;
    Cell_request_random_fish_neighbor(position_);
  }
}


__device__ void Shark::update() {
  if (kOptionSharkDie && energy_ == 0) {
    Cell_kill(position_);
  } else {
    IndexT old_position = position_;

    if (old_position != new_position_) {
      if (Cell_has_fish(new_position_)) {
        energy_ += kEngeryBoost;
        Cell_kill(new_position_);
      }

      Cell_leave(old_position);
      Cell_enter(new_position_, this);

      if (kOptionSharkSpawn && egg_timer_ > kSpawnThreshold) {
        auto* new_shark =
            new(device_allocator) Shark(curand(&random_state_));
        assert(new_shark != nullptr);
        Cell_enter(old_position, new_shark);
        egg_timer_ = 0;
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
      auto* agent = new(device_allocator) Fish(curand(&rand_state));
      assert(agent != nullptr);
      Cell_enter(i, agent);
    } else if (agent_type == 1) {
      auto* agent = new(device_allocator) Shark(curand(&rand_state));
      assert(agent != nullptr);
      Cell_enter(i, agent);
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


__global__ void print_checksum() {
  uint32_t fish_use = device_allocator->DBG_used_slots<Fish>();
  uint32_t fish_num = device_allocator->DBG_allocated_slots<Fish>();
  uint32_t shark_use = device_allocator->DBG_used_slots<Shark>();
  uint32_t shark_num = device_allocator->DBG_allocated_slots<Shark>();

  printf("%i,%u,%u,%u,%u\n",
         d_checksum, fish_use, fish_num, shark_use, shark_num);
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

  allocator_handle->parallel_do<Fish, &Fish::prepare>();

  kernel_Cell_decide<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Fish, &Fish::update>();

  // --- SHARKS ---
  kernel_Cell_prepare<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Shark, &Shark::prepare>();

  kernel_Cell_decide<<<
      (kSizeX*kSizeY + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Shark, &Shark::update>();
}


void initialize() {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  curandState_t* h_Cell_random_state;
  cudaMalloc(&h_Cell_random_state, sizeof(curandState_t)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Cell_random_state, &h_Cell_random_state,
                     sizeof(curandState_t*), 0, cudaMemcpyHostToDevice);

  Agent** h_Cell_agent;
  cudaMalloc(&h_Cell_agent, sizeof(Agent*)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Cell_agent, &h_Cell_agent,
                     sizeof(Agent**), 0, cudaMemcpyHostToDevice);

  DeviceArray<bool, 5>* h_Cell_neighbor_request;
  cudaMalloc(&h_Cell_neighbor_request, sizeof(DeviceArray<bool, 5>)*kSizeX*kSizeY);
  cudaMemcpyToSymbol(dev_Cell_neighbor_request, &h_Cell_neighbor_request,
                     sizeof(DeviceArray<bool, 5>*), 0, cudaMemcpyHostToDevice);

  gpuErrchk(cudaDeviceSynchronize());

  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
  setup_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


#ifdef OPTION_RENDER
__device__ char d_gui_map[kSizeY * kSizeX];
char gui_map[kSizeY * kSizeX];


__global__ void fill_gui_map() {
  for (int tid = threadIdx.x + blockDim.x*blockIdx.x;
       tid < kSizeX*kSizeY; tid += blockDim.x * gridDim.x) {
    if (dev_Cell_agent[tid] != nullptr) {
      d_gui_map[tid] = dev_Cell_agent[tid]->get_type();
    } else {
      d_gui_map[tid] = 0;
    }
  }
}


void update_gui_map() {
  fill_gui_map<<<kSizeX*kSizeY/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(gui_map, d_gui_map, sizeof(char)*kSizeX*kSizeY,
                       0, cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  // Draw pixels.
  draw(gui_map);
}
#endif  // OPTION_RENDER


int main(int /*argc*/, char*[] /*arvg[]*/) {
  initialize();

#ifdef OPTION_RENDER
    init_renderer();
#endif  // OPTION_RENDER

  int total_time = 0;

  for (int i = 0; i < kNumIterations; ++i) {
#ifdef OPTION_RENDER
      update_gui_map();
#endif  // OPTION_RENDER

#ifndef NDEBUG
      // Print debug information.
      allocator_handle->DBG_print_state_stats();
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

  // Print total running time, enumeration time.
  printf("%i, %lu\n", total_time, allocator_handle->DBG_get_enumeration_time());

#ifdef OPTION_RENDER
    close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
