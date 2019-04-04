#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

#include "../configuration.h"
#include "wator.h"


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ Cell::Cell(int cell_id) : agent_(nullptr) {
  curand_init(kSeed, cell_id, 0, &random_state_);
  prepare();
}


__device__ Agent* Cell::agent() const { return agent_; }


__device__ void Cell::decide() {
  if (neighbor_request_[4]) {
    // This cell has priority.
    agent_->set_new_position(this);
  } else {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if (neighbor_request_[i]) {
        candidates[num_candidates++] = i;
      }
    }

    if (num_candidates > 0) {
      uint32_t selected_index = curand(&random_state_) % num_candidates;
      neighbors_[candidates[selected_index]]->agent()->set_new_position(this);
    }
  }
}


__device__ void Cell::enter(Agent* agent) {
  assert(agent_ == nullptr);
  assert(agent != nullptr);

  agent_ = agent;
  agent->set_position(this);
}


__device__ bool Cell::has_fish() const {
  return agent_->cast<Fish>() != nullptr;
}


__device__ bool Cell::has_shark() const {
  return agent_->cast<Shark>() != nullptr;
}


__device__ bool Cell::is_free() const { return agent_ == nullptr; }


__device__ void Cell::leave() {
  assert(agent_ != nullptr);
  agent_ = nullptr;
}


__device__ void Cell::prepare() {
  for (int i = 0; i < 5; ++i) { neighbor_request_[i] = false; }
}


__device__ curandState_t& Cell::random_state() { return random_state_; }


__device__ void Cell::request_random_fish_neighbor() {
  if (!request_random_neighbor<&Cell::has_fish>(agent_->random_state())) {
    // No fish found. Look for free cell.
    if (!request_random_neighbor<&Cell::is_free>(agent_->random_state())) {
      neighbor_request_[4] = true;
    }
  }
}


__device__ void Cell::request_random_free_neighbor() {
  if (!request_random_neighbor<&Cell::is_free>(agent_->random_state())) {
    neighbor_request_[4] = true;
  }
}


template<bool(Cell::*predicate)() const>
__device__ bool Cell::request_random_neighbor(curandState_t& random_state) {
  uint8_t candidates[4];
  uint8_t num_candidates = 0;

  for (int i = 0; i < 4; ++i) {
    if ((neighbors_[i]->*predicate)()) {
      candidates[num_candidates++] = i;
    }
  }

  if (num_candidates == 0) {
    return false;
  } else {
    uint32_t selected_index = curand(&random_state) % num_candidates;
    uint8_t selected = candidates[selected_index];
    uint8_t neighbor_index = (selected + 2) % 4;
    neighbors_[selected]->neighbor_request_[neighbor_index] = true;

    // Check correctness of neighbor calculation.
    assert(neighbors_[selected]->neighbors_[neighbor_index] == this);

    return true;
  }
}


__device__ void Cell::set_neighbors(Cell* left, Cell* top,
                                    Cell* right, Cell* bottom) {
  neighbors_[0] = left;
  neighbors_[1] = top;
  neighbors_[2] = right;
  neighbors_[3] = bottom;
}


__device__ Agent::Agent(int seed) { curand_init(seed, 0, 0, &random_state_); }


__device__ curandState_t& Agent::random_state() { return random_state_; }


__device__ void Agent::set_new_position(Cell* new_pos) {
  // Check for race condition. (This is not bullet proof.)
  assert(new_position_ == position_);

  new_position_ = new_pos;
}


__device__ Cell* Agent::position() const { return position_; }


__device__ void Agent::set_position(Cell* cell) { position_ = cell; }


__device__ Fish::Fish(int seed)
    : Agent(seed), egg_timer_(seed % kSpawnThreshold) {}


__device__ void Fish::prepare() {
  egg_timer_++;
  // Fallback: Stay on current cell.
  new_position_ = position_;

  assert(position_ != nullptr);
  position_->request_random_free_neighbor();
}


__device__ void Fish::update() {
  Cell* old_position = position_;

  if (old_position != new_position_) {
    old_position->leave();
    new_position_->enter(this);

    if (kOptionFishSpawn && egg_timer_ > kSpawnThreshold) {
      auto* new_fish = new(device_allocator) Fish(curand(&random_state_));
      assert(new_fish != nullptr);
      old_position->enter(new_fish);
      egg_timer_ = (uint32_t) 0;
    }
  }
}


__device__ Shark::Shark(int seed)
    : Agent(seed), energy_(kEngeryStart), egg_timer_(seed % kSpawnThreshold) {}


__device__ void Shark::prepare() {
  egg_timer_++;
  energy_--;

  assert(position_ != nullptr);
  if (kOptionSharkDie && energy_ == 0) {
    // Do nothing. Shark will die.
  } else {
    // Fallback: Stay on current cell.
    new_position_ = position_;
    position_->request_random_fish_neighbor();
  }
}


__device__ void Shark::update() {
  if (kOptionSharkDie && energy_ == 0) {
    position_->kill();
  } else {
    Cell* old_position = position_;

    if (old_position != new_position_) {
      if (new_position_->has_fish()) {
        energy_ += kEngeryBoost;
        new_position_->kill();
      }

      old_position->leave();
      new_position_->enter(this);

      if (kOptionSharkSpawn && egg_timer_ > kSpawnThreshold) {
        auto* new_shark =
            new(device_allocator) Shark(curand(&random_state_));
        assert(new_shark != nullptr);
        old_position->enter(new_shark);
        egg_timer_ = 0;
      }
    }
  }
}


__device__ void Cell::kill() {
  assert(agent_ != nullptr);
  //device_allocator->free<Agent>(agent_);
  destroy(device_allocator, agent_);
  agent_ = nullptr;
}


// ----- KERNELS -----

__device__ Cell* cells[kSizeX * kSizeY];  // Only for setup.
__device__ int d_checksum;

__device__ void Cell::add_to_checksum() {
  if (has_fish()) {
    atomicAdd(&d_checksum, 3);
  } else if (has_shark()) {
    atomicAdd(&d_checksum, 7);
  }
}


__global__ void reset_checksum() {
  d_checksum = 0;
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    Cell* new_cell = new(device_allocator) Cell(i);
    assert(new_cell != nullptr);
    cells[i] = new_cell;
  }
}


__global__ void setup_cells() {
  for (int i = threadIdx.x + blockDim.x*blockIdx.x;
       i < kSizeX*kSizeY; i += blockDim.x * gridDim.x) {
    int x = i % kSizeX;
    int y = i / kSizeX;

    Cell* left = x > 0 ? cells[y*kSizeX + x - 1]
                       : cells[y*kSizeX + kSizeX - 1];
    Cell* right = x < kSizeX - 1 ? cells[y*kSizeX + x + 1]
                                      : cells[y*kSizeX];
    Cell* top = y > 0 ? cells[(y - 1)*kSizeX + x]
                      : cells[(kSizeY - 1)*kSizeX + x];
    Cell* bottom = y < kSizeY - 1 ? cells[(y + 1)*kSizeX + x]
                                       : cells[x];

    // left, top, right, bottom
    cells[i]->set_neighbors(left, top, right, bottom);

    // Initialize with random agent.
    auto& rand_state = cells[i]->random_state();
    uint32_t agent_type = curand(&rand_state) % 4;
    if (agent_type == 0) {
      auto* agent = new(device_allocator) Fish(curand(&rand_state));
      assert(agent != nullptr);
      cells[i]->enter(agent);
    } else if (agent_type == 1) {
      auto* agent = new(device_allocator) Shark(curand(&rand_state));
      assert(agent != nullptr);
      cells[i]->enter(agent);
    } else {
      // Free cell.
    }
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


void step() {
  // --- FISH ---
  allocator_handle->parallel_do<Cell, &Cell::prepare>();
  allocator_handle->parallel_do<Fish, &Fish::prepare>();
  allocator_handle->parallel_do<Cell, &Cell::decide>();
  allocator_handle->parallel_do<Fish, &Fish::update>();

  // --- SHARKS ---
  allocator_handle->parallel_do<Cell, &Cell::prepare>();
  allocator_handle->parallel_do<Shark, &Shark::prepare>();
  allocator_handle->parallel_do<Cell, &Cell::decide>();
  allocator_handle->parallel_do<Shark, &Shark::update>();
}


void initialize() {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
  setup_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void print_stats() {
  reset_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Cell, &Cell::add_to_checksum>();

  print_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

int main(int /*argc*/, char*[] /*arvg[]*/) {
#ifdef OPTION_RENDER
  printf("Run wator_soa_no_cell for rendering.\n");
#endif  // OPTION_RENDER

  initialize();

  int total_time = 0;

  for (int i = 0; i < kNumIterations; ++i) {
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

  return 0;
}

