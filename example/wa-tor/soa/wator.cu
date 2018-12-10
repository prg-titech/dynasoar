#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

#include "wator.h"

#define SPAWN_THRESHOLD 4
#define ENERGY_BOOST 4
#define ENERGY_START 2

#define GRID_SIZE_X 2048
#define GRID_SIZE_Y 512

#define OPTION_SHARK_DIE true
#define OPTION_SHARK_SPAWN true
#define OPTION_FISH_SPAWN true
#define OPTION_DEFRAG false
#define OPTION_PRINT_STATS false

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024

namespace wa_tor {

static const int kSeed = 42;

__device__ AllocatorT* device_allocator;

// Host side pointer.
AllocatorHandle<AllocatorT>* allocator_handle;


__global__ void DBG_stats_kernel() {
  device_allocator->DBG_print_state_stats();
}

__device__ Cell::Cell() : agent_(nullptr) {
  curand_init(kSeed, threadIdx.x + blockIdx.x * blockDim.x, 0, &random_state_);
  prepare();
}

__device__ Agent* Cell::agent() const { return agent_; }

__device__ void Cell::decide() {
  //device_allocator->device_do<Fish>(&Fish::update);

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

//#ifndef NDEBUG
//  // Ensure that no two agents are trying to enter this cell at the same time.
//  uint64_t old_val = atomicExch(reinterpret_cast<unsigned long long int*>(&agent_),
//                                reinterpret_cast<unsigned long long int>(agent));
//  assert(old_val == 0);
//#else
  agent_ = agent;
//#endif

  agent->set_position(this);
}

__device__ bool Cell::has_fish() const {
  return agent_ != nullptr && agent_->cast<Fish>() != nullptr;
}

__device__ bool Cell::has_shark() const {
  return agent_ != nullptr && agent_->cast<Shark>() != nullptr;
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
  agent_->random_state();
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
    : Agent(seed), egg_timer_(seed % SPAWN_THRESHOLD) {}

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

    if (OPTION_FISH_SPAWN && egg_timer_ > SPAWN_THRESHOLD) {
      auto* new_fish = device_allocator->make_new<Fish>(curand(&random_state_));
      assert(new_fish != nullptr);
      old_position->enter(new_fish);
      egg_timer_ = (uint32_t) 0;
    }
  }
}


__device__ Shark::Shark(int seed)
    : Agent(seed), energy_(ENERGY_START), egg_timer_(seed % SPAWN_THRESHOLD) {}

__device__ void Shark::prepare() {
  egg_timer_++;
  energy_--;

  assert(position_ != nullptr);
  if (OPTION_SHARK_DIE && energy_ == 0) {
    // Do nothing. Shark will die.
  } else {
    // Fallback: Stay on current cell.
    new_position_ = position_;
    position_->request_random_fish_neighbor();
  }
}

__device__ void Shark::update() {
  if (OPTION_SHARK_DIE && energy_ == 0) {
    position_->kill();
  } else {
    Cell* old_position = position_;

    if (old_position != new_position_) {
      if (new_position_->has_fish()) {
        energy_ += ENERGY_BOOST;
        new_position_->kill();
      }

      old_position->leave();
      new_position_->enter(this);

      if (OPTION_SHARK_SPAWN && egg_timer_ > SPAWN_THRESHOLD) {
        auto* new_shark =
            device_allocator->make_new<Shark>(curand(&random_state_));
        assert(new_shark != nullptr);
        old_position->enter(new_shark);
        egg_timer_ = 0;
      }
    }
  }
}

__device__ void Cell::kill() {
  assert(agent_ != nullptr);
  device_allocator->free<Agent>(agent_);
  agent_ = nullptr;
}


// ----- KERNELS -----

__device__ Cell* cells[GRID_SIZE_X * GRID_SIZE_Y];

__global__ void create_cells() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    Cell* new_cell = device_allocator->make_new<Cell>();
    assert(new_cell != nullptr);
    cells[tid] = new_cell;
  }
}

__global__ void setup_cells() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    int x = tid % GRID_SIZE_X;
    int y = tid / GRID_SIZE_X;

    Cell* left = x > 0 ? cells[y*GRID_SIZE_X + x - 1]
                       : cells[y*GRID_SIZE_X + GRID_SIZE_X - 1];
    Cell* right = x < GRID_SIZE_X - 1 ? cells[y*GRID_SIZE_X + x + 1]
                                      : cells[y*GRID_SIZE_X];
    Cell* top = y > 0 ? cells[(y - 1)*GRID_SIZE_X + x]
                      : cells[(GRID_SIZE_Y - 1)*GRID_SIZE_X + x];
    Cell* bottom = y < GRID_SIZE_Y - 1 ? cells[(y + 1)*GRID_SIZE_X + x]
                                       : cells[x];

    // left, top, right, bottom
    cells[tid]->set_neighbors(left, top, right, bottom);

    // Initialize with random agent.
    auto& rand_state = cells[tid]->random_state();
    uint32_t agent_type = curand(&rand_state) % 4;
    if (agent_type == 0) {
      auto* agent = device_allocator->make_new<Fish>(curand(&rand_state));
      assert(agent != nullptr);
      cells[tid]->enter(agent);
    } else if (agent_type == 1) {
      auto* agent = device_allocator->make_new<Shark>(curand(&rand_state));
      assert(agent != nullptr);
      cells[tid]->enter(agent);
    } else {
      // Free cell.
    }
  }
}

// Problem: It is not easy to keep track of all objects of a class if they are
// dynamically allocated. But we want to benchmark the performance of new/
// delete in CUDA.
// Solution: Fill these arrays in a separate kernel by iterating over all
// cells, storing agents in the respective array slots, and compacting the
// arrays. We do not measure the performance of these steps.
__device__ uint32_t num_sharks = 0;
__device__ Shark* sharks[GRID_SIZE_Y * GRID_SIZE_X];
__device__ uint32_t num_fish = 0;
__device__ Fish*  fish[GRID_SIZE_Y * GRID_SIZE_X];

__global__ void print_checksum() {
  uint64_t chksum = 0;

  // Sorting of the array does not matter in the calculation here.
  for (int i = 0; i < num_sharks; ++i) {
    chksum += curand(&sharks[i]->position()->random_state()) % 601;
  }

  for (int i = 0; i < num_fish; ++i) {
    chksum += curand(&fish[i]->position()->random_state()) % 601;
  }

  uint32_t fish_use = device_allocator->DBG_used_slots<Fish>();
  uint32_t fish_num = device_allocator->DBG_allocated_slots<Fish>();
  uint32_t shark_use = device_allocator->DBG_used_slots<Shark>();
  uint32_t shark_num = device_allocator->DBG_allocated_slots<Shark>();

  printf("%" PRIu64, chksum);
  printf(",%u,%u,%u,%u\n",
         fish_use, fish_num, shark_use, shark_num);
}

// One thread per cell.
__global__ void find_fish() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->has_fish()) {
      uint32_t idx = atomicAdd(&num_fish, 1);
      fish[idx] = cells[tid]->agent()->cast<Fish>();
    }
  }
}

// One thread per cell.
__global__ void find_sharks() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->has_shark()) {
      uint32_t idx = atomicAdd(&num_sharks, 1);
      sharks[idx] = cells[tid]->agent()->cast<Shark>();
    }
  }
}

__global__ void reset_fish_array() { num_fish = 0; }
__global__ void reset_shark_array() { num_sharks = 0; }

void generate_fish_array() {
  reset_fish_array<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  find_fish<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

void generate_shark_array() {
  reset_shark_array<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  find_sharks<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

void defrag() {
  allocator_handle->parallel_defrag<Fish>(/*max_records=*/ 32,
                                          /*min_records=*/ 32);
  allocator_handle->parallel_defrag<Shark>(/*max_records=*/ 32,
                                           /*min_records=*/ 32);
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

  create_cells<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
  setup_cells<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__device__ uint32_t d_gui_map[GRID_SIZE_Y * GRID_SIZE_X];
uint32_t gui_map[GRID_SIZE_Y * GRID_SIZE_X];

__global__ void fill_gui_map() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->agent() != nullptr) {
      d_gui_map[tid] = cells[tid]->agent()->get_type();
    } else {
      d_gui_map[tid] = 0;
    }
  }
}

void update_gui_map() {
  fill_gui_map<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(gui_map, d_gui_map, sizeof(uint32_t)*GRID_SIZE_X*GRID_SIZE_Y,
                       0, cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}


void print_stats() {
  generate_fish_array();
  generate_shark_array();
  print_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  printf("           ");
}

int main(int /*argc*/, char*[] /*arvg[]*/) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*1024U*1024*1024);
  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  //printf("CUDA heap size: %lu\n", heap_size);

  initialize();

  int total_time = 0;
  for (int i = 0; i < 500; ++i) {
    if (OPTION_PRINT_STATS) {
      printf("ITERATION: %i\n", i);
      DBG_stats_kernel<<<1, 1>>>();
      gpuErrchk(cudaDeviceSynchronize());
    }

    auto time_before = std::chrono::system_clock::now();
    step();

    if (OPTION_DEFRAG) {
      for (int j = 0; j < 000; ++j) {
        defrag();
      }
    }

    auto time_after = std::chrono::system_clock::now();
    int time_running = std::chrono::duration_cast<std::chrono::microseconds>(
        time_after - time_before).count();
    total_time += time_running;
  }

  printf("%i,%i,", GRID_SIZE_Y, total_time);
  print_stats();
  return 0;
}

}  // namespace wa_tor

int main(int /*argc*/, char*[] /*arvg[]*/) {
  return wa_tor::main(0, nullptr);
}
