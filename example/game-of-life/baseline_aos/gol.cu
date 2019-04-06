#include <assert.h>
#include <chrono>
#include <cub/cub.cuh>
#include <stdio.h>

#include "../configuration.h"
#include "../dataset_loader.h"

using IndexT = int;

static const int kActionNone = 0;
static const int kActionDie = 1;
static const int kActionSpawnAlive = 2;

static const int kNumBlockSize = 256;

static const int kAgentTypeNone = 0;
static const int kAgentTypeCandidate = 1;
static const int kAgentTypeAlive = 2;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Cell {
  char agent_type;
  char agent_action;
  bool alive_is_new;
  int redirection_index;
};

__device__ Cell* cells;

// Object/array size counters.
__device__ int d_num_candidates;
__device__ int d_num_alive;
int host_num_candidates;
int host_num_alive;

// Allocation data structures.
__device__ IndexT* d_candidates;
__device__ IndexT* d_alive;
__device__ int* d_Candidate_active;
__device__ int* d_Alive_active;

// Host arrays for prefix sum compaction.
IndexT* h_candidates;
IndexT* h_alive;
int* h_Candidate_active;
int* h_Alive_active;
IndexT* h_candidates_2;
IndexT* h_alive_2;
int* h_Candidate_active_2;
int* h_Alive_active_2;
int* h_prefix_sum_temp;
int* h_prefix_sum_output;

// For prefix sum compaction.
__device__ int* d_prefix_sum_temp;
__device__ int* d_prefix_sum_output;
__device__ IndexT* d_candidates_2;
__device__ IndexT* d_alive_2;
__device__ int* d_Candidate_active_2;
__device__ int* d_Alive_active_2;

// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
dataset_t dataset;


__device__ void new_Candidate(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeNone);

  int idx = atomicAdd(&d_num_candidates, 1);
  d_candidates[idx] = self;
  d_Candidate_active[idx] = 1;

  cells[self].redirection_index = idx;
  cells[self].agent_type = kAgentTypeCandidate;
  cells[self].agent_action = kActionNone;
}


__device__ void delete_Candidate(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeCandidate);
  cells[self].agent_type = kAgentTypeNone;

  int idx = cells[self].redirection_index;
  assert(d_Candidate_active[idx] == 1);
  d_Candidate_active[idx] = 0;
}


__device__ void new_Alive(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeNone);

  int idx = atomicAdd(&d_num_alive, 1);
  d_alive[idx] = self;
  d_Alive_active[idx] = 1;

  cells[self].redirection_index = idx;
  cells[self].agent_type = kAgentTypeAlive;
  cells[self].agent_action = kActionNone;
  cells[self].alive_is_new = true;
}


__device__ void delete_Alive(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeAlive);
  cells[self].agent_type = kAgentTypeNone;

  int idx = cells[self].redirection_index;
  assert(d_Alive_active[idx] == 1);
  d_Alive_active[idx] = 0;
}


__device__ void change_Alive_to_Candidate(IndexT self) {
  // Delete alive. Without reseting type.
  assert(cells[self].agent_type == kAgentTypeAlive);
  int idx = cells[self].redirection_index;
  assert(d_Alive_active[idx] == 1);
  d_Alive_active[idx] = 0;

  // Create candidate.
  idx = atomicAdd(&d_num_candidates, 1);
  d_candidates[idx] = self;
  d_Candidate_active[idx] = 1;
  cells[self].redirection_index = idx;
  cells[self].agent_type = kAgentTypeCandidate;
  cells[self].agent_action = kActionNone;
}


__device__ int device_checksum;
__device__ int device_chk_num_candidates;

__device__ void Alive_update_checksum(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeAlive);
  atomicAdd(&device_checksum, 1);
}

__device__ void Candidate_update_checksum(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeCandidate);
  atomicAdd(&device_chk_num_candidates, 1);
}


__device__ int Cell_num_alive_neighbors(IndexT self) {
  int cell_x = self % SIZE_X;
  int cell_y = self / SIZE_X;
  int result = 0;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        assert(cells[ny*SIZE_X + nx].agent_type >= 0
            && cells[ny*SIZE_X + nx].agent_type <= 2);

        if (cells[ny*SIZE_X + nx].agent_type == kAgentTypeAlive) {
          result++;
        }
      }
    }
  }

  return result;
}


__device__ void Alive_prepare(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeAlive);

  cells[self].alive_is_new = false;

  // Also counts this object itself.
  int alive_neighbors = Cell_num_alive_neighbors(self) - 1;

  if (alive_neighbors < 2 || alive_neighbors > 3) {
    cells[self].agent_action = kActionDie;
  }
}


__device__ void Alive_maybe_create_candidate(IndexT self, int x, int y) {
  // Check neighborhood of cell to determine who should create Candidate.
  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        IndexT n_cell = ny*SIZE_X + nx;
        if (cells[n_cell].agent_type == kAgentTypeAlive) {
          if (cells[n_cell].alive_is_new) {
            if (n_cell == self) {
              // Create candidate now.
              new_Candidate(y*SIZE_X + x);
            }  // else: Created by other thread.

            return;
          }
        }
      }
    }
  }

  assert(false);
}


__device__ void Alive_create_candidates(IndexT self) {
  assert(cells[self].alive_is_new);

  // TODO: Consolidate with Agent::num_alive_neighbors().
  int cell_x = self % SIZE_X;
  int cell_y = self / SIZE_X;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        if (cells[ny*SIZE_X + nx].agent_type == kAgentTypeNone) {
          // Candidate should be created here.
          Alive_maybe_create_candidate(self, nx, ny);
        }
      }
    }
  }
}


__device__ void Alive_update(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeAlive);

  // TODO: Consider splitting in two classes for less divergence.
  if (cells[self].alive_is_new) {
    // Create candidates in neighborhood.
    Alive_create_candidates(self);
  } else {
    if (cells[self].agent_action == kActionDie) {
      // Replace with Candidate. Or should we?
      change_Alive_to_Candidate(self);
    }
  }
}



__device__ void Candidate_prepare(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeCandidate);

  int alive_neighbors = Cell_num_alive_neighbors(self);

  if (alive_neighbors == 3) {
    cells[self].agent_action = kActionSpawnAlive;
  } else if (alive_neighbors == 0) {
    cells[self].agent_action = kActionDie;
  }
}


__device__ void Candidate_update(IndexT self) {
  assert(cells[self].agent_type == kAgentTypeCandidate);

  if (cells[self].agent_action == kActionSpawnAlive) {
    delete_Candidate(self);
    new_Alive(self);
  } else if (cells[self].agent_action == kActionDie) {
    delete_Candidate(self);
  }
}


__global__ void create_cells() {
  if (blockDim.x == 0 && gridDim.x == 0) {
    d_num_candidates = d_num_alive = 0;
  }

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < SIZE_X*SIZE_Y; i += blockDim.x * gridDim.x) {
    cells[i].agent_type = kAgentTypeNone;
    d_Alive_active[i] = 0;
    d_Candidate_active[i] = 0;
  }
}


// Must be followed by Alive::update().
__global__ void load_game(int* cell_ids, int num_cells) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_cells; i += blockDim.x * gridDim.x) {
    new_Alive(cell_ids[i]);
  }
}


__global__ void kernel_Alive_prepare() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_alive; i += blockDim.x * gridDim.x) {
    if (d_Alive_active[i]) {
      Alive_prepare(d_alive[i]);
    }
  }
}


__global__ void kernel_Alive_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_alive; i += blockDim.x * gridDim.x) {
    if (d_Alive_active[i]) {
      Alive_update(d_alive[i]);
    }
  }
}


__global__ void kernel_Alive_update_checksum() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_alive; i += blockDim.x * gridDim.x) {
    if (d_Alive_active[i]) {
      Alive_update_checksum(d_alive[i]);
    }
  }
}


__global__ void kernel_Candidate_update_checksum() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_candidates; i += blockDim.x * gridDim.x) {
    if (d_Candidate_active[i]) {
      Candidate_update_checksum(d_candidates[i]);
    }
  }
}


__global__ void kernel_Candidate_prepare() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_candidates; i += blockDim.x * gridDim.x) {
    if (d_Candidate_active[i]) {
      Candidate_prepare(d_candidates[i]);
    }
  }
}


__global__ void kernel_Candidate_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_candidates; i += blockDim.x * gridDim.x) {
    if (d_Candidate_active[i]) {
      Candidate_update(d_candidates[i]);
    }
  }
}


// Only for debugging.
__global__ void kernel_check_consistency() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_candidates; i += blockDim.x * gridDim.x) {
    if (d_Candidate_active[i]) {
      assert(cells[d_candidates[i]].agent_type == kAgentTypeCandidate);
    }
  }

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < d_num_alive; i += blockDim.x * gridDim.x) {
    if (d_Alive_active[i]) {
      assert(cells[d_alive[i]].agent_type == kAgentTypeAlive);
    }
  }
}


void update_object_counters() {
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&host_num_alive, d_num_alive, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&host_num_candidates, d_num_candidates, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}


void dbg_check_consistency() {
  update_object_counters();

  kernel_check_consistency<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void transfer_dataset() {
  int* dev_cell_ids;
  cudaMalloc(&dev_cell_ids, sizeof(int)*dataset.num_alive);
  cudaMemcpy(dev_cell_ids, dataset.alive_cells, sizeof(int)*dataset.num_alive,
             cudaMemcpyHostToDevice);

#ifndef NDEBUG
  printf("Loading on GPU: %i alive cells.\n", dataset.num_alive);
  printf("Number of cells: %i\n", dataset.x*dataset.y);
#endif  // NDEBUG

  load_game<<<128, 128>>>(dev_cell_ids, dataset.num_alive);
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(dev_cell_ids);

  update_object_counters();

  kernel_Alive_update<<<
      (host_num_alive + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  update_object_counters();
  dbg_check_consistency();
}


int checksum() {
  update_object_counters();

  int host_candidates = 0;
  int host_checksum = 0;
  cudaMemcpyToSymbol(device_checksum, &host_checksum, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(device_chk_num_candidates, &host_candidates, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  kernel_Alive_update_checksum<<<
      (host_num_alive + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Candidate_update_checksum<<<
      (host_num_candidates + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&host_checksum, device_checksum, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);

  cudaMemcpyFromSymbol(&host_candidates, device_chk_num_candidates, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);

  return host_checksum;
}


__global__ void kernel_compact_alive() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_alive; i += blockDim.x * gridDim.x) {
    if (d_Alive_active[i]) {
      int target = d_prefix_sum_output[i];

      cells[d_alive[i]].redirection_index = target;
      d_Alive_active_2[target] = 1;
      d_alive_2[target] = d_alive[i];
    }
  }
}


__global__ void kernel_compact_swap_pointers_alive() {
  // Update size.
  d_num_alive = d_prefix_sum_output[d_num_alive - 1] + d_Alive_active[d_num_alive - 1];

  {
    auto* tmp = d_alive_2;
    d_alive_2 = d_alive;
    d_alive = tmp;
  }

  {
    auto* tmp = d_Alive_active_2;
    d_Alive_active_2 = d_Alive_active;
    d_Alive_active = tmp;
  }
}


__global__ void kernel_compact_candidates() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_candidates; i += blockDim.x * gridDim.x) {
    if (d_Candidate_active[i]) {
      int target = d_prefix_sum_output[i];

      cells[d_candidates[i]].redirection_index = target;
      d_Candidate_active_2[target] = 1;
      d_candidates_2[target] = d_candidates[i];
    }
  }
}


__global__ void kernel_compact_swap_pointers_candidates() {
  // Update size.
  d_num_candidates = d_prefix_sum_output[d_num_candidates - 1]
      + d_Candidate_active[d_num_candidates - 1];

  {
    auto* tmp = d_candidates_2;
    d_candidates_2 = d_candidates;
    d_candidates = tmp;
  }

  {
    auto* tmp = d_Candidate_active_2;
    d_Candidate_active_2 = d_Candidate_active;
    d_Candidate_active = tmp;
  }
}


void prefix_sum_step() {
  update_object_counters();

  if (true) {
    // TODO: Prefix sum broken for num_objects < 256.
    auto prefix_sum_size = host_num_alive < 256 ? 256 : host_num_alive;
    size_t temp_size = 3*dataset.x*dataset.y;
    cub::DeviceScan::ExclusiveSum(h_prefix_sum_temp,
                                  temp_size,
                                  h_Alive_active,
                                  h_prefix_sum_output,
                                  prefix_sum_size);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_alive<<<
        (host_num_alive + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_swap_pointers_alive<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    auto* tmp = h_Alive_active_2;
    h_Alive_active_2 = h_Alive_active;
    h_Alive_active = tmp;
  }

  if (true) {
    // TODO: Prefix sum broken for num_objects < 256.
    auto prefix_sum_size = host_num_candidates < 256 ? 256 : host_num_candidates;
    size_t temp_size = 3*dataset.x*dataset.y;
    cub::DeviceScan::ExclusiveSum(h_prefix_sum_temp,
                                  temp_size,
                                  h_Candidate_active,
                                  h_prefix_sum_output,
                                  prefix_sum_size);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_candidates<<<
        (host_num_candidates + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_swap_pointers_candidates<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    auto* tmp = h_Candidate_active_2;
    h_Candidate_active_2 = h_Candidate_active;
    h_Candidate_active = tmp;
  }

  update_object_counters();
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s filename.pgm\n", argv[0]);
    exit(1);
  } else {
    // Load data set.
    dataset = load_from_file(argv[1]);
  }

  cudaMemcpyToSymbol(SIZE_X, &dataset.x, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(SIZE_Y, &dataset.y, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*1024U*1024*1024);
  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Allocate memory.
  Cell* h_cells;
  cudaMalloc(&h_cells, sizeof(Cell)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(cells, &h_cells, sizeof(Cell*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_candidates, sizeof(IndexT)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_candidates, &h_candidates, sizeof(IndexT*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_alive, sizeof(IndexT)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_alive, &h_alive, sizeof(IndexT*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_Candidate_active, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_Candidate_active, &h_Candidate_active, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_Alive_active, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_Alive_active, &h_Alive_active, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_candidates_2, sizeof(IndexT)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_candidates_2, &h_candidates_2, sizeof(IndexT*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_alive_2, sizeof(IndexT)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_alive_2, &h_alive_2, sizeof(IndexT*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_Candidate_active_2, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_Candidate_active_2, &h_Candidate_active_2, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_Alive_active_2, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_Alive_active_2, &h_Alive_active_2, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_prefix_sum_temp, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_prefix_sum_temp, &h_prefix_sum_temp, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  cudaMalloc(&h_prefix_sum_output, sizeof(int)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(d_prefix_sum_output, &h_prefix_sum_output, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  // Initialize cells.
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  transfer_dataset();

  auto time_start = std::chrono::system_clock::now();

  // Run simulation.
  for (int i = 0; i < kNumIterations; ++i) {
#ifndef NDEBUG
    dbg_check_consistency();
#endif  // NDEBUG

    kernel_Candidate_prepare<<<
        (host_num_candidates + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_Alive_prepare<<<
        (host_num_alive + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_Candidate_update<<<
        (host_num_candidates + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());
    update_object_counters();

    kernel_Alive_update<<<
        (host_num_alive + kNumBlockSize - 1) / kNumBlockSize,
        kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());
    update_object_counters();

    // Compact arrays.
    prefix_sum_step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

  printf("%lu\n", micros);

  return 0;
}
