#include <chrono>

#include "gol.h"
#include "configuration.h"
#include "dataset_loader.h"
#include "rendering.h"


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Rendering array.
// TODO: Fix variable names.
__device__ char* device_render_cells;
char* host_render_cells;
char* d_device_render_cells;


// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
__device__ Cell** cells;
dataset_t dataset;


__device__ Cell::Cell() : agent_(nullptr) {}


__device__ Agent* Cell::agent() { return agent_; }


__device__ bool Cell::is_empty() { return agent_ == nullptr; }


__device__ Agent::Agent(int cell_id)
    : cell_id_(cell_id), action_(kActionNone) {}


__device__ int Agent::cell_id() { return cell_id_; }


__device__ int Agent::num_alive_neighbors() {
  int cell_x = cell_id_ % SIZE_X;
  int cell_y = cell_id_ / SIZE_X;
  int result = 0;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        if (cells[ny*SIZE_X + nx]->agent()->cast<Alive>() != nullptr) {
          result++;
        }
      }
    }
  }

  return result;
}


__device__ Alive::Alive(int cell_id) : Agent(cell_id), is_new_(true) {}


__device__ void Alive::prepare() {
  is_new_ = false;

  // Also counts this object itself.
  int alive_neighbors = num_alive_neighbors() - 1;

  if (alive_neighbors < 2 || alive_neighbors > 3) {
    action_ = kActionDie;
  }
}


__device__ void Alive::update() {
  int cid = cell_id_;

  // TODO: Consider splitting in two classes for less divergence.
  if (is_new_) {
    // Create candidates in neighborhood.
    create_candidates();
  } else {
    if (action_ == kActionDie) {
      // Replace with Candidate. Or should we?
      cells[cid]->agent_ =
          device_allocator->make_new<Candidate>(cid);
      device_allocator->free<Alive>(this);
    }
  }
}


__device__ void Alive::create_candidates() {
  assert(is_new_);

  // TODO: Consolidate with Agent::num_alive_neighbors().
  int cell_x = cell_id_ % SIZE_X;
  int cell_y = cell_id_ / SIZE_X;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        if (cells[ny*SIZE_X + nx]->is_empty()) {
          // Candidate should be created here.
          maybe_create_candidate(nx, ny);
        }
      }
    }
  }
}


__device__ void Alive::maybe_create_candidate(int x, int y) {
  // Check neighborhood of cell to determine who should create Candidate.
  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        Alive* alive = cells[ny*SIZE_X + nx]->agent()->cast<Alive>();
        if (alive != nullptr) {
          if (alive->is_new_) {
            if (alive == this) {
              // Create candidate now.
              cells[y*SIZE_X + x]->agent_ =
                  device_allocator->make_new<Candidate>(y*SIZE_X + x);
            }  // else: Created by other thread.

            return;
          }
        }
      }
    }
  }

  assert(false);
}


__device__ void Alive::update_render_array() {
  device_render_cells[cell_id_] = 1;
}


__device__ Candidate::Candidate(int cell_id) : Agent(cell_id) {}


__device__ void Candidate::prepare() {
  int alive_neighbors = num_alive_neighbors();

  if (alive_neighbors == 3) {
    action_ = kActionSpawnAlive;
  } else if (alive_neighbors == 0) {
    action_ = kActionDie;
  }
}


__device__ void Candidate::update() {
  // TODO: Why is this necessary?
  int cid = cell_id_;

  if (action_ == kActionSpawnAlive) {
    cells[cid]->agent_ = device_allocator->make_new<Alive>(cid);
    device_allocator->free<Candidate>(this);
  } else if (action_ == kActionDie) {
    cells[cid]->agent_ = nullptr;
    device_allocator->free<Candidate>(this);
  }
}


__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < SIZE_X*SIZE_Y; i += blockDim.x * gridDim.x) {
    cells[i] = device_allocator->make_new<Cell>();
  }
}


// Must be followed by Alive::update().
__global__ void load_game(int* cell_ids, int num_cells) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_cells; i += blockDim.x * gridDim.x) {
    cells[cell_ids[i]]->agent_ =
        device_allocator->make_new<Alive>(cell_ids[i]);
    assert(cells[cell_ids[i]]->agent()->cell_id() == cell_ids[i]);
  }
}


__global__ void initialize_render_arrays() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < SIZE_X*SIZE_Y; i += blockDim.x * gridDim.x) {
    device_render_cells[i] = 0;
  }
}


void render() {
  initialize_render_arrays<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
  allocator_handle->parallel_do<Alive, &Alive::update_render_array>();

  cudaMemcpy(host_render_cells, d_device_render_cells,
             sizeof(char)*dataset.x*dataset.y, cudaMemcpyDeviceToHost);
  draw(host_render_cells);
}


void transfer_dataset() {
  int* dev_cell_ids;
  cudaMalloc(&dev_cell_ids, sizeof(int)*dataset.num_alive);
  cudaMemcpy(dev_cell_ids, dataset.alive_cells, sizeof(int)*dataset.num_alive,
             cudaMemcpyHostToDevice);

#ifndef NDEBUG
  printf("Loading on GPU: %i alive cells.\n", dataset.num_alive);
#endif  // NDEBUG

  load_game<<<128, 128>>>(dev_cell_ids, dataset.num_alive);
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(dev_cell_ids);

  allocator_handle->parallel_do<Alive, &Alive::update>();
}


__device__ int device_checksum;
__device__ int device_num_candidates;

__device__ void Alive::update_checksum() {
  atomicAdd(&device_checksum, 1);
}


__device__ void Candidate::update_counter() {
  atomicAdd(&device_num_candidates, 1);
}

int checksum() {
  int host_checksum = 0;
  int host_num_candidates = 0;
  cudaMemcpyToSymbol(device_checksum, &host_checksum, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(device_num_candidates, &host_num_candidates, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  allocator_handle->parallel_do<Alive, &Alive::update_checksum>();
  allocator_handle->parallel_do<Candidate, &Candidate::update_counter>();

  cudaMemcpyFromSymbol(&host_checksum, device_checksum, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&host_num_candidates, device_num_candidates, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);

  return host_checksum;
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

  if (kOptionRender) {
    init_renderer();
  }
  
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*1024U*1024*1024);
  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Allocate memory.
  Cell** host_cells;
  cudaMalloc(&host_cells, sizeof(Cell*)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(cells, &host_cells, sizeof(Cell**), 0,
                     cudaMemcpyHostToDevice);

  cudaMalloc(&d_device_render_cells, sizeof(char)*dataset.x*dataset.y);
  cudaMemcpyToSymbol(device_render_cells, &d_device_render_cells,
                     sizeof(char*), 0, cudaMemcpyHostToDevice);

  host_render_cells = new char[dataset.x*dataset.y];

  // Initialize cells.
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  transfer_dataset();

  auto time_start = std::chrono::system_clock::now();

  // Run simulation.
  for (int i = 0; i < kNumIterations; ++i) {
    if (kOptionPrintStats) {
      printf("%i\n", i);
      //allocator_handle->DBG_print_state_stats();
      allocator_handle->DBG_collect_stats();
    }

    allocator_handle->parallel_do<Candidate, &Candidate::prepare>();
    allocator_handle->parallel_do<Alive, &Alive::prepare>();
    allocator_handle->parallel_do<Candidate, &Candidate::update>();
    allocator_handle->parallel_do<Alive, &Alive::update>();

    if (kOptionRender) {
      render();
    }
  }

  if (kOptionRender) {
    close_renderer();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

  printf("%lu,%lu\n", millis, allocator_handle->DBG_get_enumeration_time());

  if (kOptionPrintStats) {
    allocator_handle->DBG_print_collected_stats();
  }

  delete[] host_render_cells;
  cudaFree(host_cells);
  cudaFree(d_device_render_cells);

  return 0;
}
