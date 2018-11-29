#include "example/game-of-life/soa/gol.h"
#include "example/game-of-life/soa/configuration.h"
#include "example/game-of-life/soa/rendering.h"


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Rendering array.
__device__ char device_render_cells[SIZE_X*SIZE_Y];
char host_render_cells[SIZE_X*SIZE_Y];


__device__ Cell* cells[SIZE_X*SIZE_Y];


__device__ Cell::Cell() : agent_(nullptr) {}


__device__ Agent* Cell::agent() { return agent_; }


__device__ bool Cell::is_empty() { return agent_ == nullptr; }


__device__ bool Cell::is_alive() {
  return !is_empty() && agent_->get_type() == TYPE_ID(AllocatorT, Alive);
}


__device__ bool Cell::is_candidate() {
  return !is_empty() && agent_->get_type() == TYPE_ID(AllocatorT, Candidate);
}


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
        if (cells[ny*SIZE_X + nx]->is_alive()) {
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
        if (cells[ny*SIZE_X + nx]->is_alive()) {
          Alive* alive = static_cast<Alive*>(cells[ny*SIZE_X + nx]->agent());
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


int encode_cell_coords(int x, int y) {
  return SIZE_X*y + x;
}


void load_glider() {
  // Create data set.
  int cell_ids[5];
  cell_ids[0] = encode_cell_coords(1, 0);
  cell_ids[1] = encode_cell_coords(2, 1);
  cell_ids[2] = encode_cell_coords(0, 2);
  cell_ids[3] = encode_cell_coords(1, 2);
  cell_ids[4] = encode_cell_coords(2, 2);

  int* dev_cell_ids;
  cudaMalloc(&dev_cell_ids, sizeof(int)*5);
  cudaMemcpy(dev_cell_ids, cell_ids, sizeof(int)*5, cudaMemcpyHostToDevice);

  load_game<<<1, 5>>>(dev_cell_ids, 5);
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(dev_cell_ids);

  allocator_handle->parallel_do<Alive, &Alive::update>();
}


void render() {
  initialize_render_arrays<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
  allocator_handle->parallel_do<Alive, &Alive::update_render_array>();

  cudaMemcpyFromSymbol(host_render_cells, device_render_cells,
                       sizeof(char)*SIZE_X*SIZE_Y, 0, cudaMemcpyDeviceToHost);
  draw(host_render_cells);
}


int main(int argc, char** argv) {
  if (OPTION_DRAW) {
    init_renderer();
  }

  AllocatorT::DBG_print_stats();
  
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*1024U*1024*1024);
  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Initialize cells.
  create_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Load data set.
  load_glider();

  // Run simulation.
  for (int i = 0; i < 500; ++i) {
    printf("Iteration %i\n", i);
    allocator_handle->parallel_do<Candidate, &Candidate::prepare>();
    allocator_handle->parallel_do<Alive, &Alive::prepare>();
    allocator_handle->parallel_do<Candidate, &Candidate::update>();
    allocator_handle->parallel_do<Alive, &Alive::update>();

    if (OPTION_DRAW) {
      render();
    }
  }

  if (OPTION_DRAW) {
    close_renderer();
  }

  return 0;
}
