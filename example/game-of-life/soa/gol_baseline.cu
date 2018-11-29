#include <chrono>
#include <stdio.h>

#include "example/game-of-life/soa/configuration.h"
#include "example/game-of-life/soa/rendering.h"


using CellT = char;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Data structure.
__device__ CellT* dev_cells;
__device__ CellT* dev_next_cells;
CellT* host_cells;
CellT* host_next_cells;

// Only for rendering.
CellT host_render_cells[SIZE_X*SIZE_Y];


__global__ void initialize_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < SIZE_X*SIZE_Y; i += blockDim.x * gridDim.x) {
    dev_cells[i] = 0;
    dev_next_cells[i] = 0;
  }
}


__global__ void load_game(int* cell_ids, int num_cells) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_cells; i += blockDim.x * gridDim.x) {
    dev_cells[cell_ids[i]] = 1;
    dev_next_cells[cell_ids[i]] = 1;
  }
}


__global__ void update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < SIZE_X*SIZE_Y; i += blockDim.x * gridDim.x) {
    // Check all neigboring cells.
    int num_alive = 0;

    int cell_x = i % SIZE_X;
    int cell_y = i / SIZE_X;

    for (int dx = -1; dx < 2; ++dx) {
      for (int dy = -1; dy < 2; ++dy) {
        int nx = cell_x + dx;
        int ny = cell_y + dy;

        if ((dx != 0 || dy != 0)
             && nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
          num_alive += dev_cells[i];
        }
      }
    }

    if (num_alive < 2 || num_alive > 3) {
      dev_next_cells[i] = 0;
    } else if (num_alive == 3) {
      dev_next_cells[i] = 1;
    }
  }
}


__global__ void swap_arrays() {
  if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
    auto* tmp = dev_cells;
    dev_cells = dev_next_cells;
    dev_next_cells = tmp;
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
}


void render() {
  cudaMemcpy(host_render_cells, host_cells, sizeof(CellT)*SIZE_X*SIZE_Y,
             cudaMemcpyDeviceToHost);
  draw(host_render_cells);
}


int main(int argc, char** argv) {
  if (OPTION_DRAW) {
    init_renderer();
  }

  // Allocate device memory.
  cudaMalloc(&host_cells, sizeof(CellT)*SIZE_X*SIZE_Y);
  cudaMalloc(&host_next_cells, sizeof(CellT)*SIZE_X*SIZE_Y);
  cudaMemcpyToSymbol(dev_cells, &host_cells, sizeof(CellT*), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_next_cells, &host_next_cells, sizeof(CellT*), 0,
                     cudaMemcpyHostToDevice);

  // Initialize cells.
  initialize_cells<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Load data set.
  load_glider();

  // Run simulation.
  for (int i = 0; i < 500; ++i) {
    printf("Iteration %i\n", i);
    // TODO: Tune launch configuration.
    update<<<128, 256>>>();
    gpuErrchk(cudaDeviceSynchronize());
    swap_arrays<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    if (OPTION_DRAW) {
      render();
    }
  }

  if (OPTION_DRAW) {
    close_renderer();
  }

  // Free device memory.
  cudaFree(host_cells);
  cudaFree(host_next_cells);

  return 0;
}
