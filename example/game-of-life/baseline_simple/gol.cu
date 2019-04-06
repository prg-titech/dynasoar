// Source adapted from:
// https://www.olcf.ornl.gov/tutorials/cuda-game-of-life/

#include <chrono>
#include <stdio.h>

#include "../configuration.h"
#include "../dataset_loader.h"
#include "../rendering.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// GoL dataset.
dataset_t dataset;


// Data structure.
CellT* host_cells;
CellT* host_next_cells;

// Only for rendering.
CellT* render_cells;


__global__ void initialize_cells(CellT* dev_cells, CellT* dev_next_cells,
                                 int size_x, int size_y) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < size_x*size_y; i += blockDim.x * gridDim.x) {
    dev_cells[i] = 0;
    dev_next_cells[i] = 0;
  }
}


__global__ void load_game(int* cell_ids, int num_cells,
                          CellT* dev_cells, CellT* dev_next_cells) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_cells; i += blockDim.x * gridDim.x) {
    dev_cells[cell_ids[i]] = 1;
  }
}


__global__ void update(CellT* dev_cells, CellT* dev_next_cells,
                       int size_x, int size_y) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < size_x*size_y; i += blockDim.x * gridDim.x) {
    // Check all neigboring cells.
    int num_alive = 0;

    int cell_x = i % size_x;
    int cell_y = i / size_x;

    for (int dx = -1; dx < 2; ++dx) {
      for (int dy = -1; dy < 2; ++dy) {
        int nx = cell_x + dx;
        int ny = cell_y + dy;

        if ((dx != 0 || dy != 0)
             && nx > -1 && nx < size_x && ny > -1 && ny < size_y) {
          num_alive += dev_cells[ny*size_x + nx];
        }
      }
    }

    if (dev_cells[i] == 1 && (num_alive < 2 || num_alive > 3)) {
      dev_next_cells[i] = 0;
    } else if (dev_cells[i] == 1 && (num_alive == 3 || num_alive == 2)) {
      dev_next_cells[i] = 1;
    } else if (dev_cells[i] == 0 && num_alive == 3) {
      dev_next_cells[i] = 1;
    } else {
      dev_next_cells[i] = dev_cells[i];
    }
  }
}


void transfer_dataset() {
  int* dev_cell_ids;
  cudaMalloc(&dev_cell_ids, sizeof(int)*dataset.num_alive);
  cudaMemcpy(dev_cell_ids, dataset.alive_cells, sizeof(int)*dataset.num_alive,
             cudaMemcpyHostToDevice);
#ifndef NDEBUG
  printf("Loading on GPU: %i alive cells.\n", dataset.num_alive);
#endif  // NDEBUG

  load_game<<<128, 128>>>(dev_cell_ids, dataset.num_alive,
                          host_cells, host_next_cells);
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(dev_cell_ids);
}


void render() {
  cudaMemcpy(render_cells, host_cells, sizeof(CellT)*dataset.x*dataset.y,
             cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  // Make a copy of array because CellT might not be char.
  char* char_render_cells = new char[dataset.x*dataset.y];
  for (int i = 0; i < dataset.x*dataset.y; ++i) {
    char_render_cells[i] = render_cells[i];
  }

  draw(char_render_cells);
  delete[] char_render_cells;
}


int checksum() {
  cudaMemcpy(render_cells, host_cells, sizeof(CellT)*dataset.x*dataset.y,
             cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  // Count number of alive cells.
  int result = 0;
  for (int i = 0; i < dataset.x*dataset.y; ++i) {
    result += render_cells[i];
  }

  return result;
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s filename.pgm\n", argv[0]);
    exit(1);
  } else {
    // Load data set.
    dataset = load_from_file(argv[1]);
  }

  if (kOptionRender) {
    init_renderer();
  }

  // Allocate device memory.
  cudaMalloc(&host_cells, sizeof(CellT)*dataset.x*dataset.y);
  cudaMalloc(&host_next_cells, sizeof(CellT)*dataset.x*dataset.y);

  // Allocate memory for rendering.
  render_cells = new CellT[dataset.x*dataset.y];

  // Initialize cells.
  initialize_cells<<<128, 128>>>(host_cells, host_next_cells,
                                 dataset.x, dataset.y);
  gpuErrchk(cudaDeviceSynchronize());

  transfer_dataset();

  if (kOptionRender) {
    render();
  }

  auto time_start = std::chrono::system_clock::now();

  // Run simulation.
  for (int i = 0; i < kNumIterations; ++i) {
    // TODO: Tune launch configuration.
    update<<<128, 256>>>(host_cells, host_next_cells,
                         dataset.x, dataset.y);
    gpuErrchk(cudaDeviceSynchronize());

    auto* tmp = host_cells;
    host_cells = host_next_cells;
    host_next_cells = tmp;

    if (kOptionRender) {
      render();
    }
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

  printf("%lu\n", micros);

  if (kOptionRender) {
    close_renderer();
  }

  // Free device memory.
  cudaFree(host_cells);
  cudaFree(host_next_cells);

  return 0;
}
