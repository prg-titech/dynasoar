#include <stdio.h>
#include <iostream>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define NUM 10000000

__device__ float pos_x[NUM];
__device__ float pos_y[NUM];
__device__ float vel_x[NUM];
__device__ float vel_y[NUM];

__device__ uint64_t ptrs[NUM];

#define DELTA 0.5

__global__ void init() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    pos_x[tid] = (tid % 1231241) % 2000;
    pos_y[tid] = (tid % 1231247) % 2000;
    vel_x[tid] = (tid % 1231243) % 20 - 10;
    vel_y[tid] = (tid % 123124789) % 20 - 10;

    ptrs[tid] = tid;
  }
}

__global__ void kernel() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    uint32_t obj_id = ptrs[tid];

    pos_x[obj_id] += DELTA * vel_x[obj_id];
    pos_y[obj_id] += DELTA * vel_y[obj_id];

    if (pos_x[obj_id] < 0 || pos_x[obj_id] >= 2000) {
      vel_x[obj_id] = -vel_x[obj_id];
    }
    if (pos_y[obj_id] < 0 || pos_y[obj_id] >= 2000) {
      vel_y[obj_id] = -vel_y[obj_id];
    }
  }
}

int main() {
  init<<<NUM/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  kernel<<<NUM/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
  kernel<<<NUM/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
  kernel<<<NUM/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
  auto finish = std::chrono::steady_clock::now();
  double elapsed_seconds = std::chrono::duration_cast<
      std::chrono::duration<double>>(finish - start).count();

  std::cout << elapsed_seconds << "\n";
}