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

#define NUM 20000000

struct Body {
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
};

__device__ Body objs[NUM];

__device__ uint64_t ptrs[NUM];

#define DELTA 0.5

__global__ void init() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    objs[tid].pos_x = (tid % 1231241) % 2000;
    objs[tid].pos_y = (tid % 1231247) % 2000;
    objs[tid].vel_x = (tid % 1231243) % 20 - 10;
    objs[tid].vel_y = (tid % 123124789) % 20 - 10;

    ptrs[tid] = tid;
  }
}

__global__ void kernel() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    uint32_t obj_id = ptrs[tid];

    objs[obj_id].pos_x += DELTA * objs[obj_id].vel_x;
    objs[obj_id].pos_y += DELTA * objs[obj_id].vel_y;

    if (objs[obj_id].pos_x < 0 || objs[obj_id].pos_x >= 2000) {
      objs[obj_id].vel_x = -objs[obj_id].vel_x;
    }
    if (objs[obj_id].pos_y < 0 || objs[obj_id].pos_y >= 2000) {
      objs[obj_id].vel_y = -objs[obj_id].vel_y;
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