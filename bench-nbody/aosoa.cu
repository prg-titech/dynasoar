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
#define SOA_SIZE 32

struct soa_container {
  float pos_x[SOA_SIZE];
  float pos_y[SOA_SIZE];
  float vel_x[SOA_SIZE];
  float vel_y[SOA_SIZE];
};

struct reference {
  __device__ reference() {}
  __device__ reference(uint32_t a, uint32_t b) : container_id(a), soa_id(b) {}

  uint32_t container_id;
  uint32_t soa_id;
};

__device__ soa_container aosoa[NUM/SOA_SIZE + 1];

__device__ reference ptrs[NUM];

#define DELTA 0.5

__global__ void init() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    int aos_idx = tid / SOA_SIZE;
    int soa_idx = tid % SOA_SIZE;

    aosoa[aos_idx].pos_x[soa_idx] = (tid % 1231241) % 2000;
    aosoa[aos_idx].pos_y[soa_idx] = (tid % 1231247) % 2000;
    aosoa[aos_idx].vel_x[soa_idx] = (tid % 1231243) % 20 - 10;
    aosoa[aos_idx].vel_y[soa_idx] = (tid % 123124789) % 20 - 10;

    ptrs[tid] = reference(aos_idx, soa_idx);
  }
}

__global__ void kernel() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    int aos_idx = ptrs[tid].container_id;
    int soa_idx = ptrs[tid].soa_id;

    aosoa[aos_idx].pos_x[soa_idx] += DELTA * aosoa[aos_idx].vel_x[soa_idx];
    aosoa[aos_idx].pos_y[soa_idx] += DELTA * aosoa[aos_idx].vel_y[soa_idx];

    if (aosoa[aos_idx].pos_x[soa_idx] < 0 || aosoa[aos_idx].pos_x[soa_idx] >= 2000) {
      aosoa[aos_idx].vel_x[soa_idx] = -aosoa[aos_idx].vel_x[soa_idx];
    }
    if (aosoa[aos_idx].pos_y[soa_idx] < 0 || aosoa[aos_idx].pos_y[soa_idx] >= 2000) {
      aosoa[aos_idx].vel_y[soa_idx] = -aosoa[aos_idx].vel_y[soa_idx];
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