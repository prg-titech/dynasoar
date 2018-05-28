#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>

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
#define SOA_SIZE 64

struct soa_container {
  float pos_x[SOA_SIZE];
  float pos_y[SOA_SIZE];
  float vel_x[SOA_SIZE];
  float vel_y[SOA_SIZE];
};

using reference = uint64_t;

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

    uint64_t base = reinterpret_cast<uint64_t>(aosoa + aos_idx);
    assert((base & 0x000000000000003F) == 0);

    ptrs[tid] = base + soa_idx;
  }
}

__global__ void kernel() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < NUM) {
    uint64_t base = ptrs[tid] & ~((uint64_t)0x000000000000003F);
    int soa_idx = ptrs[tid] & 0x000000000000003F;

    soa_container& cont = *reinterpret_cast<soa_container*>(base);

    cont.pos_x[soa_idx] += DELTA * cont.vel_x[soa_idx];
    cont.pos_y[soa_idx] += DELTA * cont.vel_y[soa_idx];

    if (cont.pos_x[soa_idx] < 0 || cont.pos_x[soa_idx] >= 2000) {
      cont.vel_x[soa_idx] = -cont.vel_x[soa_idx];
    }
    if (cont.pos_y[soa_idx] < 0 || cont.pos_y[soa_idx] >= 2000) {
      cont.vel_y[soa_idx] = -cont.vel_y[soa_idx];
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
