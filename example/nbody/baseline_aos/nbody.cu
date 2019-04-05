#include <chrono>
#include <curand_kernel.h>
#include <stdio.h>

#include "../configuration.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const int kCudaBlockSize = 256;

struct Body {
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float mass;
  float force_x;
  float force_y;
};

__device__ Body* dev_bodies;
__device__ float device_checksum;


__device__ void new_Body(int id, float pos_x, float pos_y,
                         float vel_x, float vel_y, float mass) {
  dev_bodies[id].pos_x = pos_x;
  dev_bodies[id].pos_y = pos_y;
  dev_bodies[id].vel_x = vel_x;
  dev_bodies[id].vel_y = vel_y;
  dev_bodies[id].mass = mass;
}


__device__ void Body_compute_force(int id) {
  dev_bodies[id].force_x = 0.0f;
  dev_bodies[id].force_y = 0.0f;

  for (int i = 0; i < kNumBodies; ++i) {
    // Do not compute force with the body itself.
    if (id != i) {
      float dx = dev_bodies[i].pos_x - dev_bodies[id].pos_x;
      float dy = dev_bodies[i].pos_y - dev_bodies[id].pos_y;
      float dist = sqrt(dx*dx + dy*dy);
      float F = kGravityConstant * dev_bodies[id].mass * dev_bodies[i].mass
          / (dist * dist + kDampeningFactor);
      dev_bodies[id].force_x += F*dx / dist;
      dev_bodies[id].force_y += F*dy / dist;
    }
  }
}


__device__ void Body_update(int id) {
  dev_bodies[id].vel_x += dev_bodies[id].force_x*kDt / dev_bodies[id].mass;
  dev_bodies[id].vel_y += dev_bodies[id].force_y*kDt / dev_bodies[id].mass;
  dev_bodies[id].pos_x += dev_bodies[id].vel_x*kDt;
  dev_bodies[id].pos_y += dev_bodies[id].vel_y*kDt;

  if (dev_bodies[id].pos_x < -1 || dev_bodies[id].pos_x > 1) {
    dev_bodies[id].vel_x = -dev_bodies[id].vel_x;
  }

  if (dev_bodies[id].pos_y < -1 || dev_bodies[id].pos_y > 1) {
    dev_bodies[id].vel_y = -dev_bodies[id].vel_y;
  }
}


__device__ void Body_add_checksum(int id) {
  atomicAdd(&device_checksum, dev_bodies[id].pos_x + dev_bodies[id].pos_y*2
      + dev_bodies[id].vel_x*3 + dev_bodies[id].vel_y*4);
}


__global__ void kernel_initialize_bodies(Body* bodies) {
  dev_bodies = bodies;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    // Initialize random state.
    curandState rand_state;
    curand_init(kSeed, i, 0, &rand_state);

    // Create new Body object.
    new_Body(/*id=*/ i,
             /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
             /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
             /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


__global__ void kernel_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    Body_compute_force(i);
  }
}


__global__ void kernel_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    Body_update(i);
  }
}


__global__ void kernel_compute_checksum() {
  device_checksum = 0.0f;
  for (int i = 0; i < kNumBodies; ++i) {
    Body_add_checksum(i);
  }
}


int main(int /*argc*/, char** /*argv*/) {
  Body* host_bodies;

  // Allocate and create Body objects.
  cudaMalloc(&host_bodies, sizeof(Body)*kNumBodies);

  auto time_start = std::chrono::system_clock::now();

  kernel_initialize_bodies<<<128, 128>>>(host_bodies);
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumIterations; ++i) {
    kernel_compute_force<<<
        (kNumBodies + kCudaBlockSize - 1)/kCudaBlockSize,
        kCudaBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_update<<<
        (kNumBodies + kCudaBlockSize - 1)/kCudaBlockSize,
        kCudaBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

  printf("%lu\n", micros);

#ifndef NDEBUG
  kernel_compute_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  float checksum;
  cudaMemcpyFromSymbol(&checksum, device_checksum, sizeof(device_checksum), 0,
                       cudaMemcpyDeviceToHost);
  printf("Checksum: %f\n", checksum);
#endif  // NDEBUG

  cudaFree(host_bodies);

  return 0;
}

