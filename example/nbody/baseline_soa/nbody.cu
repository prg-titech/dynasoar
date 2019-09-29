#include <chrono>
#include <curand_kernel.h>
#include <stdio.h>

#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER

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

// Arrays containing all Body objects on device.
__device__ float* dev_Body_pos_x;
__device__ float* dev_Body_pos_y;
__device__ float* dev_Body_vel_x;
__device__ float* dev_Body_vel_y;
__device__ float* dev_Body_mass;
__device__ float* dev_Body_force_x;
__device__ float* dev_Body_force_y;
__device__ float device_checksum;


__device__ void new_Body(int id, float pos_x, float pos_y,
                         float vel_x, float vel_y, float mass) {
  dev_Body_pos_x[id] = pos_x;
  dev_Body_pos_y[id] = pos_y;
  dev_Body_vel_x[id] = vel_x;
  dev_Body_vel_y[id] = vel_y;
  dev_Body_mass[id] = mass;
}


__device__ void Body_compute_force(int id) {
  dev_Body_force_x[id] = 0.0f;
  dev_Body_force_y[id] = 0.0f;

  for (int i = 0; i < kNumBodies; ++i) {
    // Do not compute force with the body itself.
    if (id != i) {
      float dx = dev_Body_pos_x[i] - dev_Body_pos_x[id];
      float dy = dev_Body_pos_y[i] - dev_Body_pos_y[id];
      float dist = sqrt(dx*dx + dy*dy);
      float F = kGravityConstant * dev_Body_mass[id] * dev_Body_mass[i]
          / (dist * dist + kDampeningFactor);
      dev_Body_force_x[id] += F*dx / dist;
      dev_Body_force_y[id] += F*dy / dist;
    }
  }
}


__device__ void Body_update(int id) {
  dev_Body_vel_x[id] += dev_Body_force_x[id]*kDt / dev_Body_mass[id];
  dev_Body_vel_y[id] += dev_Body_force_y[id]*kDt / dev_Body_mass[id];
  dev_Body_pos_x[id] += dev_Body_vel_x[id]*kDt;
  dev_Body_pos_y[id] += dev_Body_vel_y[id]*kDt;

  if (dev_Body_pos_x[id] < -1 || dev_Body_pos_x[id] > 1) {
    dev_Body_vel_x[id] = -dev_Body_vel_x[id];
  }

  if (dev_Body_pos_y[id] < -1 || dev_Body_pos_y[id] > 1) {
    dev_Body_vel_y[id] = -dev_Body_vel_y[id];
  }
}


__device__ void Body_add_checksum(int id) {
  atomicAdd(&device_checksum, dev_Body_pos_x[id] + dev_Body_pos_y[id]*2
      + dev_Body_vel_x[id]*3 + dev_Body_vel_y[id]*4);
}


__global__ void kernel_initialize_bodies(float* pos_x, float* pos_y,
                                         float* vel_x, float* vel_y,
                                         float* mass, float* force_x,
                                         float* force_y) {
  dev_Body_pos_x = pos_x;
  dev_Body_pos_y = pos_y;
  dev_Body_vel_x = vel_x;
  dev_Body_vel_y = vel_y;
  dev_Body_mass = mass;
  dev_Body_force_x = force_x;
  dev_Body_force_y = force_y;

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
#ifdef OPTION_RENDER
  init_renderer();
#endif  // OPTION_RENDER

  float* host_Body_pos_x;
  float* host_Body_pos_y;
  float* host_Body_vel_x;
  float* host_Body_vel_y;
  float* host_Body_mass;
  float* host_Body_force_x;
  float* host_Body_force_y;

  // Allocate and create Body objects.
  cudaMalloc(&host_Body_pos_x, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_pos_y, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_vel_x, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_vel_y, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_mass, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_force_x, sizeof(float)*kNumBodies);
  cudaMalloc(&host_Body_force_y, sizeof(float)*kNumBodies);

#ifdef OPTION_RENDER
  float Body_pos_x[kNumBodies];
  float Body_pos_y[kNumBodies];
  float Body_mass[kNumBodies];
#endif  // OPTION_RENDER

  auto time_start = std::chrono::system_clock::now();

  kernel_initialize_bodies<<<128, 128>>>(
      host_Body_pos_x, host_Body_pos_y,
      host_Body_vel_x, host_Body_vel_y,
      host_Body_mass, host_Body_force_x,
      host_Body_force_y);
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

#ifdef OPTION_RENDER
    cudaMemcpy(Body_pos_x, host_Body_pos_x, sizeof(float)*kNumBodies,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(Body_pos_y, host_Body_pos_y, sizeof(float)*kNumBodies,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(Body_mass, host_Body_mass, sizeof(float)*kNumBodies,
               cudaMemcpyDeviceToHost);

    init_frame();
    for (int i = 0; i < kNumBodies; ++i) {
      draw_body(Body_pos_x[i], Body_pos_y[i], Body_mass[i]);
    }
    show_frame();
#endif  // OPTION_RENDER
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

  cudaFree(host_Body_pos_x);
  cudaFree(host_Body_pos_y);
  cudaFree(host_Body_vel_x);
  cudaFree(host_Body_vel_y);
  cudaFree(host_Body_mass);
  cudaFree(host_Body_force_x);
  cudaFree(host_Body_force_y);

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
