#include <chrono>
#include <curand_kernel.h>
#include <limits>

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

using IndexT = int;
static const IndexT kNullptr = std::numeric_limits<IndexT>::max();

static const int kThreads = 256;
static const int kBlocks = (kNumBodies + kThreads - 1)/kThreads;

__device__ IndexT* dev_Body_merge_target;
__device__ float* dev_Body_pos_x;
__device__ float* dev_Body_pos_y;
__device__ float* dev_Body_vel_x;
__device__ float* dev_Body_vel_y;
__device__ float* dev_Body_force_x;
__device__ float* dev_Body_force_y;
__device__ float* dev_Body_mass;
__device__ bool* dev_Body_has_incoming_merge;
__device__ bool* dev_Body_successful_merge;
__device__ bool* dev_Body_is_active;


// Helper variables for rendering and checksum computation.
__device__ int r_draw_counter = 0;
__device__ float r_Body_pos_x[kNumBodies];
__device__ float r_Body_pos_y[kNumBodies];
__device__ float r_Body_vel_x[kNumBodies];
__device__ float r_Body_vel_y[kNumBodies];
__device__ float r_Body_mass[kNumBodies];
int host_draw_counter;
float host_Body_pos_x[kNumBodies];
float host_Body_pos_y[kNumBodies];
float host_Body_vel_x[kNumBodies];
float host_Body_vel_y[kNumBodies];
float host_Body_mass[kNumBodies];
float host_Body_is_active[kNumBodies];


__device__ void new_Body(IndexT id, float pos_x, float pos_y,
                         float vel_x, float vel_y, float mass) {
  dev_Body_pos_x[id] = pos_x;
  dev_Body_pos_y[id] = pos_y;
  dev_Body_vel_x[id] = vel_x;
  dev_Body_vel_y[id] = vel_y;
  dev_Body_mass[id] = mass;
  dev_Body_is_active[id] = true;
}


__device__ void Body_apply_force(IndexT id, IndexT other) {
  // Update `other`.
  if (other != id) {
    float dx = dev_Body_pos_x[id] - dev_Body_pos_x[other];
    float dy = dev_Body_pos_y[id] - dev_Body_pos_y[other];
    float dist = sqrt(dx*dx + dy*dy);
    float F = kGravityConstant * dev_Body_mass[id] * dev_Body_mass[other]
        / (dist * dist + kDampeningFactor);
    dev_Body_force_x[other] += F*dx / dist;
    dev_Body_force_y[other] += F*dy / dist;
  }
}


__device__ void Body_compute_force(IndexT id) {
  dev_Body_force_x[id] = 0.0f;
  dev_Body_force_y[id] = 0.0f;

  // device_do
  for (IndexT i = 0; i < kNumBodies; ++i) {
    if (dev_Body_is_active[i]) {
      Body_apply_force(i, id);
    }
  }
}


__device__ void Body_update(IndexT id) {
  dev_Body_vel_x[id] += dev_Body_force_x[id]*kTimeInterval / dev_Body_mass[id];
  dev_Body_vel_y[id] += dev_Body_force_y[id]*kTimeInterval / dev_Body_mass[id];
  dev_Body_pos_x[id] += dev_Body_vel_x[id]*kTimeInterval;
  dev_Body_pos_y[id] += dev_Body_vel_y[id]*kTimeInterval;

  if (dev_Body_pos_x[id] < -1 || dev_Body_pos_x[id] > 1) {
    dev_Body_vel_x[id] = -dev_Body_vel_x[id];
  }

  if (dev_Body_pos_y[id] < -1 || dev_Body_pos_y[id] > 1) {
    dev_Body_vel_y[id] = -dev_Body_vel_y[id];
  }
}


__device__ void Body_check_merge_into_this(IndexT id, IndexT other) {
  // Only merge into larger body.
  if (!dev_Body_has_incoming_merge[other]
      && dev_Body_mass[id] > dev_Body_mass[other]) {
    float dx = dev_Body_pos_x[id] - dev_Body_pos_x[other];
    float dy = dev_Body_pos_y[id] - dev_Body_pos_y[other];
    float dist_square = dx*dx + dy*dy;

    if (dist_square < kMergeThreshold*kMergeThreshold) {
      // Try to merge this one.
      // There is a race condition here: Multiple threads may try to merge
      // this body.
      dev_Body_merge_target[id] = other;
      dev_Body_has_incoming_merge[other] = true;
    }
  }
}


__device__ void Body_initialize_merge(IndexT id) {
  dev_Body_merge_target[id] = kNullptr;
  dev_Body_has_incoming_merge[id] = false;
  dev_Body_successful_merge[id] = false;
}


__device__ void Body_prepare_merge(IndexT id) {
  // device_do
  for (IndexT i = 0; i < kNumBodies; ++i) {
    if (dev_Body_is_active[i]) {
      Body_check_merge_into_this(i, id);
    }
  }
}


__device__ void Body_update_merge(IndexT id) {
  IndexT m = dev_Body_merge_target[id];
  if (m != kNullptr) {
    if (dev_Body_merge_target[m] == kNullptr) {
      // Perform merge.
      float new_mass = dev_Body_mass[id] + dev_Body_mass[m];
      float new_vel_x = (dev_Body_vel_x[id]*dev_Body_mass[id]
                         + dev_Body_vel_x[m]*dev_Body_mass[m]) / new_mass;
      float new_vel_y = (dev_Body_vel_y[id]*dev_Body_mass[id]
                         + dev_Body_vel_y[m]*dev_Body_mass[m]) / new_mass;
      dev_Body_mass[m] = new_mass;
      dev_Body_vel_x[m] = new_vel_x;
      dev_Body_vel_y[m] = new_vel_y;
      dev_Body_pos_x[m] = (dev_Body_pos_x[id] + dev_Body_pos_x[m]) / 2;
      dev_Body_pos_y[m] = (dev_Body_pos_y[id] + dev_Body_pos_y[m]) / 2;

      dev_Body_successful_merge[id] = true;
    }
  }
}


__device__ void Body_delete_merged(IndexT id) {
  if (dev_Body_successful_merge[id]) {
    dev_Body_is_active[id] = false;
  }
}


__device__ void Body_add_to_draw_array(IndexT id) {
  int idx = atomicAdd(&r_draw_counter, 1);
  r_Body_pos_x[idx] = dev_Body_pos_x[id];
  r_Body_pos_y[idx] = dev_Body_pos_y[id];
  r_Body_vel_x[idx] = dev_Body_vel_x[id];
  r_Body_vel_y[idx] = dev_Body_vel_y[id];
  r_Body_mass[idx] = dev_Body_mass[id];
}


__global__ void kernel_initialize_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
    new_Body(id,
             /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
             /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
             /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
             /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


__global__ void kernel_reset_draw_counters() {
  r_draw_counter = 0;
}


template<void (*func)(IndexT)>
__global__ void parallel_do() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
    if (dev_Body_is_active[id]) {
      func(id);
    }
  }
}


void transfer_data() {
  // Extract data from SoaAlloc data structure.
  kernel_reset_draw_counters<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  parallel_do<&Body_add_to_draw_array><<<kBlocks, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Copy data to host.
  cudaMemcpyFromSymbol(host_Body_pos_x, r_Body_pos_x,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_pos_y, r_Body_pos_y,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_x, r_Body_vel_x,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_y, r_Body_vel_y,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_mass, r_Body_mass, sizeof(float)*kNumBodies, 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&host_draw_counter, r_draw_counter, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
}


int checksum() {
  transfer_data();
  int result = 0;

  for (int i = 0; i < kNumBodies; ++i) {
    int Body_checksum = static_cast<int>((host_Body_pos_x[i]*1000 + host_Body_pos_y[i]*2000
                        + host_Body_vel_x[i]*3000 + host_Body_vel_y[i]*4000)) % 123456;
    result += Body_checksum;
  }

  return result;
}


int main(int /*argc*/, char** /*argv*/) {
#ifdef OPTION_RENDER
  init_renderer();
#endif  // OPTION_RENDER

  // Allocate memory.
  IndexT* h_Body_merge_target;
  float* h_Body_pos_x;
  float* h_Body_pos_y;
  float* h_Body_vel_x;
  float* h_Body_vel_y;
  float* h_Body_force_x;
  float* h_Body_force_y;
  float* h_Body_mass;
  bool* h_Body_has_incoming_merge;
  bool* h_Body_successful_merge;
  bool* h_Body_is_active;

  cudaMalloc(&h_Body_merge_target, sizeof(IndexT)*kNumBodies);
  cudaMalloc(&h_Body_pos_x, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_pos_y, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_vel_x, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_vel_y, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_force_x, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_force_y, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_mass, sizeof(float)*kNumBodies);
  cudaMalloc(&h_Body_has_incoming_merge, sizeof(bool)*kNumBodies);
  cudaMalloc(&h_Body_successful_merge, sizeof(bool)*kNumBodies);
  cudaMalloc(&h_Body_is_active, sizeof(bool)*kNumBodies);

  cudaMemcpyToSymbol(dev_Body_merge_target, &h_Body_merge_target,
                     sizeof(IndexT*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_pos_x, &h_Body_pos_x,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_pos_y, &h_Body_pos_y,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_vel_x, &h_Body_vel_x,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_vel_y, &h_Body_vel_y,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_force_x, &h_Body_force_x,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_force_y, &h_Body_force_y,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_mass, &h_Body_mass,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_has_incoming_merge, &h_Body_has_incoming_merge,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_successful_merge, &h_Body_successful_merge,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_Body_is_active, &h_Body_is_active,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);

  // Allocate and create Body objects.
  kernel_initialize_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kIterations; ++i) {
    // printf("%i\n", i);
    parallel_do<&Body_compute_force><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    parallel_do<&Body_update><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    parallel_do<&Body_initialize_merge><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    parallel_do<&Body_prepare_merge><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    parallel_do<&Body_update_merge><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    parallel_do<&Body_delete_merged><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

#ifdef OPTION_RENDER
    // Transfer and render.
    transfer_data();
    draw(host_Body_pos_x, host_Body_pos_y, host_Body_mass,
         host_draw_counter);
#endif  // OPTION_RENDER
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
  printf("#bodies: %i\n", host_draw_counter);
#endif  // NDEBUG

  printf("%lu\n", micros);

  // Free memory
  cudaFree(h_Body_merge_target);
  cudaFree(h_Body_pos_x);
  cudaFree(h_Body_pos_y);
  cudaFree(h_Body_vel_x);
  cudaFree(h_Body_vel_y);
  cudaFree(h_Body_force_x);
  cudaFree(h_Body_force_y);
  cudaFree(h_Body_mass);
  cudaFree(h_Body_has_incoming_merge);
  cudaFree(h_Body_successful_merge);
  cudaFree(h_Body_is_active);

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
