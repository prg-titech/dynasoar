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

struct Body {
  IndexT merge_target;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float force_x;
  float force_y;
  float mass;
  bool has_incoming_merge;
  bool successful_merge;
  bool is_active;
};

__device__ Body* d_bodies;


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
  d_bodies[id].pos_x = pos_x;
  d_bodies[id].pos_y = pos_y;
  d_bodies[id].vel_x = vel_x;
  d_bodies[id].vel_y = vel_y;
  d_bodies[id].mass = mass;
  d_bodies[id].is_active = true;
}


__device__ void Body_apply_force(IndexT id, IndexT other) {
  // Update `other`.
  if (other != id) {
    float dx = d_bodies[id].pos_x - d_bodies[other].pos_x;
    float dy = d_bodies[id].pos_y - d_bodies[other].pos_y;
    float dist = sqrt(dx*dx + dy*dy);
    float F = kGravityConstant * d_bodies[id].mass * d_bodies[other].mass
        / (dist * dist + kDampeningFactor);
    d_bodies[other].force_x += F*dx / dist;
    d_bodies[other].force_y += F*dy / dist;
  }
}


__device__ void Body_compute_force(IndexT id) {
  d_bodies[id].force_x = 0.0f;
  d_bodies[id].force_y = 0.0f;

  // device_do
  for (IndexT i = 0; i < kNumBodies; ++i) {
    if (d_bodies[i].is_active) {
      Body_apply_force(i, id);
    }
  }
}


__device__ void Body_update(IndexT id) {
  d_bodies[id].vel_x += d_bodies[id].force_x*kTimeInterval / d_bodies[id].mass;
  d_bodies[id].vel_y += d_bodies[id].force_y*kTimeInterval / d_bodies[id].mass;
  d_bodies[id].pos_x += d_bodies[id].vel_x*kTimeInterval;
  d_bodies[id].pos_y += d_bodies[id].vel_y*kTimeInterval;

  if (d_bodies[id].pos_x < -1 || d_bodies[id].pos_x > 1) {
    d_bodies[id].vel_x = -d_bodies[id].vel_x;
  }

  if (d_bodies[id].pos_y < -1 || d_bodies[id].pos_y > 1) {
    d_bodies[id].vel_y = -d_bodies[id].vel_y;
  }
}


__device__ void Body_check_merge_into_this(IndexT id, IndexT other) {
  // Only merge into larger body.
  if (!d_bodies[other].has_incoming_merge
      && d_bodies[id].mass > d_bodies[other].mass) {
    float dx = d_bodies[id].pos_x - d_bodies[other].pos_x;
    float dy = d_bodies[id].pos_y - d_bodies[other].pos_y;
    float dist_square = dx*dx + dy*dy;

    if (dist_square < kMergeThreshold*kMergeThreshold) {
      // Try to merge this one.
      // There is a race condition here: Multiple threads may try to merge
      // this body.
      d_bodies[id].merge_target = other;
      d_bodies[other].has_incoming_merge = true;
    }
  }
}


__device__ void Body_initialize_merge(IndexT id) {
  d_bodies[id].merge_target = kNullptr;
  d_bodies[id].has_incoming_merge = false;
  d_bodies[id].successful_merge = false;
}


__device__ void Body_prepare_merge(IndexT id) {
  // device_do
  for (IndexT i = 0; i < kNumBodies; ++i) {
    if (d_bodies[i].is_active) {
      Body_check_merge_into_this(i, id);
    }
  }
}


__device__ void Body_update_merge(IndexT id) {
  IndexT m = d_bodies[id].merge_target;
  if (m != kNullptr) {
    if (d_bodies[m].merge_target == kNullptr) {
      // Perform merge.
      float new_mass = d_bodies[id].mass + d_bodies[m].mass;
      float new_vel_x = (d_bodies[id].vel_x*d_bodies[id].mass
                         + d_bodies[m].vel_x*d_bodies[m].mass) / new_mass;
      float new_vel_y = (d_bodies[id].vel_y*d_bodies[id].mass
                         + d_bodies[m].vel_y*d_bodies[m].mass) / new_mass;
      d_bodies[m].mass = new_mass;
      d_bodies[m].vel_x = new_vel_x;
      d_bodies[m].vel_y = new_vel_y;
      d_bodies[m].pos_x = (d_bodies[id].pos_x + d_bodies[m].pos_x) / 2;
      d_bodies[m].pos_y = (d_bodies[id].pos_y + d_bodies[m].pos_y) / 2;

      d_bodies[id].successful_merge = true;
    }
  }
}


__device__ void Body_delete_merged(IndexT id) {
  if (d_bodies[id].successful_merge) {
    d_bodies[id].is_active = false;
  }
}


__device__ void Body_add_to_draw_array(IndexT id) {
  int idx = atomicAdd(&r_draw_counter, 1);
  r_Body_pos_x[idx] = d_bodies[id].pos_x;
  r_Body_pos_y[idx] = d_bodies[id].pos_y;
  r_Body_vel_x[idx] = d_bodies[id].vel_x;
  r_Body_vel_y[idx] = d_bodies[id].vel_y;
  r_Body_mass[idx] = d_bodies[id].mass;
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
    if (d_bodies[id].is_active) {
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
  Body* h_bodies;

  cudaMalloc(&h_bodies, sizeof(Body)*kNumBodies);

  cudaMemcpyToSymbol(d_bodies, &h_bodies,
                     sizeof(Body*), 0, cudaMemcpyHostToDevice);

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
  cudaFree(h_bodies);

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
