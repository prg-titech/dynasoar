#include <chrono>

#include "example/nbody/soa/configuration.h"
#include "example/nbody/soa/nbody.h"
#include "example/nbody/soa/rendering.h"

namespace nbody {

__device__ AllocatorT* device_allocator;
__device__ double device_checksum;

// Helper variables for drawing.
__device__ int draw_counter = 0;
__device__ float Body_pos_x[kNumBodies];
__device__ float Body_pos_y[kNumBodies];
__device__ float Body_mass[kNumBodies];

// Host side pointer.
AllocatorHandle<AllocatorT>* allocator_handle;


__DEV__ Body::Body(float pos_x, float pos_y,
                   float vel_x, float vel_y, float mass)
    : pos_x_(pos_x), pos_y_(pos_y),
      vel_x_(vel_x), vel_y_(vel_y), mass_(mass) {}


__DEV__ void Body::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;
  device_allocator->template device_do<Body>(&Body::apply_force, this);
}


__DEV__ void Body::apply_force(Body* other) {
  // Update `other`.
  if (other != this) {
    float dx = pos_x_ - other->pos_x_;
    float dy = pos_y_ - other->pos_y_;
    float dist = sqrt(dx*dx + dy*dy);
    float F = kGravityConstant * mass_ * other->mass_
        / (dist * dist + kDampeningFactor);
    other->force_x_ += F*dx / dist;
    other->force_y_ += F*dy / dist;
  }
}


__DEV__ void Body::update() {
  vel_x_ += force_x_*kDt / mass_;
  vel_y_ += force_y_*kDt / mass_;
  pos_x_ += vel_x_*kDt;
  pos_y_ += vel_y_*kDt;

  if (pos_x_ < -1 || pos_x_ > 1) {
    vel_x_ = -vel_x_;
  }

  if (pos_y_ < -1 || pos_y_ > 1) {
    vel_y_ = -vel_y_;
  }
}


__DEV__ void Body::add_checksum() {
  device_checksum += pos_x_ + pos_y_*2 + vel_x_*3 + vel_y_*4;
}


__DEV__ void Body::add_to_draw_array() {
  int idx = atomicAdd(&draw_counter, 1);
  Body_pos_x[idx] = pos_x_;
  Body_pos_y[idx] = pos_y_;
  Body_mass[idx] = mass_;
}


__global__ void kernel_compute_checksum() {
  device_checksum = 0.0f;
  device_allocator->template device_do<Body>(&Body::add_checksum);
}


__global__ void kernel_initialize_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
    device_allocator->make_new<Body>(
        /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
        /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
        /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


__global__ void kernel_reset_draw_counters() {
  draw_counter = 0;
}


int main(int argc, char** argv) {
  if (OPTION_DRAW) {
    init_renderer();
  }

  // Host-side variables for rendering.
  float host_Body_pos_x[kNumBodies];
  float host_Body_pos_y[kNumBodies];
  float host_Body_mass[kNumBodies];

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  auto time_start = std::chrono::system_clock::now();

  kernel_initialize_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumIterations; ++i) {
    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();

    if (OPTION_DRAW) {
      kernel_reset_draw_counters<<<1, 1>>>();
      gpuErrchk(cudaDeviceSynchronize());
      allocator_handle->parallel_do<Body, &Body::add_to_draw_array>();
      gpuErrchk(cudaDeviceSynchronize());

      cudaMemcpyFromSymbol(host_Body_pos_x, Body_pos_x,
                           sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
      cudaMemcpyFromSymbol(host_Body_pos_y, Body_pos_y, sizeof(float)*kNumBodies, 0,
                 cudaMemcpyDeviceToHost);
      cudaMemcpyFromSymbol(host_Body_mass, Body_mass, sizeof(float)*kNumBodies, 0,
                 cudaMemcpyDeviceToHost);
      draw(host_Body_pos_x, host_Body_pos_y, host_Body_mass);
    }
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("Time: %lu ms\n", millis);

  kernel_compute_checksum<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  double checksum;
  cudaMemcpyFromSymbol(&checksum, device_checksum, sizeof(device_checksum), 0,
                       cudaMemcpyDeviceToHost);
  printf("Checksum: %f\n", checksum);


  if (OPTION_DRAW) {
    close_renderer();
  }

  return 0;
}

}

int main(int argc, char** argv) { return nbody::main(argc, argv); }
