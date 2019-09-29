#include <chrono>

#include "nbody.h"

#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Helper variable for checksum computation.
float host_checksum;


__device__ Body::Body(float pos_x, float pos_y,
                      float vel_x, float vel_y, float mass)
    : pos_x_(pos_x), pos_y_(pos_y),
      vel_x_(vel_x), vel_y_(vel_y), mass_(mass) {}


__device__ void Body::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;
  device_allocator->template device_do<Body>(&Body::apply_force, this);
}


__device__ void Body::apply_force(Body* other) {
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


__device__ void Body::update() {
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


void Body::add_checksum() {
  host_checksum += pos_x_ + pos_y_*2 + vel_x_*3 + vel_y_*4;
}


__global__ void kernel_initialize_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
    new(device_allocator) Body(
        /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
        /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
        /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


#ifdef OPTION_RENDER
void render_frame() {
  init_frame();

  allocator_handle->template device_do<Body>([&](Body* body) {
    draw_body(body->pos_x(), body->pos_y(), body->mass());
  });

  show_frame();
}
#endif  // OPTION_RENDER


int main(int /*argc*/, char** /*argv*/) {
#ifdef OPTION_RENDER
  init_renderer();
#endif  // OPTION_RENDER

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  auto time_start = std::chrono::system_clock::now();

  kernel_initialize_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumIterations; ++i) {
#ifndef NDEBUG
    // Print debug information.
    allocator_handle->DBG_print_state_stats();
#endif  // NDEBUG

    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();

#ifdef OPTION_RENDER
    render_frame();
#endif  // OPTION_RENDER
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

  printf("%lu, %lu\n", micros, allocator_handle->DBG_get_enumeration_time());

#ifndef NDEBUG
  host_checksum = 0.0f;
  allocator_handle->template device_do<Body>(&Body::add_checksum);
  printf("Checksum: %f\n", host_checksum);
#endif  // NDEBUG

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}

