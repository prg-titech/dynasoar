#include <chrono>
#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER


// Pre-declare all classes.
class Body;

using AllocatorT = SoaAllocator<kNumObjects, Body>;

class Body : public AllocatorT::Base {
 public:
  declare_field_types(
      Body,
      Body*,          // merge_target_
      float,          // pos_x_
      float,          // pos_y_,
      float,          // vel_x_,
      float,          // vel_y_,
      float,          // force_x_,
      float,          // force_y_,
      float,          // mass_
      bool,           // has_incoming_merge_
      bool)           // successful_merge_

 private:
  Field<Body, 0> merge_target_;
  Field<Body, 1> pos_x_;
  Field<Body, 2> pos_y_;
  Field<Body, 3> vel_x_;
  Field<Body, 4> vel_y_;
  Field<Body, 5> force_x_;
  Field<Body, 6> force_y_;
  Field<Body, 7> mass_;
  Field<Body, 8> has_incoming_merge_;
  Field<Body, 9> successful_merge_;

 public:
  __DEV__ Body(float pos_x, float pos_y, float vel_x, float vel_y, float mass);

  __DEV__ void compute_force();

  __DEV__ void apply_force(Body* other);

  __DEV__ void update();

  __DEV__ void check_merge_into_this(Body* other);

  __DEV__ void initialize_merge();

  __DEV__ void prepare_merge();

  __DEV__ void update_merge();

  __DEV__ void delete_merged();

  // Only for rendering purposes.
  __DEV__ void add_to_draw_array();
};


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Helper variables for rendering and checksum computation.
__device__ int draw_counter = 0;
__device__ float Body_pos_x[kNumBodies];
__device__ float Body_pos_y[kNumBodies];
__device__ float Body_vel_x[kNumBodies];
__device__ float Body_vel_y[kNumBodies];
__device__ float Body_mass[kNumBodies];
int host_draw_counter;
float host_Body_pos_x[kNumBodies];
float host_Body_pos_y[kNumBodies];
float host_Body_vel_x[kNumBodies];
float host_Body_vel_y[kNumBodies];
float host_Body_mass[kNumBodies];


__DEV__ Body::Body(float pos_x, float pos_y,
                   float vel_x, float vel_y, float mass)
    : pos_x_(pos_x), pos_y_(pos_y),
      vel_x_(vel_x), vel_y_(vel_y), mass_(mass) {}


__DEV__ void Body::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;
  device_allocator->device_do<Body>(&Body::apply_force, this);
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
  vel_x_ += force_x_*kTimeInterval / mass_;
  vel_y_ += force_y_*kTimeInterval / mass_;
  pos_x_ += vel_x_*kTimeInterval;
  pos_y_ += vel_y_*kTimeInterval;

  if (pos_x_ < -1 || pos_x_ > 1) {
    vel_x_ = -vel_x_;
  }

  if (pos_y_ < -1 || pos_y_ > 1) {
    vel_y_ = -vel_y_;
  }
}


__DEV__ void Body::check_merge_into_this(Body* other) {
  // Only merge into larger body.
  if (!other->has_incoming_merge_ && mass_ > other->mass_) {
    float dx = pos_x_ - other->pos_x_;
    float dy = pos_y_ - other->pos_y_;
    float dist_square = dx*dx + dy*dy;

    if (dist_square < kMergeThreshold*kMergeThreshold) {
      // Try to merge this one.
      // There is a race condition here: Multiple threads may try to merge
      // this body. Only one can win. That's OK.
      this->merge_target_ = other;
      other->has_incoming_merge_ = true;
    }
  }
}


__DEV__ void Body::initialize_merge() {
  merge_target_ = nullptr;
  has_incoming_merge_ = false;
  successful_merge_ = false;
}


__DEV__ void Body::prepare_merge() {
  device_allocator->template device_do<Body>(&Body::check_merge_into_this,
                                             this);
}


__DEV__ void Body::update_merge() {
  Body* m = merge_target_;
  if (m != nullptr) {
    if (m->merge_target_ == nullptr) {
      // Perform merge.
      float new_mass = mass_ + m->mass_;
      float new_vel_x = (vel_x_*mass_ + m->vel_x_*m->mass_) / new_mass;
      float new_vel_y = (vel_y_*mass_ + m->vel_y_*m->mass_) / new_mass;
      m->mass_ = new_mass;
      m->vel_x_ = new_vel_x;
      m->vel_y_ = new_vel_y;
      m->pos_x_ = (pos_x_ + m->pos_x_) / 2;
      m->pos_y_ = (pos_y_ + m->pos_y_) / 2;

      successful_merge_ = true;
    }
  }
}


__DEV__ void Body::delete_merged() {
  if (successful_merge_) {
    destroy(device_allocator, this);
  }
}


__DEV__ void Body::add_to_draw_array() {
  int idx = atomicAdd(&draw_counter, 1);
  Body_pos_x[idx] = pos_x_;
  Body_pos_y[idx] = pos_y_;
  Body_vel_x[idx] = vel_x_;
  Body_vel_y[idx] = vel_y_;
  Body_mass[idx] = mass_;
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


__global__ void kernel_reset_draw_counters() {
  draw_counter = 0;
}


void transfer_data() {
  // Extract data from SoaAlloc data structure.
  kernel_reset_draw_counters<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  allocator_handle->parallel_do<Body, &Body::add_to_draw_array>();
  gpuErrchk(cudaDeviceSynchronize());

  // Copy data to host.
  cudaMemcpyFromSymbol(host_Body_pos_x, Body_pos_x,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_pos_y, Body_pos_y,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_x, Body_vel_x,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_vel_y, Body_vel_y,
                       sizeof(float)*kNumBodies, 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(host_Body_mass, Body_mass, sizeof(float)*kNumBodies, 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&host_draw_counter, draw_counter, sizeof(int), 0,
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

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  // Allocate and create Body objects.
  kernel_initialize_bodies<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kIterations; ++i) {
#ifndef NDEBUG
    // Print debug information.
    allocator_handle->DBG_print_state_stats();
#endif  // NDEBUG

    allocator_handle->parallel_do<Body, &Body::compute_force>();
    allocator_handle->parallel_do<Body, &Body::update>();
    allocator_handle->parallel_do<Body, &Body::initialize_merge>();
    allocator_handle->parallel_do<Body, &Body::prepare_merge>();
    allocator_handle->parallel_do<Body, &Body::update_merge>();
    allocator_handle->parallel_do<Body, &Body::delete_merged>();

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

  printf("%lu, %lu\n", micros, allocator_handle->DBG_get_enumeration_time());

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
