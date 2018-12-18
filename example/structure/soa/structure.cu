#include <chrono>

#include "rendering.h"
#include "structure.h"

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ NodeBase::NodeBase(float pos_x, float pos_y)
    : pos_x_(pos_x), pos_y_(pos_y), num_springs_(0) {}


__device__ AnchorNode::AnchorNode(float pos_x, float pos_y)
    : NodeBase(pos_x, pos_y) {}


__device__ Node::Node(float pos_x, float pos_y)
    : NodeBase(pos_x, pos_y) {}


__device__ Spring::Spring(NodeBase* p1, NodeBase* p2, float spring_factor,
                          float max_force)
    : p1_(p1), p2_(p2), spring_factor_(spring_factor), force_(0.0f),
      max_force_(max_force), initial_length_(p1->distance_to(p2)) {
  assert(initial_length_ > 0.0f);
}


__device__ float NodeBase::distance_to(NodeBase* other) const {
  float dx = pos_x_ - other->pos_x_;
  float dy = pos_y_ - other->pos_y_;
  float dist_sq = dx*dx + dy*dy;
  return sqrt(dist_sq);
}


__device__ void AnchorNode::pull() {
  pos_x_ += kPullX;
  pos_y_ += kPullY;
}


__device__ void Spring::compute_force() {
  float dist = p1_->distance_to(p2_);
  float displacement = max(0.0f, dist - initial_length_);
  force_ = spring_factor_ * displacement;
}


__device__ void Node::move() {
  float force_x = 0.0f;
  float force_y = 0.0f;

  for (int i = 0; i < num_springs_; ++i) {
    Spring* s = springs_[i];
    NodeBase* from;
    NodeBase* to;

    if (s->p1() == this) {
      from = this;
      to = s->p2();
    } else {
      assert(s->p2() == this);
      from = this;
      to = s->p1();
    }

    // Calculate unit vector.
    float dx = to->pos_x() - from->pos_x();
    float dy = to->pos_y() - from->pos_y();
    float dist = sqrt(dx*dx + dy*dy);
    float unit_x = dx/dist;
    float unit_y = dy/dist;

    // Apply force.
    force_x += unit_x*s->force();
    force_y += unit_y*s->force();
  }

  // Calculate new velocity and position.
  vel_x_ += force_x*kDt / mass_;
  vel_y_ += force_y*kDt / mass_;
  pos_x_ += vel_x_*kDt;
  pos_y_ += vel_y_*kDt;
}


// Only for rendering.
__device__ int dev_num_springs;
__device__ SpringInfo dev_spring_info[kMaxSprings];
int host_num_springs;
SpringInfo host_spring_info[kMaxSprings];

__device__ void Spring::add_to_rendering_array() {
  int idx = atomicAdd(&dev_num_springs, 1);
  dev_spring_info[idx].p1_x = p1_->pos_x();
  dev_spring_info[idx].p1_y = p1_->pos_y();
  dev_spring_info[idx].p2_x = p2_->pos_x();
  dev_spring_info[idx].p2_y = p2_->pos_y();
  dev_spring_info[idx].force = force_;
  dev_spring_info[idx].max_force = max_force_;
}


void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_springs, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<Spring, &Spring::add_to_rendering_array>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&host_num_springs, dev_num_springs, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(host_spring_info, dev_spring_info,
                       sizeof(SpringInfo)*host_num_springs, 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}


void compute() {
  allocator_handle->parallel_do<Spring, &Spring::compute_force>();
  allocator_handle->parallel_do<Node, &Node::move>();
}


void step() {
  allocator_handle->parallel_do<AnchorNode, &AnchorNode::pull>();

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }

  if (kOptionRender) {
    transfer_data();
    draw(host_num_springs, host_spring_info);
  }
}


int main(int /*argc*/, char** /*argv*/) {
  if (kOptionRender) {
    init_renderer();
  }

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  for (int i = 0; i < kNumSteps; ++i) {
    step();
  }

  if (kOptionRender) {
    close_renderer();
  }
}
