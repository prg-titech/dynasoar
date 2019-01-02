#include <chrono>
#include <curand_kernel.h>

#include "dataset.h"
#include "rendering.h"
#include "structure.h"

// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ NodeBase::NodeBase(float pos_x, float pos_y)
    : pos_x_(pos_x), pos_y_(pos_y), num_springs_(0) {}


__device__ AnchorNode::AnchorNode(float pos_x, float pos_y)
    : NodeBase(pos_x, pos_y) {}


__device__ AnchorPullNode::AnchorPullNode(float pos_x, float pos_y,
                                          float vel_x, float vel_y)
    : AnchorNode(pos_x, pos_y), vel_x_(vel_x), vel_y_(vel_y) {}


__device__ Node::Node(float pos_x, float pos_y, float mass)
    : NodeBase(pos_x, pos_y), mass_(mass), vel_x_(0.0f), vel_y_(0.0f) {}


__device__ Spring::Spring(NodeBase* p1, NodeBase* p2, float spring_factor,
                          float max_force)
    : p1_(p1), p2_(p2), spring_factor_(spring_factor), force_(0.0f),
      max_force_(max_force), initial_length_(p1->distance_to(p2)) {
  assert(initial_length_ > 0.0f);
  p1_->add_spring(this);
  p2_->add_spring(this);
}


__device__ void NodeBase::add_spring(Spring* spring) {
  int idx = atomicAdd(&num_springs_, 1);
  springs_[idx] = spring;
  assert(idx + 1 <= kMaxDegree);
  assert(spring->p1() == this || spring->p2() == this);
}


__device__ void NodeBase::remove_spring(Spring* spring) {
  // TODO: This won't work if two springs break at the same time.

  int i = 0;
  Spring* s = nullptr;

  do {
    assert(i < kMaxDegree);
    s = springs_[i];
    ++i;
  } while(s != spring);

  for (; i < num_springs_; ++i) {
    springs_[i - 1] = springs_[i];
  }

  --num_springs_;

  if (num_springs_ == 0) {
    device_allocator->free<NodeBase>(this);
  }
}


__device__ float NodeBase::distance_to(NodeBase* other) const {
  float dx = pos_x_ - other->pos_x_;
  float dy = pos_y_ - other->pos_y_;
  float dist_sq = dx*dx + dy*dy;
  return sqrt(dist_sq);
}


__device__ void AnchorPullNode::pull() {
  pos_x_ += vel_x_ * kDt;
  pos_y_ += vel_y_ * kDt;
}


__device__ void Spring::compute_force() {
  float dist = p1_->distance_to(p2_);
  float displacement = max(0.0f, dist - initial_length_);
  force_ = spring_factor_ * displacement;

  if (force_ > max_force_) {
    p1_->remove_spring(this);
    p2_->remove_spring(this);
    device_allocator->free<Spring>(this);
  }
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
  vel_x_ *= 1.0f - kVelocityDampening;
  vel_y_ *= 1.0f - kVelocityDampening;
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


float checksum() {
  transfer_data();
  float result = 0.0f;

  for (int i = 0; i < host_num_springs; ++i) {
    result += host_spring_info[i].p1_x*host_spring_info[i].p2_y
              *host_spring_info[i].force;
  }

  return result;
}


void compute() {
  allocator_handle->parallel_do<Spring, &Spring::compute_force>();
  allocator_handle->parallel_do<Node, &Node::move>();
}


void step() {
  allocator_handle->parallel_do<AnchorPullNode, &AnchorPullNode::pull>();

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }

  if (kOptionRender) {
    transfer_data();
    draw(host_num_springs, host_spring_info);
  }
}


__device__ NodeBase* tmp_nodes[kMaxNodes];

__global__ void kernel_create_nodes(DsNode* nodes, int num_nodes) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_nodes; i += blockDim.x * gridDim.x) {
    if (nodes[i].type == kTypeNode) {
      tmp_nodes[i] = device_allocator->make_new<Node>(nodes[i].pos_x,
                                                      nodes[i].pos_y,
                                                      nodes[i].mass);
    } else if (nodes[i].type == kTypeAnchorPullNode) {
      tmp_nodes[i] = device_allocator->make_new<AnchorPullNode>(nodes[i].pos_x,
                                                                nodes[i].pos_y,
                                                                nodes[i].vel_x,
                                                                nodes[i].vel_y);
    } else {
      assert(false);
    }
  }
}


__global__ void kernel_create_springs(DsSpring* springs, int num_springs) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_springs; i += blockDim.x * gridDim.x) {
    assert(tmp_nodes[springs[i].p1] != nullptr);
    assert(tmp_nodes[springs[i].p2] != nullptr);

    device_allocator->make_new<Spring>(tmp_nodes[springs[i].p1],
                                       tmp_nodes[springs[i].p2],
                                       springs[i].spring_factor,
                                       springs[i].max_force);
  }
}


void load_dataset(Dataset& dataset) {
  DsNode* host_nodes;
  cudaMalloc(&host_nodes, sizeof(DsNode)*dataset.nodes.size());
  cudaMemcpy(host_nodes, dataset.nodes.data(),
             sizeof(DsNode)*dataset.nodes.size(), cudaMemcpyHostToDevice);

  DsSpring* host_springs;
  cudaMalloc(&host_springs, sizeof(DsSpring)*dataset.springs.size());
  cudaMemcpy(host_springs, dataset.springs.data(),
             sizeof(DsSpring)*dataset.springs.size(), cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_nodes<<<128, 128>>>(host_nodes, dataset.nodes.size());
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_springs<<<128, 128>>>(host_springs, dataset.springs.size());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(host_nodes);
  cudaFree(host_springs);
}


__global__ void load_example() {
  assert(threadIdx.x == 0 && blockIdx.x == 0);

  float spring_factor = 5.0f;
  float max_force = 100.0f;
  float mass = 500.0f;

  auto* a1 = device_allocator->make_new<AnchorPullNode>(0.1, 0.5, 0.0, -0.02);
  auto* a2 = device_allocator->make_new<AnchorPullNode>(0.3, 0.5, 0.0, -0.02);
  auto* a3 = device_allocator->make_new<AnchorPullNode>(0.5, 0.5, 0.0, -0.02);

  auto* n1 = device_allocator->make_new<Node>(0.05, 0.6, mass);
  auto* n2 = device_allocator->make_new<Node>(0.3, 0.6, mass);
  auto* n3 = device_allocator->make_new<Node>(0.7, 0.6, mass);

  auto* n4 = device_allocator->make_new<Node>(0.2, 0.7, mass);
  auto* n5 = device_allocator->make_new<Node>(0.4, 0.7, mass);
  auto* n6 = device_allocator->make_new<Node>(0.8, 0.7, mass);

  auto* a4 = device_allocator->make_new<AnchorNode>(0.1, 0.9);
  auto* a5 = device_allocator->make_new<AnchorNode>(0.3, 0.9);
  auto* a6 = device_allocator->make_new<AnchorNode>(0.6, 0.9);

  device_allocator->make_new<Spring>(a1, n1, spring_factor, max_force);
  device_allocator->make_new<Spring>(a2, n2, spring_factor, max_force);
  device_allocator->make_new<Spring>(a3, n3, spring_factor, max_force);

  device_allocator->make_new<Spring>(n1, n4, spring_factor, max_force);
  device_allocator->make_new<Spring>(n2, n5, spring_factor, max_force);
  device_allocator->make_new<Spring>(n3, n6, spring_factor, max_force);
  device_allocator->make_new<Spring>(n2, n6, spring_factor, max_force);

  device_allocator->make_new<Spring>(n4, a4, spring_factor, max_force);
  device_allocator->make_new<Spring>(n5, a5, spring_factor, max_force);
  device_allocator->make_new<Spring>(n6, a6, spring_factor, max_force);
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

  //load_example<<<1, 1>>>();
  
  Dataset dataset;
  random_dataset(dataset);
  load_dataset(dataset);

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumSteps; ++i) {
    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("%lu\n", millis);

#ifndef NDEBUG
  printf("Checksum: %f\n", checksum());
#endif  // NDEBUG

  if (kOptionRender) {
    close_renderer();
  }
}
