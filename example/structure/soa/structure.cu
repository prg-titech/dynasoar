#include <chrono>
#include <curand_kernel.h>

#include "../dataset.h"
#include "../rendering.h"
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
      max_force_(max_force), initial_length_(p1->distance_to(p2)),
      delete_flag_(0) {
  assert(initial_length_ > 0.0f);
  p1_->add_spring(this);
  p2_->add_spring(this);
}


// Only used during graph creation.
__device__ void NodeBase::add_spring(Spring* spring) {
  int idx = atomicAdd(&num_springs_, 1);
  springs_[idx] = spring;
  assert(idx + 1 <= kMaxDegree);
  assert(spring->p1() == this || spring->p2() == this);
}


__device__ void NodeBase::remove_spring(Spring* spring) {
  for (int i = 0; i < kMaxDegree; ++i) {
    if (springs_[i] == spring) {
      springs_[i] = nullptr;
      if (atomicSub(&num_springs_, 1) == 1) {
        // Deleted last spring.
        destroy(device_allocator, this);
      }
      return;
    }
  }

  // Spring not found.
  assert(false);
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

  if (force_ > max_force_) { self_destruct(); }
}


__device__ void Spring::self_destruct() {
  p1_->remove_spring(this);
  p2_->remove_spring(this);
  destroy(device_allocator, this);
}


__device__ void Node::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;

  for (int i = 0; i < kMaxDegree; ++i) {
    Spring* s = springs_[i];
    if (s != nullptr) {
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
      force_x_ += unit_x*s->force();
      force_y_ += unit_y*s->force();
    }
  }
}


__device__ void Node::move() {
  // Calculate new velocity and posFition.
  vel_x_ += force_x_*kDt / mass_;
  vel_y_ += force_y_*kDt / mass_;
  vel_x_ *= 1.0f - kVelocityDampening;
  vel_y_ *= 1.0f - kVelocityDampening;
  pos_x_ += vel_x_*kDt;
  pos_y_ += vel_y_*kDt;
}


__device__ void NodeBase::initialize_bfs() {
  if (this->cast<AnchorNode>() != nullptr) {
    distance_ = 0;
  } else {
    distance_ = kMaxDistance;  // should be int_max
  }
}


__device__ bool dev_bfs_continue;

__device__ void NodeBase::bfs_visit(int distance) {
  if (distance == distance_) {
    // Continue until all vertices were visited.
    dev_bfs_continue = true;

    for (int i = 0; i < kMaxDegree; ++i) {
      auto* spring = springs_[i];

      if (spring != nullptr) {
        // Find neighboring vertices.
        NodeBase* n;
        if (this == spring->p1()) {
          n = spring->p2();
        } else {
          n = spring->p1();
        }

        if (n->distance_ == kMaxDistance) {
          // Set distance on neighboring vertex if unvisited.
          n->distance_ = distance + 1;
        }
      }
    }
  }
}


__device__ void NodeBase::bfs_set_delete_flags() {
  if (distance_ == kMaxDistance) {  // should be int_max
    for (int i = 0; i < kMaxDegree; ++i) {
      auto* spring = springs_[i];
      if (spring != nullptr) {
        spring->set_delete_flag();
      }
    }
  }
}


__device__ void Spring::bfs_delete() {
  if (delete_flag_ == 1) { self_destruct(); }
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
  allocator_handle->parallel_do<Node, &Node::compute_force>();
  allocator_handle->parallel_do<Node, &Node::move>();
}


void bfs_and_delete() {
  // Perform BFS to check reachability.
  allocator_handle->parallel_do<NodeBase, &NodeBase::initialize_bfs>();

  for (int i = 0; i < kMaxDistance; ++i) {
    bool continue_flag = false;
    cudaMemcpyToSymbol(dev_bfs_continue, &continue_flag, sizeof(bool), 0,
                       cudaMemcpyHostToDevice);
    allocator_handle->parallel_do<NodeBase, int, &NodeBase::bfs_visit>(i);
    cudaMemcpyFromSymbol(&continue_flag, dev_bfs_continue, sizeof(bool), 0,
                         cudaMemcpyDeviceToHost);

    if (!continue_flag) break;
  }

  // Delete springs (and nodes).
  allocator_handle->parallel_do<NodeBase, &NodeBase::bfs_set_delete_flags>();
  allocator_handle->parallel_do<Spring, &Spring::bfs_delete>();
}


#ifdef OPTION_DEFRAG
void defrag() {
  allocator_handle->parallel_defrag<AnchorNode>(1);
  allocator_handle->parallel_defrag<AnchorPullNode>(1);
  allocator_handle->parallel_defrag<Node>(1);
  allocator_handle->parallel_defrag<Spring>(1);
}
#endif  // OPTION_DEFRAG


void step() {
  allocator_handle->parallel_do<AnchorPullNode, &AnchorPullNode::pull>();

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }

  bfs_and_delete();

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
      tmp_nodes[i] = new(device_allocator) Node(nodes[i].pos_x,
                                                nodes[i].pos_y,
                                                nodes[i].mass);
    } else if (nodes[i].type == kTypeAnchorPullNode) {
      tmp_nodes[i] = new(device_allocator) AnchorPullNode(nodes[i].pos_x,
                                                          nodes[i].pos_y,
                                                          nodes[i].vel_x,
                                                          nodes[i].vel_y);
    } else if (nodes[i].type == kTypeAnchorNode) {
      tmp_nodes[i] = new(device_allocator) AnchorNode(nodes[i].pos_x,
                                                      nodes[i].pos_y);
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

    new(device_allocator) Spring(tmp_nodes[springs[i].p1],
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

  auto* a1 = new(device_allocator) AnchorPullNode(0.1, 0.5, 0.0, -0.02);
  auto* a2 = new(device_allocator) AnchorPullNode(0.3, 0.5, 0.0, -0.02);
  auto* a3 = new(device_allocator) AnchorPullNode(0.5, 0.5, 0.0, -0.02);

  auto* n1 = new(device_allocator) Node(0.05, 0.6, mass);
  auto* n2 = new(device_allocator) Node(0.3, 0.6, mass);
  auto* n3 = new(device_allocator) Node(0.7, 0.6, mass);

  auto* n4 = new(device_allocator) Node(0.2, 0.7, mass);
  auto* n5 = new(device_allocator) Node(0.4, 0.7, mass);
  auto* n6 = new(device_allocator) Node(0.8, 0.7, mass);

  auto* a4 = new(device_allocator) AnchorNode(0.1, 0.9);
  auto* a5 = new(device_allocator) AnchorNode(0.3, 0.9);
  auto* a6 = new(device_allocator) AnchorNode(0.6, 0.9);

  new(device_allocator) Spring(a1, n1, spring_factor, max_force);
  new(device_allocator) Spring(a2, n2, spring_factor, max_force);
  new(device_allocator) Spring(a3, n3, spring_factor, max_force);

  new(device_allocator) Spring(n1, n4, spring_factor, max_force);
  new(device_allocator) Spring(n2, n5, spring_factor, max_force);
  new(device_allocator) Spring(n3, n6, spring_factor, max_force);
  new(device_allocator) Spring(n2, n6, spring_factor, max_force);

  new(device_allocator) Spring(n4, a4, spring_factor, max_force);
  new(device_allocator) Spring(n5, a5, spring_factor, max_force);
  new(device_allocator) Spring(n6, a6, spring_factor, max_force);
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

//  allocator_handle->DBG_print_state_stats();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumSteps; ++i) {
   // printf("%i\n", i);

#ifdef OPTION_DEFRAG
    if (kOptionDefrag && i % 20 == 0) {
      // allocator_handle->DBG_print_state_stats();
      defrag();
    }
#endif  // OPTION_DEFRAG

/*
    int a_anchornode = dev_ptr->DBG_host_allocated_slots<AnchorNode>();
    int u_anchornode = dev_ptr->DBG_host_used_slots<AnchorNode>();
    int a_anchorpullnode = dev_ptr->DBG_host_allocated_slots<AnchorPullNode>();
    int u_anchorpullnode = dev_ptr->DBG_host_used_slots<AnchorPullNode>();
    int a_node = dev_ptr->DBG_host_allocated_slots<Node>();
    int u_node = dev_ptr->DBG_host_used_slots<Node>();
    int a_spring = dev_ptr->DBG_host_allocated_slots<Spring>();
    int u_spring = dev_ptr->DBG_host_used_slots<Spring>();
    printf("%i, %i, %i, %i, %i, %i, %i, %i, %i\n",
           i, a_anchornode, u_anchornode, a_anchorpullnode, u_anchorpullnode,
           a_node, u_node, a_spring, u_spring);
*/

    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("%lu,%lu\n", millis, allocator_handle->DBG_get_enumeration_time());

  allocator_handle->DBG_print_defrag_time();

  if (kOptionPrintStats) {
    //allocator_handle->DBG_print_state_stats();
  }

#ifndef NDEBUG
  printf("Checksum: %f\n", checksum());
#endif  // NDEBUG

  if (kOptionRender) {
    close_renderer();
  }
}
