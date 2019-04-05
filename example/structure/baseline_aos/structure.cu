#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>

#include "../configuration.h"
#include "../dataset.h"
#include "util/util.h"


static const int kThreads = 256;
static const int kNullptr = std::numeric_limits<int>::max();

using IndexT = int;

struct Node {
  DeviceArray<IndexT, kMaxDegree> springs;
  int num_springs;
  int distance;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float mass;
  char type;
};

struct Spring {
  IndexT p1;
  IndexT p2;
  float factor;
  float initial_length;
  float force;
  float max_force;
  bool is_active;
  bool delete_flag;
};

__device__ Node* dev_nodes;
__device__ Spring* dev_springs;


__device__ void new_NodeBase(IndexT id, float pos_x, float pos_y) {
  dev_nodes[id].pos_x = pos_x;
  dev_nodes[id].pos_y = pos_y;
  dev_nodes[id].num_springs = 0;
  dev_nodes[id].type = kTypeNodeBase;

  for (int i = 0; i < kMaxDegree; ++i) {
    dev_nodes[id].springs[i] = kNullptr;
  }
}


__device__ void new_AnchorNode(IndexT id, float pos_x, float pos_y) {
  new_NodeBase(id, pos_x, pos_y);
  dev_nodes[id].type = kTypeAnchorNode;
}


__device__ void new_AnchorPullNode(IndexT id, float pos_x, float pos_y,
                                   float vel_x, float vel_y) {
  new_AnchorNode(id, pos_x, pos_y);
  dev_nodes[id].vel_x = vel_x;
  dev_nodes[id].vel_y = vel_y;
  dev_nodes[id].type = kTypeAnchorPullNode;
}


__device__ void new_Node(IndexT id, float pos_x, float pos_y, float mass) {
  new_NodeBase(id, pos_x, pos_y);
  dev_nodes[id].mass = mass;
  dev_nodes[id].type = kTypeNode;
}


__device__ float NodeBase_distance_to(IndexT id, IndexT other) {
  float dx = dev_nodes[id].pos_x - dev_nodes[other].pos_x;
  float dy = dev_nodes[id].pos_y - dev_nodes[other].pos_y;
  float dist_sq = dx*dx + dy*dy;
  return sqrt(dist_sq);
}


__device__ void NodeBase_add_spring(IndexT id, IndexT spring) {
  assert(id >= 0 && id < kMaxNodes);

  int idx = atomicAdd(&dev_nodes[id].num_springs, 1);
  assert(idx + 1 <= kMaxDegree);
  dev_nodes[id].springs[idx] = spring;

  assert(dev_springs[spring].p1 == id || dev_springs[spring].p2 == id);
}


__device__ void new_Spring(IndexT id, IndexT p1, IndexT p2,
                           float spring_factor, float max_force) {
  dev_springs[id].is_active = true;
  dev_springs[id].p1 = p1;
  dev_springs[id].p2 = p2;
  dev_springs[id].factor = spring_factor;
  dev_springs[id].force = 0.0f;
  dev_springs[id].max_force = max_force;
  dev_springs[id].initial_length = NodeBase_distance_to(p1, p2);
  dev_springs[id].delete_flag = false;
  assert(dev_springs[id].initial_length > 0.0f);

  NodeBase_add_spring(p1, id);
  NodeBase_add_spring(p2, id);
}


__device__ void NodeBase_remove_spring(IndexT id, IndexT spring) {
  for (int i = 0; i < kMaxDegree; ++i) {
    if (dev_nodes[id].springs[i] == spring) {
      dev_nodes[id].springs[i] = kNullptr;
      if (atomicSub(&dev_nodes[id].num_springs, 1) == 1) {
        // Deleted last spring.
        dev_nodes[id].type = 0;
      }
      return;
    }
  }

  // Spring not found.
  assert(false);
}


__device__ void AnchorPullNode_pull(IndexT id) {
  dev_nodes[id].pos_x += dev_nodes[id].vel_x * kDt;
  dev_nodes[id].pos_y += dev_nodes[id].vel_y * kDt;
}


__device__ void Spring_self_destruct(IndexT id) {
  NodeBase_remove_spring(dev_springs[id].p1, id);
  NodeBase_remove_spring(dev_springs[id].p2, id);
  dev_springs[id].is_active = false;
}


__device__ void Spring_compute_force(IndexT id) {
  float dist = NodeBase_distance_to(dev_springs[id].p1, dev_springs[id].p2);
  float displacement = max(0.0f, dist - dev_springs[id].initial_length);
  dev_springs[id].force = dev_springs[id].factor * displacement;

  if (dev_springs[id].force > dev_springs[id].max_force) {
    Spring_self_destruct(id);
  }
}


__device__ void Node_move(IndexT id) {
  float force_x = 0.0f;
  float force_y = 0.0f;

  for (int i = 0; i < kMaxDegree; ++i) {
    IndexT s = dev_nodes[id].springs[i];

    if (s != kNullptr) {
      IndexT from;
      IndexT to;

      if (dev_springs[s].p1 == id) {
        from = id;
        to = dev_springs[s].p2;
      } else {
        assert(dev_springs[s].p2 == id);
        from = id;
        to = dev_springs[s].p1;
      }

      // Calculate unit vector.
      float dx = dev_nodes[to].pos_x - dev_nodes[from].pos_x;
      float dy = dev_nodes[to].pos_y - dev_nodes[from].pos_y;
      float dist = sqrt(dx*dx + dy*dy);
      float unit_x = dx/dist;
      float unit_y = dy/dist;

      // Apply force.
      force_x += unit_x*dev_springs[s].force;
      force_y += unit_y*dev_springs[s].force;
    }
  }

  // Calculate new velocity and position.
  dev_nodes[id].vel_x += force_x*kDt / dev_nodes[id].mass;
  dev_nodes[id].vel_y += force_y*kDt / dev_nodes[id].mass;
  dev_nodes[id].vel_x *= 1.0f - kVelocityDampening;
  dev_nodes[id].vel_y *= 1.0f - kVelocityDampening;
  dev_nodes[id].pos_x += dev_nodes[id].vel_x*kDt;
  dev_nodes[id].pos_y += dev_nodes[id].vel_y*kDt;
}


__device__ void NodeBase_initialize_bfs(IndexT id) {
  if (dev_nodes[id].type == kTypeAnchorNode) {
    dev_nodes[id].distance = 0;
  } else {
    dev_nodes[id].distance = kMaxDistance;  // should be int_max
  }
}


__device__ bool dev_bfs_continue;

__device__ void NodeBase_bfs_visit(IndexT id, int distance) {
  if (distance == dev_nodes[id].distance) {
    // Continue until all vertices were visited.
    dev_bfs_continue = true;

    for (int i = 0; i < kMaxDegree; ++i) {
      IndexT spring = dev_nodes[id].springs[i];

      if (spring != kNullptr) {
        // Find neighboring vertices.
        IndexT n;
        if (id == dev_springs[spring].p1) {
          n = dev_springs[spring].p2;
        } else {
          n = dev_springs[spring].p1;
        }

        if (dev_nodes[n].distance == kMaxDistance) {
          // Set distance on neighboring vertex if unvisited.
          dev_nodes[n].distance = distance + 1;
        }
      }
    }
  }
}


__device__ void NodeBase_bfs_set_delete_flags(IndexT id) {
  if (dev_nodes[id].distance == kMaxDistance) {  // should be int_max
    for (int i = 0; i < kMaxDegree; ++i) {
      IndexT spring = dev_nodes[id].springs[i];
      if (spring != kNullptr) {
        dev_springs[spring].delete_flag = true;
      }
    }
  }
}


__device__ void Spring_bfs_delete(IndexT id) {
  if (dev_springs[id].delete_flag) { Spring_self_destruct(id); }
}


// Only for rendering and checksum computation.
__device__ int dev_num_springs;
__device__ SpringInfo dev_spring_info[kMaxSprings];
int host_num_springs;
SpringInfo host_spring_info[kMaxSprings];

__device__ void Spring_add_to_rendering_array(IndexT id) {
  int idx = atomicAdd(&dev_num_springs, 1);
  dev_spring_info[idx].p1_x = dev_nodes[dev_springs[id].p1].pos_x;
  dev_spring_info[idx].p1_y = dev_nodes[dev_springs[id].p1].pos_y;
  dev_spring_info[idx].p2_x = dev_nodes[dev_springs[id].p2].pos_x;
  dev_spring_info[idx].p2_y = dev_nodes[dev_springs[id].p2].pos_y;
  dev_spring_info[idx].force = dev_springs[id].force;
  dev_spring_info[idx].max_force = dev_springs[id].max_force;
}


__global__ void kernel_AnchorPullNode_pull() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_nodes[i].type == kTypeAnchorPullNode) {
      AnchorPullNode_pull(i);
    }
  }
}


__global__ void kernel_Node_move() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_nodes[i].type == kTypeNode) {
      Node_move(i);
    }
  }
}


__global__ void kernel_NodeBase_initialize_bfs() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_nodes[i].type != 0) {
      NodeBase_initialize_bfs(i);
    }
  }
}


__global__ void kernel_NodeBase_bfs_visit(int dist) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_nodes[i].type != 0) {
      NodeBase_bfs_visit(i, dist);
    }
  }
}


__global__ void kernel_NodeBase_bfs_set_delete_flags() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_nodes[i].type != 0) {
      NodeBase_bfs_set_delete_flags(i);
    }
  }
}


__global__ void kernel_Spring_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    if (dev_springs[i].is_active) {
      Spring_compute_force(i);
    }
  }
}


__global__ void kernel_Spring_bfs_delete() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    if (dev_springs[i].is_active) {
      Spring_bfs_delete(i);
    }
  }
}


__global__ void kernel_Spring_add_to_rendering_array() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    if (dev_springs[i].is_active) {
      Spring_add_to_rendering_array(i);
    }
  }
}


__global__ void kernel_initialize_nodes() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    dev_nodes[i].type = 0;
  }
}


__global__ void kernel_initialize_springs() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    dev_springs[i].is_active = false;
  }
}


void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_springs, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Spring_add_to_rendering_array<<<128, 128>>>();
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
  kernel_Spring_compute_force<<<(kMaxSprings + kThreads - 1) / kThreads,
                                kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Node_move<<<(kMaxNodes + kThreads - 1) / kThreads,
                     kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void bfs_and_delete() {
  // Perform BFS to check reachability.
  kernel_NodeBase_initialize_bfs<<<(kMaxNodes + kThreads - 1) / kThreads,
                                    kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kMaxDistance; ++i) {
    bool continue_flag = false;
    cudaMemcpyToSymbol(dev_bfs_continue, &continue_flag, sizeof(bool), 0,
                       cudaMemcpyHostToDevice);

    kernel_NodeBase_bfs_visit<<<(kMaxNodes + kThreads - 1) / kThreads,
                                kThreads>>>(i);
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpyFromSymbol(&continue_flag, dev_bfs_continue, sizeof(bool), 0,
                         cudaMemcpyDeviceToHost);

    if (!continue_flag) break;
  }

  // Delete springs (and nodes).
  kernel_NodeBase_bfs_set_delete_flags<<<(kMaxNodes + kThreads - 1) / kThreads,
                                         kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Spring_bfs_delete<<<(kMaxSprings + kThreads - 1) / kThreads,
                             kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void step() {
  kernel_AnchorPullNode_pull<<<(kMaxNodes + kThreads - 1) / kThreads,
                               kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }

  bfs_and_delete();
}


void initialize_memory() {
  kernel_initialize_nodes<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_initialize_springs<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


__device__ IndexT dev_tmp_nodes[kMaxNodes];
__device__ IndexT dev_node_counter;
__global__ void kernel_create_nodes(DsNode* nodes, int num_nodes) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_nodes; i += blockDim.x * gridDim.x) {
    int idx = atomicAdd(&dev_node_counter, 1);
    dev_tmp_nodes[i] = idx;

    if (nodes[i].type == kTypeNode) {
      new_Node(idx, nodes[i].pos_x, nodes[i].pos_y, nodes[i].mass);
    } else if (nodes[i].type == kTypeAnchorPullNode) {
      new_AnchorPullNode(idx, nodes[i].pos_x, nodes[i].pos_y, nodes[i].vel_x,
                         nodes[i].vel_y);
    } else if (nodes[i].type == kTypeAnchorNode) {
      new_AnchorNode(idx, nodes[i].pos_x, nodes[i].pos_y);
    } else {
      assert(false);
    }
  }
}


__global__ void kernel_create_springs(DsSpring* springs, int num_springs) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_springs; i += blockDim.x * gridDim.x) {
    new_Spring(i, dev_tmp_nodes[springs[i].p1], dev_tmp_nodes[springs[i].p2],
               springs[i].spring_factor, springs[i].max_force);
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

  IndexT zero = 0;
  cudaMemcpyToSymbol(dev_node_counter, &zero, sizeof(IndexT), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_nodes<<<128, 128>>>(host_nodes, dataset.nodes.size());
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_springs<<<128, 128>>>(host_springs, dataset.springs.size());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(host_nodes);
  cudaFree(host_springs);
}


int main(int /*argc*/, char** /*argv*/) {
  // Allocate memory.
  Node* host_nodes;
  cudaMalloc(&host_nodes, sizeof(Node)*kMaxNodes);
  cudaMemcpyToSymbol(dev_nodes, &host_nodes, sizeof(Node*), 0,
                     cudaMemcpyHostToDevice);

  Spring* host_springs;
  cudaMalloc(&host_springs, sizeof(Spring)*kMaxSprings);
  cudaMemcpyToSymbol(dev_springs, &host_springs, sizeof(Spring*), 0,
                     cudaMemcpyHostToDevice);

  initialize_memory();

  Dataset dataset;
  random_dataset(dataset);
  load_dataset(dataset);

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumSteps; ++i) {
#ifndef NDEBUG
    printf("%i\n", i);
#endif  // NDEBUG
    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

  printf("%lu\n", micros);

#ifndef NDEBUG
  printf("Checksum: %f\n", checksum());
#endif  // NDEBUG
}
