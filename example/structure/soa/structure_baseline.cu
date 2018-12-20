#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>

#include "configuration.h"
#include "dataset.h"
#include "util/util.h"
#include "rendering.h"


static const int kThreads = 256;
static const int kNullptr = std::numeric_limits<int>::max();

using IndexT = int;

__device__ DeviceArray<IndexT, kMaxDegree>* dev_Node_springs;
__device__ float* dev_Node_pos_x;
__device__ float* dev_Node_pos_y;
__device__ int* dev_Node_num_springs;
__device__ float* dev_Node_vel_x;
__device__ float* dev_Node_vel_y;
__device__ float* dev_Node_mass;
__device__ char* dev_Node_type;
__device__ IndexT* dev_Spring_p1;
__device__ IndexT* dev_Spring_p2;
__device__ float* dev_Spring_factor;
__device__ float* dev_Spring_initial_length;
__device__ float* dev_Spring_force;
__device__ float* dev_Spring_max_force;
__device__ bool* dev_Spring_is_active;


__device__ void new_NodeBase(IndexT id, float pos_x, float pos_y) {
  dev_Node_pos_x[id] = pos_x;
  dev_Node_pos_y[id] = pos_y;
  dev_Node_num_springs[id] = 0;
  dev_Node_type[id] = kTypeNodeBase;
}


__device__ void new_AnchorNode(IndexT id, float pos_x, float pos_y) {
  new_NodeBase(id, pos_x, pos_y);
  dev_Node_type[id] = kTypeAnchorNode;
}


__device__ void new_AnchorPullNode(IndexT id, float pos_x, float pos_y,
                                   float vel_x, float vel_y) {
  new_AnchorNode(id, pos_x, pos_y);
  dev_Node_vel_x[id] = vel_x;
  dev_Node_vel_y[id] = vel_y;
  dev_Node_type[id] = kTypeAnchorPullNode;
}


__device__ void new_Node(IndexT id, float pos_x, float pos_y, float mass) {
  new_NodeBase(id, pos_x, pos_y);
  dev_Node_mass[id] = mass;
  dev_Node_type[id] = kTypeNode;
}


__device__ float NodeBase_distance_to(IndexT id, IndexT other) {
  float dx = dev_Node_pos_x[id] - dev_Node_pos_x[other];
  float dy = dev_Node_pos_y[id] - dev_Node_pos_y[other];
  float dist_sq = dx*dx + dy*dy;
  return sqrt(dist_sq);
}


__device__ void NodeBase_add_spring(IndexT id, IndexT spring) {
  assert(id >= 0 && id < kMaxNodes);

  int idx = atomicAdd(&dev_Node_num_springs[id], 1);
  assert(idx + 1 <= kMaxDegree);
  dev_Node_springs[id][idx] = spring;

  assert(dev_Spring_p1[spring] == id || dev_Spring_p2[spring] == id);
}


__device__ void new_Spring(IndexT id, IndexT p1, IndexT p2,
                           float spring_factor, float max_force) {
  dev_Spring_is_active[id] = true;
  dev_Spring_p1[id] = p1;
  dev_Spring_p2[id] = p2;
  dev_Spring_factor[id] = spring_factor;
  dev_Spring_force[id] = 0.0f;
  dev_Spring_max_force[id] = max_force;
  dev_Spring_initial_length[id] = NodeBase_distance_to(p1, p2);
  assert(dev_Spring_initial_length[id] > 0.0f);

  NodeBase_add_spring(p1, id);
  NodeBase_add_spring(p2, id);
}


__device__ void NodeBase_remove_spring(IndexT id, IndexT spring) {
  // TODO: This won't work if two springs break at the same time.

  int i = 0;
  IndexT s = kNullptr;

  do {
    assert(i < kMaxDegree);
    s = dev_Node_springs[id][i];
    ++i;
  } while(s != spring);

  for (; i < dev_Node_num_springs[id]; ++i) {
    dev_Node_springs[id][i - 1] = dev_Node_springs[id][i];
  }

  --dev_Node_num_springs[id];

  if (dev_Node_num_springs[id] == 0) {
    dev_Node_type[id] = 0;
  }
}


__device__ void AnchorPullNode_pull(IndexT id) {
  dev_Node_pos_x[id] += dev_Node_vel_x[id] * kDt;
  dev_Node_pos_y[id] += dev_Node_vel_y[id] * kDt;
}


__device__ void Spring_compute_force(IndexT id) {
  float dist = NodeBase_distance_to(dev_Spring_p1[id], dev_Spring_p2[id]);
  float displacement = max(0.0f, dist - dev_Spring_initial_length[id]);
  dev_Spring_force[id] = dev_Spring_factor[id] * displacement;

  if (dev_Spring_force[id] > dev_Spring_max_force[id]) {
    NodeBase_remove_spring(dev_Spring_p1[id], id);
    NodeBase_remove_spring(dev_Spring_p2[id], id);
    dev_Spring_is_active[id] = false;
  }
}


__device__ void Node_move(IndexT id) {
  float force_x = 0.0f;
  float force_y = 0.0f;

  for (int i = 0; i < dev_Node_num_springs[id]; ++i) {
    IndexT s = dev_Node_springs[id][i];
    IndexT from;
    IndexT to;

    if (dev_Spring_p1[s] == id) {
      from = id;
      to = dev_Spring_p2[s];
    } else {
      assert(dev_Spring_p2[s] == id);
      from = id;
      to = dev_Spring_p1[s];
    }

    // Calculate unit vector.
    float dx = dev_Node_pos_x[to] - dev_Node_pos_x[from];
    float dy = dev_Node_pos_y[to] - dev_Node_pos_y[from];
    float dist = sqrt(dx*dx + dy*dy);
    float unit_x = dx/dist;
    float unit_y = dy/dist;

    // Apply force.
    force_x += unit_x*dev_Spring_force[s];
    force_y += unit_y*dev_Spring_force[s];
  }

  // Calculate new velocity and position.
  dev_Node_vel_x[id] += force_x*kDt / dev_Node_mass[id];
  dev_Node_vel_y[id] += force_y*kDt / dev_Node_mass[id];
  dev_Node_vel_x[id] *= 1.0f - kVelocityDampening;
  dev_Node_vel_y[id] *= 1.0f - kVelocityDampening;
  dev_Node_pos_x[id] += dev_Node_vel_x[id]*kDt;
  dev_Node_pos_y[id] += dev_Node_vel_y[id]*kDt;
}


// Only for rendering.
__device__ int dev_num_springs;
__device__ SpringInfo dev_spring_info[kMaxSprings];
int host_num_springs;
SpringInfo host_spring_info[kMaxSprings];

__device__ void Spring_add_to_rendering_array(IndexT id) {
  int idx = atomicAdd(&dev_num_springs, 1);
  dev_spring_info[idx].p1_x = dev_Node_pos_x[dev_Spring_p1[id]];
  dev_spring_info[idx].p1_y = dev_Node_pos_y[dev_Spring_p1[id]];
  dev_spring_info[idx].p2_x = dev_Node_pos_x[dev_Spring_p2[id]];
  dev_spring_info[idx].p2_y = dev_Node_pos_y[dev_Spring_p2[id]];
  dev_spring_info[idx].force = dev_Spring_force[id];
  dev_spring_info[idx].max_force = dev_Spring_max_force[id];
}


__global__ void kernel_AnchorPullNode_pull() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_Node_type[i] == kTypeAnchorPullNode) {
      AnchorPullNode_pull(i);
    }
  }
}


__global__ void kernel_Node_move() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    if (dev_Node_type[i] == kTypeNode) {
      Node_move(i);
    }
  }
}


__global__ void kernel_Spring_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    if (dev_Spring_is_active[i]) {
      Spring_compute_force(i);
    }
  }
}


__global__ void kernel_Spring_add_to_rendering_array() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    if (dev_Spring_is_active[i]) {
      Spring_add_to_rendering_array(i);
    }
  }
}


__global__ void kernel_initialize_nodes() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxNodes; i += blockDim.x * gridDim.x) {
    dev_Node_type[i] = 0;
  }
}


__global__ void kernel_initialize_springs() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kMaxSprings; i += blockDim.x * gridDim.x) {
    dev_Spring_is_active[i] = false;
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


void step() {
  kernel_AnchorPullNode_pull<<<(kMaxNodes + kThreads - 1) / kThreads,
                               kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }
}


void initialize_memory() {
  kernel_initialize_nodes<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_initialize_springs<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


//__device__ NodeBase* tmp_nodes[kMaxNodes];

__global__ void kernel_create_nodes(DsNode* nodes, int num_nodes) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_nodes; i += blockDim.x * gridDim.x) {
    if (nodes[i].type == kTypeNode) {
      new_Node(i, nodes[i].pos_x, nodes[i].pos_y, nodes[i].mass);
    } else if (nodes[i].type == kTypeAnchorPullNode) {
      new_AnchorPullNode(i, nodes[i].pos_x, nodes[i].pos_y, nodes[i].vel_x,
                         nodes[i].vel_y);
    } else {
      assert(false);
    }
  }
}


__global__ void kernel_create_springs(DsSpring* springs, int num_springs) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < num_springs; i += blockDim.x * gridDim.x) {
    new_Spring(i, springs[i].p1, springs[i].p2,
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

  kernel_create_nodes<<<128, 128>>>(host_nodes, dataset.nodes.size());
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_springs<<<128, 128>>>(host_springs, dataset.springs.size());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(host_nodes);
  cudaFree(host_springs);
}


int main(int /*argc*/, char** /*argv*/) {
  // Allocate memory.
  DeviceArray<IndexT, kMaxDegree>* host_Node_springs;
  cudaMalloc(&host_Node_springs,
             sizeof(DeviceArray<IndexT, kMaxDegree>)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_springs, &host_Node_springs,
                     sizeof(DeviceArray<IndexT, kMaxDegree>*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Node_pos_x;
  cudaMalloc(&host_Node_pos_x, sizeof(float)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_pos_x, &host_Node_pos_x, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Node_pos_y;
  cudaMalloc(&host_Node_pos_y, sizeof(float)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_pos_y, &host_Node_pos_y, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Node_vel_x;
  cudaMalloc(&host_Node_vel_x, sizeof(float)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_vel_x, &host_Node_vel_x, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Node_vel_y;
  cudaMalloc(&host_Node_vel_y, sizeof(float)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_vel_y, &host_Node_vel_y, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  int* host_Node_num_springs;
  cudaMalloc(&host_Node_num_springs, sizeof(int)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_num_springs, &host_Node_num_springs,
                     sizeof(int*), 0, cudaMemcpyHostToDevice);

  float* host_Node_mass;
  cudaMalloc(&host_Node_mass, sizeof(float)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_mass, &host_Node_mass, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  char* host_Node_type;
  cudaMalloc(&host_Node_type, sizeof(char)*kMaxNodes);
  cudaMemcpyToSymbol(dev_Node_type, &host_Node_type, sizeof(char*), 0,
                     cudaMemcpyHostToDevice);

  IndexT* host_Spring_p1;
  cudaMalloc(&host_Spring_p1, sizeof(IndexT)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_p1, &host_Spring_p1, sizeof(IndexT*), 0,
                     cudaMemcpyHostToDevice);

  IndexT* host_Spring_p2;
  cudaMalloc(&host_Spring_p2, sizeof(IndexT)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_p2, &host_Spring_p2, sizeof(IndexT*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Spring_factor;
  cudaMalloc(&host_Spring_factor, sizeof(float)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_factor, &host_Spring_factor, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Spring_initial_length;
  cudaMalloc(&host_Spring_initial_length, sizeof(float)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_initial_length, &host_Spring_initial_length,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);

  float* host_Spring_force;
  cudaMalloc(&host_Spring_force, sizeof(float)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_force, &host_Spring_force, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);

  float* host_Spring_max_force;
  cudaMalloc(&host_Spring_max_force, sizeof(float)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_max_force, &host_Spring_max_force,
                     sizeof(float*), 0, cudaMemcpyHostToDevice);

  bool* host_Spring_is_active;
  cudaMalloc(&host_Spring_is_active, sizeof(bool)*kMaxSprings);
  cudaMemcpyToSymbol(dev_Spring_is_active, &host_Spring_is_active,
                     sizeof(bool*), 0, cudaMemcpyHostToDevice);

  initialize_memory();

  //load_example<<<1, 1>>>();
  //load_random<<<1, 1>>>();

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

  printf("Time: %lu ms\n", millis);
  printf("Checksum: %f\n", checksum());
}
