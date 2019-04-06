#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <cub/cub.cuh>

#include "../configuration.h"
#include "util/util.h"

static const int kNullptr = std::numeric_limits<int>::max();
static const int kNumBlockSize = 256;

static const char kCellTypeNormal = 1;
static const char kCellTypeProducer = 2;

using IndexT = int;
using CellPointerT = IndexT;
#include "../dataset.h"

__device__ DeviceArray<IndexT, kMaxDegree>* d_Cell_incoming;
__device__ DeviceArray<IndexT, kMaxDegree>* d_Cell_outgoing;
__device__ int* d_Cell_max_velocity;
__device__ int* d_Cell_current_max_velocity;
__device__ int* d_Cell_num_incoming;
__device__ int* d_Cell_num_outgoing;
__device__ float* d_Cell_x;
__device__ float* d_Cell_y;
__device__ bool* d_Cell_is_target;
__device__ curandState_t* d_Cell_random_state;
__device__ char* d_Cell_type;
__device__ curandState_t* d_Car_random_state;
__device__ DeviceArray<IndexT, kMaxVelocity>* d_Car_path;
__device__ int* d_Car_path_length;
__device__ int* d_Car_velocity;
__device__ int* d_Car_max_velocity;
__device__ bool* d_Cell_has_car;
__device__ bool* d_Cell_should_occupy;

__device__ int d_num_cells;
int host_num_cells;


class TrafficLight {
 private:
  DeviceArray<IndexT, kMaxDegree> cells_;
  int num_cells_;
  int timer_;
  int phase_time_;
  int phase_;

 public:
  __device__ TrafficLight(int num_cells, int phase_time)
      : num_cells_(num_cells), timer_(0), phase_time_(phase_time), phase_(0) {}

  __device__ void set_cell(int idx, IndexT cell) {
    assert(cell != kNullptr);
    cells_[idx] = cell;
  }

  __device__ void step();
};


// TODO: Consider migrating to SoaAlloc.
TrafficLight* h_traffic_lights;
__device__ TrafficLight* d_traffic_lights;


// Only for rendering.
__device__ int dev_num_cells;
__device__ float* dev_Cell_pos_x;
__device__ float* dev_Cell_pos_y;
__device__ bool* dev_Cell_occupied;
float* host_Cell_pos_x;
float* host_Cell_pos_y;
bool* host_Cell_occupied;
float* host_data_Cell_pos_x;
float* host_data_Cell_pos_y;
bool* host_data_Cell_occupied;


__device__ int Cell_current_max_velocity(IndexT self) {
  return d_Cell_current_max_velocity[self];
}

__device__ int Cell_max_velocity(IndexT self) {
  return d_Cell_max_velocity[self];
}

__device__ void Cell_set_current_max_velocity(IndexT self, int v) {
  d_Cell_current_max_velocity[self] = v;
}

__device__ void Cell_remove_speed_limit(IndexT self) {
  d_Cell_current_max_velocity[self] = d_Cell_max_velocity[self];
}

__device__ int Cell_num_incoming(IndexT self) {
  return d_Cell_num_incoming[self];
}

__device__ void Cell_set_num_incoming(IndexT self, int num) {
  d_Cell_num_incoming[self] = num;
}

__device__ int Cell_num_outgoing(IndexT self) {
  return d_Cell_num_outgoing[self];
}

__device__ void Cell_set_num_outgoing(IndexT self, int num) {
  d_Cell_num_outgoing[self] = num;
}

__device__ IndexT get_incoming(IndexT self, int idx) {
  return d_Cell_incoming[self][idx];
}

__device__ void Cell_set_incoming(IndexT self, int idx, IndexT cell) {
  assert(cell != kNullptr);
  d_Cell_incoming[self][idx] = cell;
}

__device__ IndexT Cell_get_outgoing(IndexT self, int idx) {
  return d_Cell_outgoing[self][idx];
}

__device__ void Cell_set_outgoing(IndexT self, int idx, IndexT cell) {
  assert(cell != kNullptr);
  d_Cell_outgoing[self][idx] = cell;
}

__device__ float Cell_x(IndexT self) { return d_Cell_x[self]; }

__device__ float Cell_y(IndexT self) { return d_Cell_y[self]; }

__device__ bool Cell_is_free(IndexT self) { return !d_Cell_has_car[self]; }

__device__ bool Cell_is_sink(IndexT self) { return d_Cell_num_outgoing[self] == 0; }

__device__ bool Cell_is_target(IndexT self) { return d_Cell_is_target[self]; }

__device__ void Cell_set_target(IndexT self) { d_Cell_is_target[self] = true; }

__device__ int Car_random_int(IndexT self, int a, int b) {
  return curand(&d_Car_random_state[self]) % (b - a) + a;
}

__device__ int Car_velocity(IndexT self) { return d_Car_velocity[self]; }

__device__ int Car_max_velocity(IndexT self) { return d_Car_max_velocity[self]; }

__device__ void Cell_occupy(IndexT self, IndexT car) {
  assert(d_Cell_has_car[car]);
  assert(Cell_is_free(self));
  d_Cell_should_occupy[self] = true;
  d_Car_velocity[self] = d_Car_velocity[car];
  d_Car_max_velocity[self] = d_Car_max_velocity[car];
  d_Car_random_state[self] = d_Car_random_state[car];

  for (int i = 0; i < kMaxVelocity; ++i) {
    d_Car_path[self][i] = d_Car_path[car][i];
  }
  d_Car_path_length[self] = d_Car_path_length[car];
}


__device__ void Cell_release(IndexT self) {
  assert(!Cell_is_free(self));
  d_Cell_has_car[self] = false;
}


__device__ IndexT Car_next_step(IndexT self, IndexT position) {
  // Almost random walk.
  const uint32_t num_outgoing = d_Cell_num_outgoing[position];
  assert(num_outgoing > 0);

  // Need some kind of return statement here.
  return d_Cell_outgoing[position][Car_random_int(self, 0, num_outgoing)];
}


__device__ void Car_step_initialize_iteration(IndexT self) {
  // Reset calculated path. This forces cars with a random moving behavior to
  // select a new path in every iteration. Otherwise, cars might get "stuck"
  // on a full network if many cars are waiting for the one in front of them in
  // a cycle.
  d_Car_path_length[self] = 0;
}


__device__ void Car_step_accelerate(IndexT self) {
  // Speed up the car by 1 or 2 units.
  int speedup = Car_random_int(self, 0, 2) + 1;
  d_Car_velocity[self] = d_Car_max_velocity[self] < d_Car_velocity[self] + speedup
      ? d_Car_max_velocity[self] : d_Car_velocity[self] + speedup;
}


__device__ void Car_step_extend_path(IndexT self) {
  IndexT cell = self;
  IndexT next_cell;

  for (int i = 0; i < d_Car_velocity[self]; ++i) {
    if (Cell_is_sink(cell) || Cell_is_target(cell)) {
      break;
    }

    next_cell = Car_next_step(self, cell);
    assert(next_cell != cell);

    if (!Cell_is_free(next_cell)) break;

    cell = next_cell;
    d_Car_path[self][i] = cell;
    d_Car_path_length[self] = d_Car_path_length[self] + 1;
  }

  d_Car_velocity[self] = d_Car_path_length[self];
}


__device__ void Car_step_constraint_velocity(IndexT self) {
  // This is actually only needed for the very first iteration, because a car
  // may be positioned on a traffic light cell.
  if (d_Car_velocity[self] > Cell_current_max_velocity(self)) {
    d_Car_velocity[self] = Cell_current_max_velocity(self);
  }

  int path_index = 0;
  int distance = 1;

  while (distance <= d_Car_velocity[self]) {
    // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    IndexT next_cell = d_Car_path[self][path_index];

    // Avoid collision.
    if (!Cell_is_free(next_cell)) {
      // Cannot enter cell.
      --distance;
      d_Car_velocity[self] = distance;
      break;
    } // else: Can enter next cell.

    if (d_Car_velocity[self] > Cell_current_max_velocity(next_cell)) {
      // Car is too fast for this cell.
      if (Cell_current_max_velocity(next_cell) > distance - 1) {
        // Even if we slow down, we would still make progress.
        d_Car_velocity[self] = Cell_current_max_velocity(next_cell);
      } else {
        // Do not enter the next cell.
        --distance;
        assert(distance >= 0);

        d_Car_velocity[self] = distance;
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;

#ifndef NDEBUG
  for (int i = 0; i < d_Car_velocity[self]; ++i) {
    assert(Cell_is_free(d_Car_path[self][i]));
    assert(i == 0 || d_Car_path[self][i - 1] != d_Car_path[self][i]);
  }
  // TODO: Check why the cast is necessary.
  assert(distance <= d_Car_velocity[self]);
#endif  // NDEBUG
}


__device__ void Car_step_move(IndexT self) {
  IndexT cell = self;

  for (int i = 0; i < d_Car_velocity[self]; ++i) {
    assert(d_Car_path[self][i] != cell);

    cell = d_Car_path[self][i];
    assert(cell != self);
    assert(Cell_is_free(cell));
  }

  if (d_Car_velocity[self] > 0) {
    Cell_occupy(cell, self);
    Cell_release(self);
  }
}


__device__ void Car_step_slow_down(IndexT self) {
  // 20% change of slowdown.
  if (curand_uniform(&d_Car_random_state[self]) < 0.2 && d_Car_velocity[self] > 0) {
    d_Car_velocity[self] = d_Car_velocity[self] - 1;
  }
}


__device__ void TrafficLight::step() {
  if (num_cells_ > 0) {
    timer_ = (timer_ + 1) % phase_time_;

    if (timer_ == 0) {
      assert(cells_[phase_] != kNullptr);
      Cell_set_current_max_velocity(cells_[phase_], 0);
      phase_ = (phase_ + 1) % num_cells_;
      Cell_remove_speed_limit(cells_[phase_]);
    }
  }
}


__device__ void Car_commit_occupy(IndexT self) {
  if (d_Cell_should_occupy[self]) {
    d_Cell_should_occupy[self] = false;
    d_Cell_has_car[self] = true;
  }

  if (Cell_is_sink(self) || Cell_is_target(self)) {
    // Remove car from the simulation. Will be added again in the next
    // iteration.
    Cell_release(self);
  }
}


__device__ IndexT new_Car(int seed, IndexT cell, int max_velocity) {
  assert(Cell_is_free(cell));
  d_Cell_has_car[cell] = true;

  d_Car_path_length[cell] = 0;
  d_Car_velocity[cell] = 0;
  d_Car_max_velocity[cell] = max_velocity;

  curand_init(seed, 0, 0, &d_Car_random_state[cell]);

  return cell;
}


__device__ void ProducerCell_create_car(IndexT self) {
  assert(d_Cell_type[self] == kCellTypeProducer);
  if (Cell_is_free(self)) {
    float r = curand_uniform(&d_Cell_random_state[self]);
    if (r < kCarAllocationRatio) {
      IndexT new_car = new_Car(
          /*seed=*/ curand(&d_Cell_random_state[self]), /*cell=*/ self,
          /*max_velocity=*/ curand(&d_Cell_random_state[self]) % (kMaxVelocity/2)
                            + kMaxVelocity/2);
    }
  }
}


__device__ IndexT new_Cell(int max_velocity, float x, float y) {
  IndexT idx = atomicAdd(&d_num_cells, 1);
  
  d_Cell_max_velocity[idx] = max_velocity;
  d_Cell_current_max_velocity[idx] = max_velocity;
  d_Cell_num_incoming[idx] = 0;
  d_Cell_num_outgoing[idx] = 0;
  d_Cell_x[idx] = x;
  d_Cell_y[idx] = y;
  d_Cell_is_target[idx] = false;
  d_Cell_type[idx] = kCellTypeNormal;
  d_Cell_should_occupy[idx] = false;
  d_Cell_has_car[idx] = false;

  return idx;
}


__device__ IndexT new_ProducerCell(int max_velocity, float x, float y, int seed) {
  IndexT idx = new_Cell(max_velocity, x, y);
  d_Cell_type[idx] = kCellTypeProducer;
  curand_init(seed, 0, 0, &d_Cell_random_state[idx]);

  return idx;
}


__global__ void kernel_traffic_light_step() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    d_traffic_lights[i].step();
  }
}


__global__ void kernel_create_nodes() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    curandState_t state;
    curand_init(i, 0, 0, &state);

    assert(d_nodes[i].x >= 0 && d_nodes[i].x <= 1);
    assert(d_nodes[i].y >= 0 && d_nodes[i].y <= 1);

    for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
      d_nodes[i].cell_out[j] = new_Cell(
          /*max_velocity=*/ curand(&state) % (kMaxVelocity/2)
                            + kMaxVelocity/2,
          d_nodes[i].x, d_nodes[i].y);
    }
  }
}


__device__ IndexT connect_intersections(IndexT from, Node* target,
                                        int incoming_idx, curandState_t& state) {
  // Create edge.
  float dx = target->x - d_Cell_x[from];
  float dy = target->y - d_Cell_y[from];
  float dist = sqrt(dx*dx + dy*dy);
  int steps = dist/kCellLength;
  float step_x = dx/steps;
  float step_y = dy/steps;
  IndexT prev = from;

  for (int j = 0; j < steps; ++j) {
    float new_x = d_Cell_x[from] + j*step_x;
    float new_y = d_Cell_y[from] + j*step_y;
    assert(new_x >= 0 && new_x <= 1);
    assert(new_y >= 0 && new_y <= 1);
    IndexT next;

    if (curand_uniform(&state) < kProducerRatio) {
      next = new_ProducerCell(
          d_Cell_max_velocity[prev], new_x, new_y,
          curand(&state));
    } else {
      next = new_Cell(
          d_Cell_max_velocity[prev], new_x, new_y);
    }

    if (curand_uniform(&state) < kTargetRatio) {
      Cell_set_target(next);
    }

    Cell_set_num_outgoing(prev, 1);
    Cell_set_outgoing(prev, 0, next);
    Cell_set_num_incoming(next, 1);
    Cell_set_incoming(next, 0, prev);

    prev = next;
  }

  // Connect to all outgoing nodes of target.
  Cell_set_num_outgoing(prev, target->num_outgoing);
  for (int i = 0; i < target->num_outgoing; ++i) {
    IndexT next = target->cell_out[i];
    // num_incoming set later.
    Cell_set_outgoing(prev, i, next);
    Cell_set_incoming(next, incoming_idx, prev);
  }

  return prev;
}


__global__ void kernel_create_edges() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    curandState_t state;
    curand_init(i, 0, 0, &state);

    for (int k = 0; k < d_nodes[i].num_outgoing; ++k) {
      int target = d_nodes[i].node_out[k];
      int target_pos = d_nodes[i].node_out_pos[k];

      IndexT last = connect_intersections(
          d_nodes[i].cell_out[k], &d_nodes[target], target_pos, state);

      Cell_set_current_max_velocity(last, 0);
      d_nodes[target].cell_in[target_pos] = last;
    }
  }
}


__global__ void kernel_create_traffic_lights() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    new(d_traffic_lights + i) TrafficLight(
        /*num_cells=*/ d_nodes[i].num_incoming,
        /*phase_time=*/ 5);

    for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
      Cell_set_num_incoming(d_nodes[i].cell_out[j], d_nodes[i].num_incoming);
    }

    for (int j = 0; j < d_nodes[i].num_incoming; ++j) {
      d_traffic_lights[i].set_cell(j, d_nodes[i].cell_in[j]);
      Cell_set_current_max_velocity(d_nodes[i].cell_in[j], 0);  // Set to "red".
    }
  }
}


void create_street_network() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&h_nodes, sizeof(Node)*kNumIntersections);
  cudaMemcpyToSymbol(d_nodes, &h_nodes, sizeof(Node*), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&h_traffic_lights, sizeof(TrafficLight)*kNumIntersections);
  cudaMemcpyToSymbol(d_traffic_lights, &h_traffic_lights,
                     sizeof(TrafficLight*), 0, cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  // Create basic structure on host.
  create_network_structure();

  kernel_create_nodes<<<
      (kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_edges<<<
      (kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_traffic_lights<<<
      (kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Allocate helper data structures for rendering.
  cudaMemcpyFromSymbol(&host_num_cells, d_num_cells, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  cudaMalloc(&host_Cell_pos_x, sizeof(float)*host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_pos_x, &host_Cell_pos_x, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&host_Cell_pos_y, sizeof(float)*host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_pos_y, &host_Cell_pos_y, sizeof(float*), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&host_Cell_occupied, sizeof(bool)*host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_occupied, &host_Cell_occupied, sizeof(bool*), 0,
                     cudaMemcpyHostToDevice);
  host_data_Cell_pos_x = (float*) malloc(sizeof(float)*host_num_cells);
  host_data_Cell_pos_y = (float*) malloc(sizeof(float)*host_num_cells);
  host_data_Cell_occupied = (bool*) malloc(sizeof(bool)*host_num_cells);

#ifndef NDEBUG
  printf("Number of cells: %i\n", host_num_cells);
#endif  // NDEBUG
}


void step_traffic_lights() { 
  // TODO: Consider migrating this to SoaAlloc.
  kernel_traffic_light_step<<<
      (kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


__device__ void Cell_add_to_rendering_array(IndexT self) {
  int idx = atomicAdd(&dev_num_cells, 1);
  dev_Cell_pos_x[idx] = d_Cell_x[self];
  dev_Cell_pos_y[idx] = d_Cell_y[self];
  dev_Cell_occupied[idx] = !Cell_is_free(self);
}


__global__ void kernel_Cell_add_to_rendering_array() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    Cell_add_to_rendering_array(i);
  }
}


void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  kernel_Cell_add_to_rendering_array<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(host_data_Cell_pos_x, host_Cell_pos_x,
             sizeof(float)*host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_pos_y, host_Cell_pos_y,
             sizeof(float)*host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_occupied, host_Cell_occupied,
             sizeof(bool)*host_num_cells, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaDeviceSynchronize());
}


__global__ void kernel_ProducerCell_create_car() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (d_Cell_type[i] == kCellTypeProducer) {
      ProducerCell_create_car(i);
    }
  }
}


__device__ void Car_step_prepare_path(IndexT self) {
  Car_step_initialize_iteration(self);
  Car_step_accelerate(self);
  Car_step_extend_path(self);
  Car_step_constraint_velocity(self);
  Car_step_slow_down(self);
}


__global__ void kernel_Car_step_prepare_path() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (d_Cell_has_car[i]) {
      Car_step_prepare_path(i);
    }
  }
}


__global__ void kernel_Car_commit_occupy() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (d_Cell_has_car[i] || d_Cell_should_occupy[i]) {
      Car_commit_occupy(i);
    }
  }
}


__global__ void kernel_Car_step_move() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (d_Cell_has_car[i]) {
      Car_step_move(i);
    }
  }
}


__device__ int d_checksum;
__global__ void kernel_compute_checksum() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (d_Cell_has_car[i]) {
      atomicAdd(&d_checksum, 1);
    }
  }
}


int checksum() {
  int zero = 0;
  cudaMemcpyToSymbol(d_checksum, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
  kernel_compute_checksum<<<128, 128>>>();

  int result;
  cudaMemcpyFromSymbol(&result, d_checksum, sizeof(int), 0, cudaMemcpyDeviceToHost);
  return result;
}


void step() {
  kernel_ProducerCell_create_car<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
  
  step_traffic_lights();

  kernel_Car_step_prepare_path<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Car_step_move<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Car_commit_occupy<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void allocate_memory() {
  DeviceArray<IndexT, kMaxDegree>* h_Cell_incoming;
  cudaMalloc(&h_Cell_incoming, sizeof(DeviceArray<IndexT, kMaxDegree>)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_incoming, &h_Cell_incoming,
      sizeof(DeviceArray<IndexT, kMaxDegree>*), 0, cudaMemcpyHostToDevice);

  DeviceArray<IndexT, kMaxDegree>* h_Cell_outgoing;
  cudaMalloc(&h_Cell_outgoing, sizeof(DeviceArray<IndexT, kMaxDegree>)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_outgoing, &h_Cell_outgoing,
      sizeof(DeviceArray<IndexT, kMaxDegree>*), 0, cudaMemcpyHostToDevice);

  int* h_Cell_max_velocity;
  cudaMalloc(&h_Cell_max_velocity, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_max_velocity, &h_Cell_max_velocity, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  int* h_Cell_current_max_velocity;
  cudaMalloc(&h_Cell_current_max_velocity, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_current_max_velocity, &h_Cell_current_max_velocity, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  int* h_Cell_num_incoming;
  cudaMalloc(&h_Cell_num_incoming, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_num_incoming, &h_Cell_num_incoming, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  int* h_Cell_num_outgoing;
  cudaMalloc(&h_Cell_num_outgoing, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_num_outgoing, &h_Cell_num_outgoing, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  float* h_Cell_x;
  cudaMalloc(&h_Cell_x, sizeof(float)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_x, &h_Cell_x, sizeof(float*),
                     0, cudaMemcpyHostToDevice);

  float* h_Cell_y;
  cudaMalloc(&h_Cell_y, sizeof(float)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_y, &h_Cell_y, sizeof(float*),
                     0, cudaMemcpyHostToDevice);

  bool* h_Cell_is_target;
  cudaMalloc(&h_Cell_is_target, sizeof(bool)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_is_target, &h_Cell_is_target, sizeof(bool*),
                     0, cudaMemcpyHostToDevice);

  curandState_t* h_Cell_random_state;
  cudaMalloc(&h_Cell_random_state, sizeof(curandState_t)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_random_state, &h_Cell_random_state, sizeof(curandState_t*),
                     0, cudaMemcpyHostToDevice);

  char* h_Cell_type;
  cudaMalloc(&h_Cell_type, sizeof(char)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_type, &h_Cell_type, sizeof(char*),
                     0, cudaMemcpyHostToDevice);

  curandState_t* h_Car_random_state;
  cudaMalloc(&h_Car_random_state, sizeof(curandState_t)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Car_random_state, &h_Car_random_state, sizeof(curandState_t*),
                     0, cudaMemcpyHostToDevice);

  DeviceArray<IndexT, kMaxVelocity>* h_Car_path;
  cudaMalloc(&h_Car_path, sizeof(DeviceArray<IndexT, kMaxVelocity>)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Car_path, &h_Car_path,
                     sizeof(DeviceArray<IndexT, kMaxVelocity>*),
                     0, cudaMemcpyHostToDevice);

  int* h_Car_path_length;
  cudaMalloc(&h_Car_path_length, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Car_path_length, &h_Car_path_length, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  int* h_Car_velocity;
  cudaMalloc(&h_Car_velocity, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Car_velocity, &h_Car_velocity, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  int* h_Car_max_velocity;
  cudaMalloc(&h_Car_max_velocity, sizeof(int)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Car_max_velocity, &h_Car_max_velocity, sizeof(int*),
                     0, cudaMemcpyHostToDevice);

  bool* h_Cell_has_car;
  cudaMalloc(&h_Cell_has_car, sizeof(bool)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_has_car, &h_Cell_has_car, sizeof(bool*),
                     0, cudaMemcpyHostToDevice);

  bool* h_Cell_should_occupy;
  cudaMalloc(&h_Cell_should_occupy, sizeof(bool)*kMaxNumCells);
  cudaMemcpyToSymbol(d_Cell_should_occupy, &h_Cell_should_occupy, sizeof(bool*),
                     0, cudaMemcpyHostToDevice);

  int zero = 0;
  cudaMemcpyToSymbol(d_num_cells, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
}


int main(int /*argc*/, char** /*argv*/) {
  allocate_memory();
  create_street_network();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumIterations; ++i) {
    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

  printf("%lu\n", micros);
}
