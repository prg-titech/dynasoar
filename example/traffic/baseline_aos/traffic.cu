#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <cub/cub.cuh>

#include "../configuration.h"
#include "util/util.h"

using IndexT = int;
using CellPointerT = IndexT;
#include "../dataset.h"

static const int kNullptr = std::numeric_limits<int>::max();
static const int kNumBlockSize = 256;

static const char kCellTypeNormal = 1;
static const char kCellTypeProducer = 2;


struct Cell {
  curandState_t random_state;
  curandState_t car_random_state;
  DeviceArray<IndexT, kMaxDegree> incoming;
  DeviceArray<IndexT, kMaxDegree> outgoing;
  DeviceArray<IndexT, kMaxVelocity> car_path;
  int car_path_length;
  int car_velocity;
  int car_max_velocity;
  int num_incoming;
  int num_outgoing;
  int max_velocity;
  int current_max_velocity;
  float x;
  float y;
  bool has_car;
  bool is_target;
  bool should_occupy;
  char type;
};

__device__ Cell* dev_cells;
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
  return dev_cells[self].current_max_velocity;
}

__device__ int Cell_max_velocity(IndexT self) {
  return dev_cells[self].max_velocity;
}

__device__ void Cell_set_current_max_velocity(IndexT self, int v) {
  dev_cells[self].current_max_velocity = v;
}

__device__ void Cell_remove_speed_limit(IndexT self) {
  dev_cells[self].current_max_velocity = dev_cells[self].max_velocity;
}

__device__ int Cell_num_incoming(IndexT self) {
  return dev_cells[self].num_incoming;
}

__device__ void Cell_set_num_incoming(IndexT self, int num) {
  dev_cells[self].num_incoming = num;
}

__device__ int Cell_num_outgoing(IndexT self) {
  return dev_cells[self].num_outgoing;
}

__device__ void Cell_set_num_outgoing(IndexT self, int num) {
  dev_cells[self].num_outgoing = num;
}

__device__ IndexT get_incoming(IndexT self, int idx) {
  return dev_cells[self].incoming[idx];
}

__device__ void Cell_set_incoming(IndexT self, int idx, IndexT cell) {
  assert(cell != kNullptr);
  dev_cells[self].incoming[idx] = cell;
}

__device__ IndexT Cell_get_outgoing(IndexT self, int idx) {
  return dev_cells[self].outgoing[idx];
}

__device__ void Cell_set_outgoing(IndexT self, int idx, IndexT cell) {
  assert(cell != kNullptr);
  dev_cells[self].outgoing[idx] = cell;
}

__device__ float Cell_x(IndexT self) { return dev_cells[self].x; }

__device__ float Cell_y(IndexT self) { return dev_cells[self].y; }

__device__ bool Cell_is_free(IndexT self) { return !dev_cells[self].has_car; }

__device__ bool Cell_is_sink(IndexT self) { return dev_cells[self].num_outgoing == 0; }

__device__ bool Cell_is_target(IndexT self) { return dev_cells[self].is_target; }

__device__ void Cell_set_target(IndexT self) { dev_cells[self].is_target = true; }

__device__ int Car_random_int(IndexT self, int a, int b) {
  return curand(&dev_cells[self].car_random_state) % (b - a) + a;
}

__device__ int Car_velocity(IndexT self) { return dev_cells[self].car_velocity; }

__device__ int Car_max_velocity(IndexT self) { return dev_cells[self].car_max_velocity; }


__device__ void Cell_occupy(IndexT self, IndexT car) {
  assert(dev_cells[car].has_car);
  assert(Cell_is_free(self));
  dev_cells[self].should_occupy = true;
  dev_cells[self].car_velocity = dev_cells[car].car_velocity;
  dev_cells[self].car_max_velocity = dev_cells[car].car_max_velocity;
  dev_cells[self].car_random_state = dev_cells[car].car_random_state;

  for (int i = 0; i < kMaxVelocity; ++i) {
    dev_cells[self].car_path[i] = dev_cells[car].car_path[i];
  }
  dev_cells[self].car_path_length = dev_cells[car].car_path_length;
}


__device__ void Cell_release(IndexT self) {
  assert(!Cell_is_free(self));
  dev_cells[self].has_car = false;
}


__device__ IndexT Car_next_step(IndexT self, IndexT position) {
  // Almost random walk.
  const uint32_t num_outgoing = dev_cells[position].num_outgoing;
  assert(num_outgoing > 0);

  // Need some kind of return statement here.
  return dev_cells[position].outgoing[Car_random_int(self, 0, num_outgoing)];
}


__device__ void Car_step_initialize_iteration(IndexT self) {
  // Reset calculated path. This forces cars with a random moving behavior to
  // select a new path in every iteration. Otherwise, cars might get "stuck"
  // on a full network if many cars are waiting for the one in front of them in
  // a cycle.
  dev_cells[self].car_path_length = 0;
}


__device__ void Car_step_accelerate(IndexT self) {
  // Speed up the car by 1 or 2 units.
  int speedup = Car_random_int(self, 0, 2) + 1;
  dev_cells[self].car_velocity = dev_cells[self].car_max_velocity < dev_cells[self].car_velocity + speedup
      ? dev_cells[self].car_max_velocity : dev_cells[self].car_velocity + speedup;
}


__device__ void Car_step_extend_path(IndexT self) {
  IndexT cell = self;
  IndexT next_cell;

  for (int i = 0; i < dev_cells[self].car_velocity; ++i) {
    if (Cell_is_sink(cell) || Cell_is_target(cell)) {
      break;
    }

    next_cell = Car_next_step(self, cell);
    assert(next_cell != cell);

    if (!Cell_is_free(next_cell)) break;

    cell = next_cell;
    dev_cells[self].car_path[i] = cell;
    dev_cells[self].car_path_length = dev_cells[self].car_path_length + 1;
  }

  dev_cells[self].car_velocity = dev_cells[self].car_path_length;
}


__device__ void Car_step_constraint_velocity(IndexT self) {
  // This is actually only needed for the very first iteration, because a car
  // may be positioned on a traffic light cell.
  if (dev_cells[self].car_velocity > Cell_current_max_velocity(self)) {
    dev_cells[self].car_velocity = Cell_current_max_velocity(self);
  }

  int path_index = 0;
  int distance = 1;

  while (distance <= dev_cells[self].car_velocity) {
    // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    IndexT next_cell = dev_cells[self].car_path[path_index];

    // Avoid collision.
    if (!Cell_is_free(next_cell)) {
      // Cannot enter cell.
      --distance;
      dev_cells[self].car_velocity = distance;
      break;
    } // else: Can enter next cell.

    if (dev_cells[self].car_velocity > Cell_current_max_velocity(next_cell)) {
      // Car is too fast for this cell.
      if (Cell_current_max_velocity(next_cell) > distance - 1) {
        // Even if we slow down, we would still make progress.
        dev_cells[self].car_velocity = Cell_current_max_velocity(next_cell);
      } else {
        // Do not enter the next cell.
        --distance;
        assert(distance >= 0);

        dev_cells[self].car_velocity = distance;
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;

#ifndef NDEBUG
  for (int i = 0; i < dev_cells[self].car_velocity; ++i) {
    assert(Cell_is_free(dev_cells[self].car_path[i]));
    assert(i == 0 || dev_cells[self].car_path[i - 1] != dev_cells[self].car_path[i]);
  }
  // TODO: Check why the cast is necessary.
  assert(distance <= dev_cells[self].car_velocity);
#endif  // NDEBUG
}


__device__ void Car_step_move(IndexT self) {
  IndexT cell = self;

  for (int i = 0; i < dev_cells[self].car_velocity; ++i) {
    assert(dev_cells[self].car_path[i] != cell);

    cell = dev_cells[self].car_path[i];
    assert(cell != self);
    assert(Cell_is_free(cell));
  }

  if (dev_cells[self].car_velocity > 0) {
    Cell_occupy(cell, self);
    Cell_release(self);
  }
}


__device__ void Car_commit_occupy(IndexT self) {
  if (dev_cells[self].should_occupy) {
    dev_cells[self].should_occupy = false;
    dev_cells[self].has_car = true;
  }

  if (Cell_is_sink(self) || Cell_is_target(self)) {
    // Remove car from the simulation. Will be added again in the next
    // iteration.
    Cell_release(self);
  }
}


__device__ void Car_step_slow_down(IndexT self) {
  // 20% change of slowdown.
  if (curand_uniform(&dev_cells[self].car_random_state) < 0.2 && dev_cells[self].car_velocity > 0) {
    dev_cells[self].car_velocity = dev_cells[self].car_velocity - 1;
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


__device__ IndexT new_Car(int seed, IndexT cell, int max_velocity) {
  assert(Cell_is_free(cell));
  dev_cells[cell].has_car = true;

  dev_cells[cell].car_path_length = 0;
  dev_cells[cell].car_velocity = 0;
  dev_cells[cell].car_max_velocity = max_velocity;

  curand_init(seed, 0, 0, &dev_cells[cell].car_random_state);

  return cell;
}


__device__ void ProducerCell_create_car(IndexT self) {
  assert(dev_cells[self].type == kCellTypeProducer);
  if (Cell_is_free(self)) {
    float r = curand_uniform(&dev_cells[self].random_state);
    if (r < kCarAllocationRatio) {
      IndexT new_car = new_Car(
          /*seed=*/ curand(&dev_cells[self].random_state), /*cell=*/ self,
          /*max_velocity=*/ curand(&dev_cells[self].random_state) % (kMaxVelocity/2)
                            + kMaxVelocity/2);
    }
  }
}


__device__ IndexT new_Cell(int max_velocity, float x, float y) {
  IndexT idx = atomicAdd(&d_num_cells, 1);
  
  dev_cells[idx].max_velocity = max_velocity;
  dev_cells[idx].current_max_velocity = max_velocity;
  dev_cells[idx].num_incoming = 0;
  dev_cells[idx].num_outgoing = 0;
  dev_cells[idx].x = x;
  dev_cells[idx].y = y;
  dev_cells[idx].is_target = false;
  dev_cells[idx].type = kCellTypeNormal;
  dev_cells[idx].has_car = false;
  dev_cells[idx].should_occupy = false;

  return idx;
}


__device__ IndexT new_ProducerCell(int max_velocity, float x, float y, int seed) {
  IndexT idx = new_Cell(max_velocity, x, y);
  dev_cells[idx].type = kCellTypeProducer;
  curand_init(seed, 0, 0, &dev_cells[idx].random_state);

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
  float dx = target->x - dev_cells[from].x;
  float dy = target->y - dev_cells[from].y;
  float dist = sqrt(dx*dx + dy*dy);
  int steps = dist/kCellLength;
  float step_x = dx/steps;
  float step_y = dy/steps;
  IndexT prev = from;

  for (int j = 0; j < steps; ++j) {
    float new_x = dev_cells[from].x + j*step_x;
    float new_y = dev_cells[from].y + j*step_y;
    assert(new_x >= 0 && new_x <= 1);
    assert(new_y >= 0 && new_y <= 1);
    IndexT next;

    if (curand_uniform(&state) < kProducerRatio) {
      next = new_ProducerCell(
          dev_cells[prev].max_velocity, new_x, new_y,
          curand(&state));
    } else {
      next = new_Cell(
          dev_cells[prev].max_velocity, new_x, new_y);
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
  dev_Cell_pos_x[idx] = dev_cells[self].x;
  dev_Cell_pos_y[idx] = dev_cells[self].y;
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
    if (dev_cells[i].type == kCellTypeProducer) {
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
    if (dev_cells[i].has_car) {
      Car_step_prepare_path(i);
    }
  }
}


__global__ void kernel_Car_commit_occupy() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].has_car || dev_cells[i].should_occupy) {
      Car_commit_occupy(i);
    }
  }
}


__global__ void kernel_Car_step_move() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].has_car) {
      Car_step_move(i);
    }
  }
}


__device__ int d_checksum;
__global__ void kernel_compute_checksum() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < d_num_cells; i += blockDim.x * gridDim.x) {
    if (dev_cells[i].has_car) {
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
  Cell* h_cells;
  cudaMalloc(&h_cells, sizeof(Cell)*kMaxNumCells);
  cudaMemcpyToSymbol(dev_cells, &h_cells, sizeof(Cell*),
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