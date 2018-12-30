
#include "configuration.h"
#include "rendering.h"
#include "traffic.h"


static const int kNumBlockSize = 256;

// TODO: Consider migrating to SoaAlloc.
TrafficLight* h_traffic_lights;
__device__ TrafficLight* d_traffic_lights;
Node* h_nodes;
__device__ Node* d_nodes;


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
int host_num_cells;


// Allocator handles.
__device__ AllocatorT* device_allocator;
AllocatorHandle<AllocatorT>* allocator_handle;


__device__ void Cell::occupy(Car* car) {
  assert(is_free());
  car_ = car;
}


__device__ void Cell::release() {
  assert(!is_free());
  car_ = nullptr;
}


__device__ void Car::step_prepare_path() {
  step_initialize_iteration();
  step_accelerate();
  step_extend_path();
  step_constraint_velocity();
  step_slow_down();
}


__device__ Cell* Car::next_step(Cell* position) {
  // Almost random walk.
  const uint32_t num_outgoing = position->num_outgoing();
  assert(num_outgoing > 0);

  // Need some kind of return statement here.
  return position->get_outgoing(random_int(0, num_outgoing));
}


__device__ void Car::step_initialize_iteration() {
  // Reset calculated path. This forces cars with a random moving behavior to
  // select a new path in every iteration. Otherwise, cars might get "stuck"
  // on a full network if many cars are waiting for the one in front of them in
  // a cycle.
  path_length_ = 0;
}


__device__ void Car::step_accelerate() {
  // Speed up the car by 1 or 2 units.
  int speedup = random_int(0, 2) + 1;
  velocity_ = max_velocity_ < velocity_ + speedup
      ? max_velocity_ : velocity_ + speedup;
}


__device__ void Car::step_extend_path() {
  Cell* cell = position_;
  Cell* next_cell;

  for (int i = 0; i < velocity_; ++i) {
    if (cell->is_sink()) {
      break;
    }

    next_cell = next_step(cell);
    assert(next_cell != cell);

    cell = next_cell;
    path_[i] = cell;
    path_length_ = path_length_ + 1;
  }

  velocity_ = path_length_;
}


__device__ void Car::step_constraint_velocity() {
  // This is actually only needed for the very first iteration, because a car
  // may be positioned on a traffic light cell.
  if (velocity_ > position()->max_velocity()) {
    velocity_ = position()->max_velocity();
  }

  int path_index = 0;
  int distance = 1;

  while (distance <= velocity_) {
    // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    Cell* next_cell = path_[path_index];

    // Avoid collision.
    if (!next_cell->is_free()) {
      // Cannot enter cell.
      --distance;
      velocity_ = distance;
      break;
    } // else: Can enter next cell.

    if (velocity_ > next_cell->max_velocity()) {
      // Car is too fast for this cell.
      if (next_cell->max_velocity() > distance - 1) {
        // Even if we slow down, we would still make progress.
        velocity_ = next_cell->max_velocity();
      } else {
        // Do not enter the next cell.
        --distance;
        assert(distance >= 0);

        velocity_ = distance;
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;

#ifndef NDEBUG
  for (int i = 0; i < velocity_; ++i) {
    assert(path_[i]->is_free());
    assert(i == 0 || path_[i - 1] != path_[i]);
  }
  // TODO: Check why the cast is necessary.
  assert(distance <= velocity());
#endif  // NDEBUG
}


__device__ void Car::step_move() {
  Cell* cell = position_;
  for (int i = 0; i < velocity_; ++i) {
    assert(path_[i] != cell);

    cell = path_[i];
    assert(cell->is_free());

    position()->release();
    cell->occupy(this);
    position_ = cell;
  }

  if (position()->is_sink() || position()->is_target()) {
    // Remove car from the simulation. Will be added again in the next
    // iteration.
    position()->release();
    device_allocator->free(this);
  }
}


__device__ void Car::step_slow_down() {
  // 20% change of slowdown.
  if (curand_uniform(&random_state_) < 0.2 && velocity_ > 0) {
    velocity_ = velocity_ - 1;
  }
}


__device__ void TrafficLight::step() {
  if (num_cells_ > 0) {
    timer_ = (timer_ + 1) % phase_time_;

    if (timer_ == 0) {
      assert(cells_[phase_] != nullptr);
      cells_[phase_]->set_current_max_velocity(0);
      phase_ = (phase_ + 1) % num_cells_;
      cells_[phase_]->remove_speed_limit();
    }
  }
}


__device__ void ProducerCell::create_car() {
  if (is_free()) {
    float r = curand_uniform(&random_state_);
    if (r < kCarAllocationRatio) {
      Car* new_car = device_allocator->make_new<Car>(
          /*seed=*/ curand(&random_state_), /*cell=*/ this,
          /*max_velocity=*/ curand(&random_state_) % (kMaxVelocity/2)
                            + kMaxVelocity/2);
    }
  }
}


__device__ Car::Car(int seed, Cell* cell, int max_velocity)
    : position_(cell), path_length_(0), velocity_(0),
      max_velocity_(max_velocity) {
  cell->occupy(this);
  curand_init(seed, 0, 0, &random_state_);
}


__device__ Cell::Cell(int max_velocity, float x, float y)
    : car_(nullptr), max_velocity_(max_velocity),
      current_max_velocity_(max_velocity),
      num_incoming_(0), num_outgoing_(0), x_(x), y_(y), is_target_(false) {
  atomicAdd(&dev_num_cells, 1);
}


__device__ void Cell::add_to_rendering_array() {
  int idx = atomicAdd(&dev_num_cells, 1);
  dev_Cell_pos_x[idx] = x_;
  dev_Cell_pos_y[idx] = y_;
  dev_Cell_occupied[idx] = !is_free();
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

    d_nodes[i].num_edges = curand(&state) % kMaxDegree + 1;
    d_nodes[i].num_incoming = 0;
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);
    d_nodes[i].x = x;
    d_nodes[i].y = y;

    for (int j = 0; j < d_nodes[i].num_edges; ++j) {
      d_nodes[i].cell_out[j] = device_allocator->make_new<Cell>(
          /*max_velocity=*/ curand(&state) % (kMaxVelocity/2)
                            + kMaxVelocity/2,
          x, y);
    }
  }
}


__global__ void kernel_create_edges() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    curandState_t state;
    curand_init(i, 0, 0, &state);

    for (int k = 0; k < d_nodes[i].num_edges; ++k) {
      int target = -1;
      while (true) {
        target = curand(&state) % kNumIntersections;
        int num_in = d_nodes[i].num_incoming;

        if (num_in < kMaxDegree) {
          // Try...
          if (atomicCAS(&d_nodes[i].num_incoming, num_in, num_in + 1) == num_in) {
            printf("Connect: %i --> %i\n", i, target);
            // Create edge.
            float dx = d_nodes[i].x - d_nodes[target].x;
            float dy = d_nodes[i].y - d_nodes[target].y;
            float dist = sqrt(dx*dx + dy*dy);
            int steps = dist/kCellLength;
            float step_x = dx/steps;
            float step_y = dy/steps;
            Cell* prev = d_nodes[i].cell_out[k];

            for (int j = 0; j < steps; ++j) {
              float new_x = d_nodes[i].x + j*step_x;
              float new_y = d_nodes[i].y + j*step_y;
              Cell* next;

              if (curand_uniform(&state) < kProducerRatio) {
                next = device_allocator->make_new<ProducerCell>(
                    /*max_velocity=*/ prev->max_velocity(), new_x, new_y,
                    curand(&state));
              } else {
                next = device_allocator->make_new<Cell>(
                    /*max_velocity=*/ prev->max_velocity(), new_x, new_y);
              }

              if (curand_uniform(&state) < kTargetRatio) {
                next->set_target();
              }

              prev->set_num_outgoing(1);
              prev->set_outgoing(0, next);
              next->set_num_incoming(1);
              next->set_incoming(0, prev);

              prev = next;
            }

            // Connect to all outgoing nodes.
            prev->set_num_outgoing(d_nodes[target].num_edges);
            for (int j = 0; j < d_nodes[target].num_edges; ++j) {
              prev->set_outgoing(j, d_nodes[target].cell_out[j]);
              d_nodes[target].cell_out[j]->set_incoming(num_in, prev);
            }
            d_nodes[target].cell_in[num_in] = prev;
          }
        }
      }
    }
  }
}


__global__ void kernel_create_traffic_lights() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kNumIntersections; i += blockDim.x * gridDim.x) {
    new(d_traffic_lights + i) TrafficLight(
        /*num_cells=*/ d_nodes[i].num_incoming,
        /*phase_time=*/ 5);

    for (int j = 0; j < d_nodes[i].num_edges; ++j) {
      d_nodes[i].cell_out[j]->set_num_incoming(d_nodes[i].num_incoming);
    }

    for (int j = 0; j < d_nodes[i].num_incoming; ++j) {
      d_traffic_lights[i].set_cell(j, d_nodes[i].cell_in[j]);
      d_nodes[i].cell_in[j]->set_current_max_velocity(0);  // Set to "red".
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
  cudaMemcpyFromSymbol(&host_num_cells, dev_num_cells, sizeof(int), 0,
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

  printf("Number of cells: %i\n", host_num_cells);
}


void step_traffic_lights() { 
  // TODO: Consider migrating this to SoaAlloc.
  kernel_traffic_light_step<<<
      (kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
      kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  allocator_handle->parallel_do<Cell, &Cell::add_to_rendering_array>();

  cudaMemcpy(host_data_Cell_pos_x, host_Cell_pos_x,
             sizeof(float)*host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_pos_y, host_Cell_pos_y,
             sizeof(float)*host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_occupied, host_Cell_occupied,
             sizeof(float)*host_num_cells, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaDeviceSynchronize());
}


void step() {
  allocator_handle->parallel_do<ProducerCell, &ProducerCell::create_car>();
  
  step_traffic_lights();
  allocator_handle->parallel_do<Car, &Car::step_prepare_path>();
  allocator_handle->parallel_do<Car, &Car::step_move>();
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

  create_street_network();

  for (int i = 0; i < kNumIterations; ++i) {
    if (kOptionRender) {
      transfer_data();
      draw(host_data_Cell_pos_x, host_data_Cell_pos_y, host_data_Cell_occupied,
           host_num_cells);
    }

    step();
  }

  if (kOptionRender) {
    close_renderer();
  }
}
