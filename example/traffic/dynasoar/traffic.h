#ifndef EXAMPLE_TRAFFIC_SOA_TRAFFIC_H
#define EXAMPLE_TRAFFIC_SOA_TRAFFIC_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"


// Pre-declare all classes.
class Car;
class Cell;
class ProducerCell;
class TrafficLight;

using AllocatorT = SoaAllocator<kNumObjects, Car, Cell, ProducerCell>;

class Cell : public AllocatorT::Base {
 public:
  declare_field_types(
      Cell,
      DeviceArray<Cell*, kMaxDegree>,         // incoming_
      DeviceArray<Cell*, kMaxDegree>,         // outgoing_
      Car*,                                   // car_
      int,                                    // max_velocity_
      int,                                    // current_max_velocity_
      int,                                    // num_incoming_
      int,                                    // num_outgoing_
      float,                                  // x_
      float,                                  // y_
      bool)                                   // is_target_;

 private:
  SoaField<Cell, 0> incoming_;
  SoaField<Cell, 1> outgoing_;
  SoaField<Cell, 2> car_;
  SoaField<Cell, 3> max_velocity_;
  SoaField<Cell, 4> current_max_velocity_;
  SoaField<Cell, 5> num_incoming_;
  SoaField<Cell, 6> num_outgoing_;
  SoaField<Cell, 7> x_;
  SoaField<Cell, 8> y_;
  SoaField<Cell, 9> is_target_;

 public:
  __device__ Cell(int max_velocity, float x, float y);

  __device__ int current_max_velocity() const { return current_max_velocity_; }

  __device__ int max_velocity() const { return max_velocity_; }

  __device__ void set_current_max_velocity(int v) { current_max_velocity_ = v; }

  __device__ void remove_speed_limit() { current_max_velocity_ = max_velocity_; }

  __device__ int num_incoming() const { return num_incoming_; }

  __device__ void set_num_incoming(int num) { num_incoming_ = num; }

  __device__ int num_outgoing() const { return num_outgoing_; }

  __device__ void set_num_outgoing(int num) { num_outgoing_ = num; }

  __device__ Cell* get_incoming(int idx) const { return incoming_[idx]; }

  __device__ void set_incoming(int idx, Cell* cell) {
    assert(cell != nullptr);
    incoming_[idx] = cell;
  }

  __device__ Cell* get_outgoing(int idx) const { return outgoing_[idx]; }

  __device__ void set_outgoing(int idx, Cell* cell) {
    assert(cell != nullptr);
    outgoing_[idx] = cell;
  }

  __device__ float x() const { return x_; }

  __device__ float y() const { return y_; }

  __device__ bool is_free() const { return car_ == nullptr; }

  __device__ bool is_sink() const { return num_outgoing_ == 0; }

  __device__ bool is_target() const { return is_target_; }

  __device__ void set_target() { is_target_ = true; }

  __device__ void occupy(Car* car);

  __device__ void release();

#ifdef OPTION_RENDER
  // Only for rendering.
  __device__ void add_to_rendering_array();
#endif  // OPTION_RENDER
};


class ProducerCell : public Cell {
 public:
  using BaseClass = Cell;

  declare_field_types(ProducerCell, curandState_t)  // random_state_

 private:
  SoaField<ProducerCell, 0> random_state_;

 public:
  __device__ ProducerCell(int max_velocity, float x, float y, int seed)
      : Cell(max_velocity, x, y) {
    curand_init(seed, 0, 0, &random_state_);
  }

  __device__ void create_car();
};


class Car : public AllocatorT::Base {
 public:
  declare_field_types(
      Car,
      curandState_t,                            // random_state_
      DeviceArray<Cell*, kMaxVelocity>,         // path_
      Cell*,                                    // position_
      int,                                      // path_length_
      int,                                      // velocity_
      int)                                      // max_velocity_

 private:
  SoaField<Car, 0> random_state_;
  SoaField<Car, 1> path_;
  SoaField<Car, 2> position_;
  SoaField<Car, 3> path_length_;
  SoaField<Car, 4> velocity_;
  SoaField<Car, 5> max_velocity_;

 public:
  __device__ Car(int seed, Cell* cell, int max_velocity);

  __device__ int velocity() const { return velocity_; }

  __device__ int max_velocity() const { return max_velocity_; }

  __device__ Cell* position() const { return position_; }

  __device__ void step_prepare_path();

  __device__ Cell* next_step(Cell* cell);

  __device__ void step_initialize_iteration();

  __device__ void step_accelerate();

  __device__ void step_extend_path();

  __device__ void step_constraint_velocity();

  __device__ void step_move();

  __device__ void step_slow_down();

  __device__ int random_int(int a, int b) {
    return curand(&random_state_) % (b - a) + a;
  }

  __device__ void compute_checksum();
};


// TODO: Migrating this to DynaSOAr. Not performance critical.
class TrafficLight {
 private:
  DeviceArray<Cell*, kMaxDegree> cells_;
  int num_cells_;
  int timer_;
  int phase_time_;
  int phase_;

 public:
  __device__ TrafficLight(int num_cells, int phase_time)
      : num_cells_(num_cells), timer_(0), phase_time_(phase_time), phase_(0) {}

  __device__ void set_cell(int idx, Cell* cell) {
    assert(cell != nullptr);
    cells_[idx] = cell;
  }

  __device__ void step();
};


#endif  // EXAMPLE_TRAFFIC_SOA_TRAFFIC_H
