#ifndef WA_TOR_SOA_WATOR_H
#define WA_TOR_SOA_WATOR_H

#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"

namespace wa_tor {

// Pre-declare all classes.
class Agent;
class Cell;
class Fish;
class Shark;

using AllocatorT = SoaAllocator<64*64*64*64, Agent, Fish, Shark, Cell>;

class Cell : public SoaBase<AllocatorT> {
 public:
  static const uint8_t kTypeId = 3;

  using FieldTypes = std::tuple<
      DeviceArray<Cell*, 4>,         // neighbors_
      Agent*,                     // agent_
      uint32_t,                   // random_state_
      DeviceArray<bool, 5>>;         // neighbor_request_

  using BaseClass = void;
  static const bool kIsAbstract = false;

 private:
  // left, top, right, bottom
  SoaField<Cell, 0> neighbors_;
  __device__ Cell*& arr_neighbors(size_t index) {
    return ((DeviceArray<Cell*, 4>) neighbors_)[index];
  }

  SoaField<Cell, 1> agent_;

  SoaField<Cell, 2> random_state_;

  // left, top, right, bottom, self
  SoaField<Cell, 3> neighbor_request_;
  __device__ bool& arr_neighbor_request(size_t index) {
    return ((DeviceArray<bool, 5>) neighbor_request_)[index];
  }

 public:
  __device__ Cell(uint32_t random_state);

  __device__ Agent* agent() const;

  __device__ void decide();

  __device__ void enter(Agent* agent);

  __device__ bool has_fish() const;

  __device__ bool has_shark() const;

  __device__ bool is_free() const;

  __device__ void kill();

  __device__ void leave() ;

  __device__ void prepare();

  __device__ uint32_t* random_state();

  __device__ void set_neighbors(Cell* left, Cell* top,
                                Cell* right, Cell* bottom);

  __device__ void request_random_fish_neighbor();

  __device__ void request_random_free_neighbor();

  template<bool(Cell::*predicate)() const>
  __device__ bool request_random_neighbor(uint32_t* random_state);
};

class Agent : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      Cell*,            // position_
      Cell*,            // new_position_
      uint32_t,         // random_state_
      uint8_t>;         // type_identifier_

  using BaseClass = void;
  static const bool kIsAbstract = true;

 protected:
  SoaField<Agent, 0> position_;
  SoaField<Agent, 1> new_position_;
  SoaField<Agent, 2> random_state_;
  SoaField<Agent, 3> type_identifier_;    // Custom alignment

 public:
  // Type ID must correspond to variadic template.
  static const uint8_t kTypeId = 0;

  __device__ Agent(uint32_t random_state, uint8_t type_identifier);

  __device__ Cell* position() const;

  __device__ uint32_t* random_state();

  __device__ void set_new_position(Cell* new_pos);

  __device__ void set_position(Cell* cell);

  // TODO: Verify that RTTI (dynamic_cast) does not work in device code.
  __device__ uint8_t type_identifier() const;
};

class Fish : public Agent {
 public:
  using FieldTypes = std::tuple<
      uint32_t>;       // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Fish, 0> egg_timer_;

 public:
  static const uint8_t kTypeId = 1;

  __device__ Fish(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

class Shark : public Agent {
 public:
  using FieldTypes = std::tuple<
      uint32_t,        // energy_
      uint32_t>;       // egg_timer_

  using BaseClass = Agent;
  static const bool kIsAbstract = false;

 private:
  SoaField<Shark, 0> energy_;
  SoaField<Shark, 1> egg_timer_;

 public:
  static const uint8_t kTypeId = 2;

  __device__ Shark(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

}  // namespace wa_tor

#endif  // WA_TOR_SOA_WATOR_H
