#ifndef WA_TOR_AOS_WATOR_H
#define WA_TOR_AOS_WATOR_H

#include "allocator/soa_allocator.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace wa_tor {

// Custom std::array because std::array is not available on device.
template<typename T, size_t N>
class DevArray {
 private:
  T data[N];

 public:
  __device__ T& operator[](size_t pos) {
    return data[pos];
  }
};

class Agent;

class Cell {
 public:
  static const uint8_t kTypeId = 3;
  static const int kObjectSize = 56;
  static const uint8_t kBlockSize = 36;

 private:
  // left, top, right, bottom
  SoaField<DevArray<Cell*, 4>, 0, 0> neighbors_;
  __device__ Cell*& arr_neighbors(size_t index) {
    return ((DevArray<Cell*, 4>) neighbors_)[index];
  }

  SoaField<Agent*, 1, 32> agent_;

  SoaField<uint32_t, 2, 40> random_state_;

  // left, top, right, bottom, self
  SoaField<DevArray<bool, 5>, 3, 44> neighbor_request_;
  __device__ bool& arr_neighbor_request(size_t index) {
    return ((DevArray<bool, 5>) neighbor_request_)[index];
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

class Agent {
 protected:
  SoaField<Cell*, 0, 0> position_;
  SoaField<Cell*, 1, 8> new_position_;
  SoaField<uint32_t, 2, 16> random_state_;
  SoaField<uint8_t, 3, 20> type_identifier_;    // Custom alignment

 public:
  static const int kObjectSize = 24;
  static const uint8_t kBlockSize = 64;   // Never appears.

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
 private:
  SoaField<uint32_t, 4, 24> egg_timer_;

 public:
  static const int kObjectSize = 32;
  static const uint8_t kBlockSize = 64;

  static const uint8_t kTypeId = 1;

  __device__ Fish(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

class Shark : public Agent {
 private:
  SoaField<uint32_t, 4, 24> energy_;
  SoaField<uint32_t, 5, 28> egg_timer_;

 public:
  static const int kObjectSize = 32;
  static const uint8_t kBlockSize = 64;
  static const uint8_t kTypeId = 2;

  __device__ Shark(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

}  // namespace wa_tor

#endif  // WA_TOR_AOS_WATOR_H
