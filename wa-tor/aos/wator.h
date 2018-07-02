#ifndef WA_TOR_AOS_WATOR_H
#define WA_TOR_AOS_WATOR_H

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

class Agent;

class Cell {
 private:
  // left, top, right, bottom
  Cell* neighbors_[4];

  Agent* agent_;

  uint32_t random_state_;

  // left, top, right, bottom, self
  bool neighbor_request_[5];

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
  Cell* position_;
  Cell* new_position_;
  uint32_t random_state_;
  uint8_t type_identifier_;

 public:
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
  uint32_t egg_timer_;

 public:
  static const uint8_t kTypeId = 1;

  __device__ Fish(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

class Shark : public Agent {
 private:
  uint32_t energy_;
  uint32_t egg_timer_;

 public:
  static const uint8_t kTypeId = 2;

  __device__ Shark(uint32_t random_state);

  __device__ void prepare();

  __device__ void update();
};

}  // namespace wa_tor

#endif  // WA_TOR_AOS_WATOR_H
