#define SPAWN_THRESHOLD 5
#define ENERGY_BOOST 3
#define GRID_SIZE_X 400
#define GRID_SIZE_Y 300

__device__ uint32_t random_number(uint32_t* state, uint32_t max) {
  // Advance and return random state.
  // Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
  state = static_cast<uint32_t>(
      static_cast<uint64_t>(state) * 279470273u) % 0xfffffffb;
  return state % max;
}

class Cell {
 private:
  // left, top, right, bottom
  Cell* neighbors_[4];

  Agent* agent_ = nullptr;

  uint32_t random_state_;

  // left, top, right, bottom, self
  bool neighbor_request_[5];

 public:
  __device__ Agent* agent() const {
    return agent_;
  }

  __device__ bool is_free() const {
    return agent_ == nullptr;
  }

  __device__ bool has_fish() const {
    return agent_ != nullptr && dynamic_cast<Fish*>(agent_) != nullptr;
  }

  __device__ bool has_shark() const {
    return agent_ != nullptr && dynamic_cast<Shark*>(agent_) != nullptr;
  }

  template<bool(Cell::*predicate)(uint32_t* random_state)>
  __device__ bool request_random_neighbor() {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if ((neighbors_[i]->*predicate)()) {
        candidates[num_candidates++] = i;
      }
    }

    if (num_candidates == 0) {
      return false;
    } else {
      uint32_t selected_index = random_number(random_state, num_candidates);
      uint8_t selected = candidates[selected_index];
      uint8_t neighbor_index = (selected + 2) % 4;
      neighbors_[selected]->neighbor_request_[neighbor_index] = true;
      return true;
    }
  }

  __device__ void request_random_free_neighbor(uint32_t* random_state) {
    if (!request_random_neighbor<&Cell::is_free>(random_state)) {
      neighbor_request_[4] = true;
    }
  }

  __device__ void request_random_fish_neighbor(uint32_t* random_state) {
    if (!request_random_neighbor<&Cell::has_fish>(random_state)) {
      // No fish found. Look for free cell.
      if (!request_random_neighbor<&Cell::is_free>(random_state)) {
        neighbor_request_[4] = true;
      }
    }
  }

  __device__ void kill() {
    delete agent_;
    leave();
  }

  __device__ void leave() {
    agent_ = nullptr;
  }

  __device__ void enter(Agent* agent) {
    agent_ = agent;
    agent->position_ = this;
  }

  __device__ void decide() {
    if (neighbor_request_[4]) {
      // This cell has priority.
    } else {
      uint8_t candidates[4];
      uint8_t num_candidates = 0;

      for (int i = 0; i < 4; ++i) {
        if (neighbor_request_[i]) {
          candidates[num_candidates++] = i;
        }
      }

      if (num_candidates != 0) {
        uint32_t selected_index = random_number(random_state, num_candidates);
        uint8_t selected = candidates[selected_index];
        neighbors_[selected]->new_position_ = this;
      }
    }
  }
};


class Agent {
 private:
  Cell* position_;
  Cell* new_position_;
}

class Fish : public Agent {
 private:
  uint32_t egg_timer_;
  uint32_t random_state_;

 public:
  __device__ void prepare() {
    egg_timer_++;
    // Fallback: Stay on current cell.
    new_position_ = position_;
    position_->request_random_free_neighbor(&random_state_);
  }

  __device__ void update() {
    Cell* old_position = position_;

    if (old_position != new_position_) {
      old_position->leave();
      new_position_->enter(this);

      if (egg_timer_ > SPAWN_THRESHOLD) {
        old_position->enter(new Fish());
        egg_timer_ = 0;
      }
    }
  }
};


class Shark : public Agent {
 private:
  uint32_t energy_
  uint32_t egg_timer_;
  uint32_t random_state_;

 public:
  __device__ void prepare() {
    egg_timer_++;
    energy_--;

    if (energy_ == 0) {
      position_->kill();
    } else {
      // Fallback: Stay on current cell.
      new_position_ = position_;
      position_->request_random_fish_neighbor(&random_state_);
    }
  }

  __device__ void update() {
    Cell* old_position = position_;

    if (old_position != new_position_) {
      if (new_position_->has_fish()) {
        energy_ += ENERGY_BOOST;
        new_position_->kill();
      }

      old_position->leave();
      new_position_->enter(this);

      if (egg_timer_ > SPAWN_THRESHOLD) {
        old_position->enter(new Fish());
        egg_timer_ = 0;
      }
    }
  }
};

__device__ Cell*  cells[GRID_SIZE_X * GRID_SIZE_Y];

// Problem: It is not easy to keep track of all objects of a class if they are
// dynamically allocated. But we want to benchmark the performance of new/
// delete in CUDA.
// Solution: Fill these arrays in a separate kernel by iterating over all
// cells, storing agents in the respective array slots, and compacting the
// arrays. We do not measure the performance of these steps.
__device__ uint32_t num_sharks = 0;
__device__ Shark* sharks[GRID_SIZE_Y * GRID_SIZE_X];
__device__ uint32_t num_fish = 0;
__device__ Fish*  fish[GRID_SIZE_Y * GRID_SIZE_X];

// One thread per cell.
// TODO: Reset counters to zero before running the kernel.
__global__ void find_agents() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->has_shark()) {
      uint32_t idx = atomicAdd(&num_sharks, 1);
      sharks[idx] = dynamic_cast<Shark*>(cells[tid]->agent());
    } else if (cells[tid]->has_fish()) {
      uint32_t idx = atomicAdd(&num_fish, 1);
      fish[idx] = dynamic_cast<Fish*>(cells[tid]->agent());
    }
  }
}