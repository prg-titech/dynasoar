#include <stdio.h>
#include <assert.h>

#include <SDL2/SDL2_gfxPrimitives.h>


#define SPAWN_THRESHOLD 5
#define ENERGY_BOOST 3
#define ENERGY_START 5
#define GRID_SIZE_X 400
#define GRID_SIZE_Y 300

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ uint32_t random_number(uint32_t& state, uint32_t max) {
  // Advance and return random state.
  // Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
  state = static_cast<uint32_t>(
      static_cast<uint64_t>(state) * 279470273u) % 0xfffffffb;
  return state % max;
}

class Agent;
class Fish;
class Shark;

class Cell {
 private:
  // left, top, right, bottom
  Cell* neighbors_[4];

  Agent* agent_ = nullptr;

  uint32_t random_state_;

  // left, top, right, bottom, self
  bool neighbor_request_[5];

 public:
  __device__ Cell(uint32_t random_state) : random_state_(random_state) {
    prepare();
  }

  __device__ void prepare() {
    for (int i = 0; i < 5; ++i) {
      neighbor_request_[i] = false;
    }
  }

  __device__ Agent* agent() const {
    return agent_;
  }

  __device__ bool is_free() const {
    return agent_ == nullptr;
  }

  __device__ bool has_fish() const;

  __device__ bool has_shark() const;

  __device__ uint32_t& random_state() {
    return random_state_;
  }

  __device__ void set_neighbors(Cell* left, Cell* top,
                                Cell* right, Cell* bottom) {
    neighbors_[0] = left;
    neighbors_[1] = top;
    neighbors_[2] = right;
    neighbors_[3] = bottom;
  }

  template<bool(Cell::*predicate)() const>
  __device__ bool request_random_neighbor(uint32_t& random_state) {
    uint8_t candidates[4];
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if (neighbors_[i] != nullptr) {
        // Handling of border cells.
        if ((neighbors_[i]->*predicate)()) {
          candidates[num_candidates++] = i;
        }
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

  __device__ void request_random_free_neighbor(uint32_t& random_state) {
    if (!request_random_neighbor<&Cell::is_free>(random_state)) {
      neighbor_request_[4] = true;
    }
  }

  __device__ void request_random_fish_neighbor(uint32_t& random_state) {
    if (!request_random_neighbor<&Cell::has_fish>(random_state)) {
      // No fish found. Look for free cell.
      if (!request_random_neighbor<&Cell::is_free>(random_state)) {
        neighbor_request_[4] = true;
      }
    }
  }

  __device__ void kill();

  __device__ void leave() {
    agent_ = nullptr;
  }

  __device__ void enter(Agent* agent);

  __device__ void decide();
};


class Agent {
 protected:
  Cell* position_;
  Cell* new_position_;
  uint32_t random_state_;
  uint8_t type_identifier_;

 public:
  __device__ Agent(uint32_t random_state, uint8_t type_identifier)
      : random_state_(random_state), type_identifier_(type_identifier) {}

  __device__ uint32_t& random_state() {
    return random_state_;
  }

  __device__ void set_new_position(Cell* new_pos) {
    new_position_ = new_pos;
  }

  __device__ Cell* position() const {
    return position_;
  }

  __device__ void set_position(Cell* cell) {
    position_ = cell;
  }

  // TODO: Verify that RTTI (dynamic_cast) does not work in device code.
  __device__ uint8_t type_identifier() const {
    return type_identifier_;
  }
};

__device__ void Cell::enter(Agent* agent) {
  agent_ = agent;
  agent->set_position(this);
}


class Fish : public Agent {
 private:
  uint32_t egg_timer_;

 public:
  static const uint8_t kTypeId = 1;

  __device__ Fish(uint32_t random_state) : Agent(random_state, kTypeId) {}

  __device__ void prepare() {
    egg_timer_++;
    // Fallback: Stay on current cell.
    new_position_ = position_;

    assert(position_ != nullptr);
    position_->request_random_free_neighbor(random_state_);
  }

  __device__ void update() {
    Cell* old_position = position_;

    if (old_position != new_position_) {
      old_position->leave();
      new_position_->enter(this);

      if (egg_timer_ > SPAWN_THRESHOLD) {
        old_position->enter(new Fish(random_state_ + 1));
        egg_timer_ = 0;
      }
    }
  }
};


class Shark : public Agent {
 private:
  uint32_t energy_;
  uint32_t egg_timer_;
  uint32_t random_state_;

 public:
  static const uint8_t kTypeId = 2;

  __device__ Shark(uint32_t random_state) : Agent(random_state, kTypeId),
                                            energy_(ENERGY_START) {}

  __device__ void prepare() {
    egg_timer_++;
    energy_--;

    assert(position_ != nullptr);
    if (energy_ == 0) {
      position_->kill();
    } else {
      // Fallback: Stay on current cell.
      new_position_ = position_;
      position_->request_random_fish_neighbor(random_state_);
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
        old_position->enter(new Shark(random_state_ + 1));
        egg_timer_ = 0;
      }
    }
  }
};

__device__ bool Cell::has_fish() const {
  return agent_ != nullptr && agent_->type_identifier() == Fish::kTypeId;
}

__device__ bool Cell::has_shark() const {
  return agent_ != nullptr && agent_->type_identifier() == Shark::kTypeId;
}

__device__ void Cell::kill() {
  delete agent_;
  agent_ = nullptr;
}

__device__ void Cell::decide() {
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
      uint32_t selected_index = random_number(random_state_, num_candidates);
      uint8_t selected = candidates[selected_index];
      neighbors_[selected]->agent()->set_new_position(this);
    }
  }
}



__device__ Cell* cells[GRID_SIZE_X * GRID_SIZE_Y];

__global__ void create_cells() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    Cell* new_cell = new Cell(tid + 1);
    assert(new_cell != nullptr);

    cells[tid] = new_cell;
  }
}

__global__ void setup_cells() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    int x = tid % GRID_SIZE_X;
    int y = tid / GRID_SIZE_X;

    Cell* left = x > 0 ? cells[y*GRID_SIZE_X + x - 1] : nullptr;
    Cell* right = x < GRID_SIZE_X - 1 ? cells[y*GRID_SIZE_X + x + 1] : nullptr;
    Cell* top = y > 0 ? cells[(y - 1)*GRID_SIZE_X + x] : nullptr;
    Cell* bottom = y < GRID_SIZE_Y - 1 ? cells[(y + 1)*GRID_SIZE_X + x]
                                       : nullptr;

    // left, top, right, bottom
    cells[tid]->set_neighbors(left, top, right, bottom);

    // Initialize with random agent.
    uint32_t agent_type = random_number(cells[tid]->random_state(), 3);
    if (agent_type == 0) {
      cells[tid]->enter(new Fish(tid + 10001));
    } else if (agent_type == 1) {
      cells[tid]->enter(new Shark(tid + 20001));
    } else {
      // Free cell.
    }
  }
}

void initialize() {
  create_cells<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
  setup_cells<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

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

__global__ void reset_agent_arrays() {
  num_sharks = 0;
  num_fish = 0;
}

// One thread per cell.
// TODO: Reset counters to zero before running the kernel.
__global__ void find_agents() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->has_shark()) {
      uint32_t idx = atomicAdd(&num_sharks, 1);
      sharks[idx] = reinterpret_cast<Shark*>(cells[tid]->agent());
    } else if (cells[tid]->has_fish()) {
      uint32_t idx = atomicAdd(&num_fish, 1);
      fish[idx] = reinterpret_cast<Fish*>(cells[tid]->agent());
    }
  }
}

void generate_agent_arrays() {
  reset_agent_arrays<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  find_agents<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__global__ void cell_prepare() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    cells[tid]->prepare();
  }
}

__global__ void cell_decide() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    cells[tid]->decide();
  }
}

__global__ void fish_prepare() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < num_fish) {
    assert(fish[tid] != nullptr);
    fish[tid]->prepare();
  }
}

__global__ void fish_update() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < num_fish) {
    fish[tid]->update();
  }
}

__global__ void shark_prepare() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < num_sharks) {
    assert(sharks[tid] != nullptr);
    sharks[tid]->prepare();
  }
}

__global__ void shark_update() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < num_sharks) {
    sharks[tid]->update();
  }
}

void step() {
  generate_agent_arrays();

  cell_prepare<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  fish_prepare<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cell_decide<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  fish_update<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cell_prepare<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  shark_prepare<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cell_decide<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());

  shark_update<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__device__ uint32_t d_gui_map[GRID_SIZE_Y * GRID_SIZE_X];
uint32_t gui_map[GRID_SIZE_Y * GRID_SIZE_X];

__global__ void fill_gui_map() {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < GRID_SIZE_Y*GRID_SIZE_X) {
    if (cells[tid]->agent() != nullptr) {
      d_gui_map[tid] = cells[tid]->agent()->type_identifier();
    } else {
      d_gui_map[tid] = 0;
    }
  }
}

void update_gui_map() {
  fill_gui_map<<<GRID_SIZE_X*GRID_SIZE_Y/1024 + 1, 1024>>>();
  cudaDeviceSynchronize();

  cudaMemcpy(gui_map, d_gui_map, sizeof(uint32_t)*GRID_SIZE_X*GRID_SIZE_Y,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

// SDL helper variables.
SDL_Window* window_;
SDL_Renderer* renderer_;

void render() {
  update_gui_map();

  int num_fish = 0;
  int num_sharks = 0;

  for (int i = 0; i < GRID_SIZE_X*GRID_SIZE_Y; ++i) {
    int x = i % GRID_SIZE_X;
    int y = i / GRID_SIZE_X;

    if (gui_map[i] == Fish::kTypeId) {
      pixelRGBA(renderer_, x, y, 0, 255, 0, 255);
      num_fish++;
    } else if (gui_map[i] == Shark::kTypeId) {
      pixelRGBA(renderer_, x, y, 255, 0, 0, 255);
      num_sharks++;
    } else {
      pixelRGBA(renderer_, x, y, 0, 0, 0, 255);
    }
  }

  SDL_RenderPresent(renderer_);

  printf("Fish: %i, Sharks: %i\n", num_fish, num_sharks);
}

int main(int argc, char* arvg[]) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);

  size_t heap_size;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  printf("CUDA heap size: %lu\n", heap_size);

  // Initialize renderer.
  if (SDL_Init(SDL_INIT_VIDEO)) {
    printf("SDL_Init Error: %s", SDL_GetError());
    exit(1);
  }

  window_ = SDL_CreateWindow("Wa-Tor", 100, 100,
                             GRID_SIZE_X, GRID_SIZE_Y, SDL_WINDOW_OPENGL);
  if (window_ == NULL) { 
    printf("SDL_CreateWindow Error: %s", SDL_GetError());
    SDL_Quit();
    exit(2);
  }

  renderer_ = SDL_CreateRenderer(window_, -1,
      SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (renderer_ == NULL) { 
    SDL_DestroyWindow(window_);
    printf("SDL_CreateRenderer Error: %s", SDL_GetError());
    SDL_Quit();
    exit(3);
  }

  // Draw black background.
  SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 0);
  SDL_RenderClear(renderer_);
  SDL_RenderPresent(renderer_);

  initialize();
  render();

  while (true) {
    SDL_Event e;
    if (SDL_PollEvent(&e)) {
      switch (e.type) {
        case SDL_QUIT:
          exit(0);
          break;
      }
    }

    step();
    render();
  }
}