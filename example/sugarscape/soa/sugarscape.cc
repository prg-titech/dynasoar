

__device__ Cell::Cell(int seed, int sugar, int sugar_capacity, int grow_rate,
                      int cell_id)
    : agent_(nullptr), sugar_(sugar), sugar_capacity_(sugar_capacity),
      grow_rate_(grow_rate), cell_id_(cell_id) {
  curand_init(seed, cell_id, 0, &random_state_);
}


__device__ Agent::Agent(Cell* cell, int vision, int age, int max_age,
                        int endowment, int metabolism)
    : cell_(cell), cell_request_(nullptr), vision_(vision), age_(age),
      max_age_(max_age), sugar_(endowment), endowment_(endowment),
      metabolism_(metabolism), permission_(false) {
  curand_init(cell->random_number(), 0, 0, &random_state_);
}


__device__ Male(Cell* cell, int vision, int age, int max_age, int endowment,
                int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism) {}


__device__ Female(Cell* cell, int vision, int age, int max_age, int endowment,
                  int metabolism)
    : Agent(cell, vision, age, max_age, endowment, metabolism) {}



__device__ void Agent::give_permission() { permission_ = true; }


__device__ void Agent::prepare_move() {
  // Move to cell with the most sugar.
  Cell* target_cell = nullptr;
  int target_sugar = 0;

  int this_x = cell_->cell_id() % kSizeX;
  int this_y = cell_->cell_id() / kSizeY;

  for (int dx = -vision_; dx < vision_ + 1; ++dx) {
    for (int dy = -vision; dy < vision_ + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
        int n_id = nx + ny*kSizeX;
        Cell* n_cell = cells[n_id];

        if (n_cell->is_free()) {
          if (n_cell->sugar() > target_sugar) {
            target_cell = n_cell;
            target_sugar = n_cell->sugar();
          }
        }
      }
    }
  }

  cell_request_ = target_cell;
}


__device__ void Cell::decide_permission() {
  Cell* selected_cell = nullptr;
  int turn = 0;
  int this_x = cell_id_ % kSizeX;
  int this_y = cell_id_ / kSizeY;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx > 0 && nx < kSizeX && ny > 0 && ny < kSizeY) {
        int n_id = nx + ny*kSizeX;
        Cell* n_cell = cells[n_id];

        if (n_cell->cell_request() == this) {
          ++turn;

          // Select cell with probability 1/turn.
          if (random_float() <= 1.0f/turn) {
            selected_cell = n_cell;
          }
        }
      }
    }
  }

  assert(turn == 0 || selected_cell != nullptr);

  if (selected_cell != nullptr) {
    selected_cell->give_permission();
  }
}

