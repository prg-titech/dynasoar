
static const char kNoType = 0;
static const char kClassMale = 1;
static const char kClassFemale = 2;

__device__ curandState_t* dev_Cell_random_state;
// (No field for agent)
__device__ int* dev_Cell_sugar_diffusion;
__device__ int* dev_Cell_sugar;
__device__ int* dev_Cell_sugar_capacity;
__device__ int* dev_Cell_grow_rate;
// (No field for cell_id)
__device__ char* dev_Cell_Agent_type;
__device__ curandState_t* dev_Cell_Agent_random_state;
// (No field for cell)
__device__ int* dev_Cell_Agent_cell_request;
__device__ int* dev_Cell_Agent_vision;
__device__ int* dev_Cell_Agent_age;
__device__ int* dev_Cell_Agent_max_age;
__device__ int* dev_Cell_Agent_sugar;
__device__ int* dev_Cell_Agent_metabolism;
__device__ int* dev_Cell_Agent_endowment;
__device__ bool* dev_Cell_Agent_permission;
__device__ int* dev_Cell_Male_female_request;
__device__ bool* dev_Cell_Male_proposal_accepted;


__device__ new_Cell(int cell_id, int seed, int sugar, int sugar_capacity,
                    int max_grow_rate) {
  dev_Cell_sugar[cell_id] = sugar;
  dev_Cell_sugar_capacity[cell_id] = sugar_capacity;

  curand_init(seed, cell_id, 0, &dev_Cell_random_state[cell_id]);

  // Set random grow rate.
  float r = curand_uniform(&dev_Cell_random_state[cell_id]);

  if (r <= 0.01) {
    dev_Cell_grow_rate[cell_id] = max_grow_rate;
  } else if (r <= 0.05) {
    dev_Cell_grow_rate[cell_id] = 0.5*max_grow_rate;
  } else if (r <= 0.07) {
    dev_Cell_grow_rate[cell_id] = 0.25*max_grow_rate;
  } else {
    dev_Cell_grow_rate[cell_id] = 0;
  }
}


__device__ new_Agent(int cell_id, int vision, int age, int max_age,
                     int endowment, int metabolism) {
  assert(cell != kNullptr);
  dev_Cell_Agent_cell_request[cell_id] = kNullptr;
  dev_Cell_Agent_vision[cell_id] = vision;
  dev_Cell_Agent_age[cell_id] = age;
  dev_Cell_Agent_max_age[cell_id] = max_age;
  dev_Cell_Agent_sugar[cell_id] = endowment;
  dev_Cell_Agent_endowment[cell_id] = endowment;
  dev_Cell_Agent_metabolism[cell_id] = metabolism;
  dev_Cell_Agent_permission[cell_id] = false;

  curand_init(Cell_random_int(cell_id, 0, kSize*kSize), 0, 0,
  	          &dev_Cell_Agent_random_state[cell_id]);
}


__device__ new_Male(int cell_id, int vision, int age, int max_age,
                    int endowment, int metabolism) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_Cell_Male_proposal_accepted[cell_id] = false;
  dev_Cell_Male_female_request[cell_id] = kNullptr;
  dev_Cell_Agent_type[cell_id] = kClassMale;
}


__device__ new_Female(int cell_id, int vision, int age, int max_age,
                      int endowment, int metabolism) {
  new_Agent(cell_id, vision, age, max_age, endowment, metabolism);
  dev_Cell_Agent_type[cell_id] = kClassFemale;
}


__device__ void Agent_give_permission(int cell_id) {
  dev_Cell_Agent_permission[cell_id] = true;
}


__device__ void Agent_age_and_metabolize(int cell_id) {
  bool dead = false;

  dev_Cell_Agent_age[cell_id] = dev_Cell_Agent_age[cell_id] + 1;
  dead = dev_Cell_Agent_age[cell_id] > dev_Cell_Agent_max_age[cell_id];

  dev_Cell_Agent_sugar[cell_id] -= dev_Cell_Agent_metabolism[cell_id];
  dead = dead || dev_Cell_Agent_sugar[cell_id] <= 0;

  if (dead) {
    Cell_leave(cell_id);
    // No delete in baseline implementation.
  }
}


__device__ void Agent_prepare_move(int cell_id) {
  // Move to cell with the most sugar.
  int turn = 0;
  int target_cell = kNullptr;
  int target_sugar = 0;

  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -dev_Cell_Agent_vision[cell_id];
       dx < dev_Cell_Agent_vision[cell_id] + 1; ++dx) {
    for (int dy = -dev_Cell_Agent_vision[cell_id];
         dy < dev_Cell_Agent_vision[cell_id] + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if ((dx != 0 || dy != 0)
          && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        if (Cell_is_free(n_cell)) {
          if (dev_Cell_sugar[n_cell] > target_sugar) {
            target_cell = n_cell;
            target_sugar = dev_Cell_sugar[n_cell];
            turn = 1;
          } else if (dev_Cell_sugar[n_cell] == target_sugar) {
            // Select cell with probability 1/turn.
            if (Agent_random_float(cell_id) <= 1.0f/(++turn)) {
              target_cell = n_cell;
            }
          }
        }
      }
    }
  }

  dev_Cell_Agent_cell_request[cell_id] = target_cell;
}


__device__ void Agent_update_move(int cell_id) {
  if (dev_Cell_Agent_permission[cell_id] == true) {
    // Have permission to enter the cell.
    assert(dev_Cell_Agent_cell_request[cell_id] != kNullptr);
    assert(Cell_is_free(dev_Cell_Agent_cell_request[cell_id]));
    Cell_leave(cell_id);
    Cell_enter(dev_Cell_Agent_cell_request[cell_id], cell_id);
  }

  Agent_harvest_sugar(cell_id);

  dev_Cell_Agent_cell_request[cell_id] = kNullptr;
  dev_Cell_Agent_permission[cell_id] = false;
}


__device__ void Agent_harvest_sugar(int cell_id) {
  // Harvest as much sugar as possible.
  // TODO: Do we need two sugar fields here?
  dev_Cell_Agent_sugar[cell_id] += dev_Cell_sugar[cell_id];
  dev_Cell_sugar[cell_id] = 0;
}


__device__ bool Agent_ready_to_mate(int cell_id) {
  // Half of endowment of sugar will go to the child. And the parent still
  // needs some sugar to survive.
  return (dev_Cell_Agent_sugar[cell_id]
          >= dev_Cell_Agent_endowment[cell_id] * 2 / 3)
      && dev_Cell_Agent_age[cell_id] >= 18;
}


__device__ float Agent_random_float(int cell_id) {
  return curand_uniform(&dev_Cell_Agent_random_state[cell_id]);
}


__device__ void Cell_prepare_diffuse(int cell_id) {
  dev_Cell_sugar_diffusion[cell_id] =
      kSugarDiffusionRate * dev_Cell_sugar[cell_id];
  int max_diff = kMaxSugarDiffusion;
  if (dev_Cell_sugar_diffusion[cell_id] > max_diff) {
    dev_Cell_sugar_diffusion[cell_id] = max_diff;
  }

  dev_Cell_sugar[cell_id] -= dev_Cell_sugar_diffusion[cell_id];
}


__device__ void Cell_update_diffuse(int cell_id) {
  int new_sugar = 0;
  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        // Add sugar from neighboring 8 cells.
        new_sugar += 0.125f * dev_Cell_sugar_diffusion[n_cell];
      }
    }
  }

  dev_Cell_sugar[cell_id] += new_sugar;
}


__device__ float Cell_random_float(int cell_id) {
  return curand_uniform(&dev_Cell_random_state[cell_id]);
}


__device__ int Cell_random_int(int cell_id, int a, int b) {
  return curand(&dev_Cell_random_state[cell_id]) % (b - a) + a;
}


__device__ void Cell_decide_permission(int cell_id) {
  Agent* selected_agent = kNullptr;
  int turn = 0;
  int this_x = cell_id % kSize;
  int this_y = cell_id / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;

        if (dev_Cell_Agent_type[n_id] != kNoType
            && dev_Cell_Agent_cell_request[n_id] == cell_id) {
          ++turn;

          // Select cell with probability 1/turn.
          if (Cell_random_float(cell_id) <= 1.0f/turn) {
            selected_agent = n_id;
          } else {
            assert(turn > 1);
          }
        }
      }
    }
  }

  assert((turn == 0) == (selected_agent == kNullptr));

  if (selected_agent != kNullptr) {
    Agent_give_permission(selected_agent);
  }
}


// CONTINUE HERE


__device__ bool Cell::is_free() { return agent_ == nullptr; }


__device__ void Cell::enter(Agent* agent) {
  assert(agent_ == nullptr);
  assert(agent != nullptr);
  agent_ = agent;
}


__device__ void Cell::leave() {
  assert(agent_ != nullptr);
  agent_ = nullptr;
}


__device__ int Cell::sugar() { return sugar_; }


__device__ void Cell::take_sugar(int amount) { sugar_ -= amount; }


__device__ void Cell::grow_sugar() {
  // if(threadIdx.x == 0 && blockIdx.x == 0) {
  //   device_allocator->DBG_print_state_stats();
  // }

  sugar_ += min(sugar_capacity_ - sugar_, grow_rate_);
}


__device__ Agent* Cell::agent() { return agent_; }


__device__ void Male::propose() {
  if (ready_to_mate()) {
    // Propose to female with highest endowment.
    Female* target_agent = nullptr;
    int target_sugar = -1;

    int this_x = cell_->cell_id() % kSize;
    int this_y = cell_->cell_id() / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
          Cell* n_cell = cells[n_id];
          Female* n_female = n_cell->agent()->cast<Female>();

          if (n_female != nullptr && n_female->ready_to_mate()) {
            if (n_female->sugar() > target_sugar) {
              target_agent = n_female;
              target_sugar = n_female->sugar();
            }
          }
        }
      }
    }

    assert((target_sugar == -1) == (target_agent == nullptr));
    female_request_ = target_agent;
  }
}


__device__ void Male::accept_proposal() {
  proposal_accepted_ = true;
}


__device__ Female* Male::female_request() { return female_request_; }


__device__ void Male::propose_offspring_target() {
  if (proposal_accepted_) {
    assert(female_request_ != nullptr);

    // Select a random cell.
    Cell* target_cell = nullptr;
    int turn = 0;

    int this_x = cell_->cell_id() % kSize;
    int this_y = cell_->cell_id() / kSize;

    for (int dx = -vision_; dx < vision_ + 1; ++dx) {
      for (int dy = -vision_; dy < vision_ + 1; ++dy) {
        int nx = this_x + dx;
        int ny = this_y + dy;
        if ((dx != 0 || dy != 0)
            && nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
          int n_id = nx + ny*kSize;
          Cell* n_cell = cells[n_id];

          if (n_cell->is_free()) {
            ++turn;

            // Select cell with probability 1/turn.
            if (random_float() <= 1.0f/turn) {
              target_cell = n_cell;
            }
          }
        }
      }
    }

    assert((turn == 0) == (target_cell == nullptr));
    cell_request_ = target_cell;
  }
}


__device__ void Male::mate() {
  if (proposal_accepted_ && permission_) {
    assert(female_request_ != nullptr);
    assert(cell_request_ != nullptr);

    // Take sugar from endowment.
    int c_endowment = (endowment_ + female_request_->endowment()) / 2;
    sugar_ -= endowment_ / 2;
    female_request_->take_sugar(female_request_->endowment() / 2);

    // Calculate other properties.
    int c_vision = (vision_ + female_request_->vision()) / 2;
    int c_max_age = (max_age_ + female_request_->max_age()) / 2;
    int c_metabolism = (metabolism_ + female_request_->metabolism()) / 2;

    // Create agent.
    // TODO: Check why type cast is necessary here.
    // Otherwise: unspecified launch failure.
    Agent* child;
    if (random_float() <= 0.5f) {
      child = device_allocator->make_new<Male>(
          (Cell*) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    } else {
      child = device_allocator->make_new<Female>(
          (Cell*) cell_request_, c_vision, /*age=*/ 0, c_max_age, c_endowment,
          c_metabolism);
    }

    // Add agent to target cell.
    assert(cell_request_ != nullptr);
    assert(child != nullptr);
    assert(cell_request_->is_free());
    cell_request_->enter(child);
  }

  permission_ = false;
  proposal_accepted_ = false;
  female_request_ = nullptr;
  cell_request_ = nullptr;
}


__device__ void Female::decide_proposal() {
  Male* selected_agent = nullptr;
  int selected_sugar = -1;
  int this_x = cell_->cell_id() % kSize;
  int this_y = cell_->cell_id() / kSize;

  for (int dx = -kMaxVision; dx < kMaxVision + 1; ++dx) {
    for (int dy = -kMaxVision; dy < kMaxVision + 1; ++dy) {
      int nx = this_x + dx;
      int ny = this_y + dy;
      if (nx >= 0 && nx < kSize && ny >= 0 && ny < kSize) {
        int n_id = nx + ny*kSize;
        Cell* n_cell = cells[n_id];
        Male* n_male = n_cell->agent()->cast<Male>();

        if (n_male != nullptr) {
          if (n_male->female_request() == this
              && n_male->sugar() > selected_sugar) {
            selected_agent = n_male;
            selected_sugar = n_male->sugar();
          }
        }
      }
    }
  }

  assert((selected_sugar == -1) == (selected_agent == nullptr));

  if (selected_agent != nullptr) {
    selected_agent->accept_proposal();
  }
}
