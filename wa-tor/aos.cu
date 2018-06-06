#define SPAWN_THRESHOLD 5
#define ENERGY_BOOST 3

class Cell {
 private:
  // left, top, right, bottom
  Cell* neighbors_[4];

  Agent* agent_ = nullptr;

  uint32_t random_state_;

  // left, top, right, bottom, self
  bool neighbor_request_[5];

 public:
  bool is_free() const {
    return agent_ == nullptr;
  }

  bool has_fish() const {
    return agent_ != nullptr && dynamic_cast<Fish*>(agent_) != nullptr;
  }

  template<bool(Cell::*predicate)(uint32_t* random_state)>
  bool request_random_neighbor() {
    Cell* candidates[4];
    uint8_t candidate_index;
    uint8_t num_candidates = 0;

    for (int i = 0; i < 4; ++i) {
      if ((neighbors_[i]->*predicate)()) {
        candidates[num_candidates] = neighbors_[i];
        candidate_index[num_candidates++] = i;
      }
    }

    if (num_candidates == 0) {
      return false;
    } else {
      uint32_t selected = random_number(random_state, num_candidates);
      uint8_t neighbor_index = (candidate_index + 2) % 4;
      candidates[selected]->neighbor_request_[candidate_index[selected]] = true;
      return true;
    }
  }

  void request_random_free_neighbor(uint32_t* random_state) {
    if (!request_random_neighbor<&Cell::is_free>(random_state)) {
      neighbor_request_[4] = true;
    }
  }

  void request_random_fish_neighbor(uint32_t* random_state) {
    if (!request_random_neighbor<&Cell::has_fish>(random_state)) {
      // No fish found. Look for free cell.
      if (!request_random_neighbor<&Cell::is_free>(random_state)) {
        neighbor_request_[4] = true;
      }
    }
  }

  void kill() {
    delete agent_;
    leave();
  }

  void leave() {
    agent_ = nullptr;
  }

  void enter(Agent* agent) {
    agent_ = agent;
    agent->position_ = this;
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
  void prepare() {
    egg_timer_++;
    position_->request_random_free_neighbor(&random_state_);
  }

  void update() {
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
  void prepare() {
    egg_timer_++;
    energy_--;

    if (energy_ == 0) {
      position_->kill();
    } else {
      position_->request_random_fish_neighbor(&random_state_);
    }
  }

  void update() {
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
