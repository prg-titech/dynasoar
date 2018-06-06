
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

  template<bool(Cell::*predicate)()>
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
      uint32_t selected = random_number(num_candidates);
      uint8_t neighbor_index = (candidate_index + 2) % 4;
      candidates[selected]->neighbor_request_[candidate_index[selected]] = true;
      return true;
    }
  }

  void request_random_free_neighbor() {
    if (!request_random_neighbor<&Cell::is_free>()) {
      neighbor_request_[4] = true;
    }
  }

  void request_random_fish_neighbor() {
    if (!request_random_neighbor<&Cell::has_fish>()) {
      // No fish found. Look for free cell.
      if (!request_random_neighbor<&Cell::is_free>()) {
        neighbor_request_[4] = true;
      }
    }
  }

  uint32_t random_number(uint32_t max) {

  }
};
