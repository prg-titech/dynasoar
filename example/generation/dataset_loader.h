#ifndef EXAMPLE_GENERATION_DATASET_LOADER_H
#define EXAMPLE_GENERATION_DATASET_LOADER_H

#include <vector>

struct dataset_t {
  dataset_t() {}
  dataset_t(int x_p, int y_p) : x(x_p), y(y_p) {}

  int x;
  int y;
  std::vector<int> alive_cells;
  int disp_x = 0;
  int disp_y = 0;

  void set_displacement(int px, int py) {
    disp_x = px;
    disp_y = py;
  }

  void add(int px, int py) {
    alive_cells.push_back(px + disp_x + (py + disp_y)*x);
  }
};


dataset_t load_burst();


#endif  // EXAMPLE_GENERATION_DATASET_LOADER_H
