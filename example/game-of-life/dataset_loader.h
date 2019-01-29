#ifndef EXAMPLE_GAME_OF_LIFE_SOA_DATASET_LOADER_H
#define EXAMPLE_GAME_OF_LIFE_SOA_DATASET_LOADER_H


struct dataset_t {
  dataset_t() {}
  dataset_t(int x_p, int y_p, int* alive_cells_p, int num_alive_p)
      : x(x_p), y(y_p), alive_cells(alive_cells_p), num_alive(num_alive_p) {}

  int x;
  int y;
  int* alive_cells;
  int num_alive;
};


dataset_t load_from_file(char* filename);
dataset_t load_glider();


#endif  // EXAMPLE_GAME_OF_LIFE_SOA_DATASET_LOADER_H
