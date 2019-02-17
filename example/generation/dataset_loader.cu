#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "dataset_loader.h"
#include "configuration.h"

using namespace std;

/*
dataset_t load_fireworks() {
  dataset_t result;
  result.x = 300;
  result.y = 300;

  // Create data set.
  int* cell_ids = new int[3];
  cell_ids[0] = 50 + 50*result.x;
  cell_ids[1] = 200 + 100*result.x;
  cell_ids[0] = 100 + 250*result.x;
  result.alive_cells = cell_ids;
  result.num_alive = 3;

  return result;
}
*/

static const int kBurstLen = 100;

void build_block(dataset_t& r, int dx, int dy) {
  r.set_displacement(dx, dy);
  for (int y = -kBurstLen; y <= -6; ++y) r.add(0, y);
  for (int x = -1; x <= 1; ++x) r.add(x, -5);
  for (int x = -1; x <= 1; ++x) r.add(x, -4);
  for (int x = -3; x <= 3; ++x) r.add(x, -3);
  for (int x = -3; x <= 3; ++x) r.add(x, -2);
  for (int x = -5; x <= 5; ++x) r.add(x, -1);
  for (int x = -kBurstLen; x <= kBurstLen; ++x) r.add(x, 0);
  for (int x = -5; x <= 5; ++x) r.add(x, 1);
  for (int x = -3; x <= 3; ++x) r.add(x, 2);
  for (int x = -3; x <= 3; ++x) r.add(x, 3);
  for (int x = -1; x <= 1; ++x) r.add(x, 4);
  for (int x = -1; x <= 1; ++x) r.add(x, 5);
  for (int y = 6; y <= kBurstLen; ++y) r.add(0, y);
}

dataset_t load_burst() {
  dataset_t r(/*x=*/ kSize, /*y=*/ kSize);

  for (int dx = kBurstLen+10; dx < r.x - kBurstLen - 10; dx += 2.1*kBurstLen) {
    for (int dy = kBurstLen+10; dy < r.y - kBurstLen - 10; dy += 2.1*kBurstLen) {
      build_block(r, dx, dy);
    }
  }

  for (int dx = kBurstLen+10 + 1.05*kBurstLen; dx < r.x - kBurstLen - 10; dx += 2.1*kBurstLen) {
    for (int dy = kBurstLen+10 + 1.05*kBurstLen; dy < r.y - kBurstLen - 10; dy += 2.1*kBurstLen) {
      build_block(r, dx, dy);
    }
  }

  return r;
}
