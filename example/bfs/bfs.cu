#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <cuda.h>
#include <limits>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <math.h>
#include <iterator>
#include <sys/time.h>

using namespace std;

#include "wa-tor/soa/wator.h"
#include "allocator/allocator_handle.h"

static const int kNumInlineNeighbors = 8;
static const int kNumMaxVertices = 32768;
static const int kNumMaxEdges = 32768;

class VertexActivation;

using AllocatorT = SoaAllocator<64*64*64*64, VertexActivation>;
__device__ AllocatorT* device_allocator;


class Vertex {
 private:
  int distance_;
  int num_neighbors_;
  int start_edge_;
  bool visited_;

 public:
  __DEV__ void visit();
};

__device__ Vertex dev_vertices[kNumMaxVertices];
__device__ int dev_edges[kNumMaxEdges];


void Vertex::visit() {
  visited_ = true;

  for (int i = 0; i < num_neighbors_; ++i) {
    Vertex* neighbor = &dev_neighbors[dev_edges[start_edge_ + i]];
    if (!neighbor->visited_) {
      neighbor->distance_ = distance_ + 1;
      device_allocator->make_new<VertexActivation>(neighbor);
    }
  }
}


class VertexActivation : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<Vertex*>;

 private:
  SoaField<VertexActivation, 0> vertex_;

 public:
  __DEV__ VertexActivation(Vertex* vertex) : vertex_(vertex) {}

  __DEV__ void visit() {
    vertex_->visit();
    device_allocator->free(this);
  }
};


int main() {
  char* filename = argv[1];
  int start_vertex = atoi(argv[2]);

  // Read in file.
  ifstream infile(filename);
  int from, to;
  int num_edges = 0;

  map<int, int> index_map;
  int next_index = 0;

  while (infile >> from >> to) {
    if (!index_map.count(from)) {
      index_map[from] = next_index++;
    }

    if (!index_map.count(to)) {
      index_map[to] = next_index++;
    }

    ++num_edges;
  }

  int num_vertices = next_index;

  printf("Input file has %d vertices and %i edges\n", num_vertices, num_edges);

  // Build adajacency lists (still reading file).
  infile.clear();
  infile.seekg(0, ios::beg);

  int *v_adj_begin = new int[num_vertices];
  int *v_adj_length = new int[num_vertices];
  vector<int> *v_adj_lists = new vector<int>[num_vertices]();
  int *v_adj_list = new int[num_edges];

  int max_degree = 0;

  while (infile >> from >> to) {
    v_adj_lists[index_map[from]].push_back(index_map[to]);
    max_degree = max(max_degree, (int) v_adj_lists[index_map[from]].size());
  }

  // Show degree distribution
  printf("Compute out-degree histogram\n");
  int *degree_histogram = new int[max_degree + 1]();
  unsigned long long total_degree = 0;

  for (int i = 0; i < num_vertices; i++) {
    degree_histogram[v_adj_lists[i].size()]++;
    total_degree += v_adj_lists[i].size();
  }

  double avg_degree = total_degree / (double) num_vertices;
  double degree_variance = 0.0;

  for (int i = 0; i < num_vertices; i++) {
    degree_variance += (avg_degree - v_adj_lists[i].size()) * (avg_degree - v_adj_lists[i].size());
  }
  degree_variance /= num_vertices;

  double degree_stddev = sqrt(degree_variance);

  // Compute median
  int *degs = new int[num_vertices];
  for (int i = 0; i < num_vertices; i++) {
    degs[i] = v_adj_lists[i].size();
  }

  printf("avg deg = %f, deg stddev = %f, median = %i\n",
         avg_degree, degree_stddev, degs[num_vertices / 2]);

  printf("Histogram for Vertex Degrees\n");
  for (int i = 0; i < max_degree + 1; i++) {
    printf("deg %i        %i\n", i, degree_histogram[i]);
  }

  // Generate data structure
  printf("Build ajacency lists\n");
  int next_offset = 0;

  for (int i = 0; i < num_vertices; i++) {
    int list_size = v_adj_lists[i].size();
    
    v_adj_begin[i] = next_offset;
    v_adj_length[i] = list_size;

    memcpy(v_adj_list + next_offset, &v_adj_lists[i][0], list_size * sizeof(int));
    next_offset += list_size;
  }
}