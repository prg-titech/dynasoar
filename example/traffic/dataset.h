#ifndef EXAMPLE_TRAFFIC_DATASET_H
#define EXAMPLE_TRAFFIC_DATASET_H

// Only for creating the street network. Should be loaded from file.
template<typename CellPointerT>
struct NodeTemplate {
  int num_outgoing;
  int num_incoming;

  CellPointerT cell_out[kMaxDegree];
  CellPointerT cell_in[kMaxDegree];

  int node_out[kMaxDegree];
  int node_out_pos[kMaxDegree];

  float x, y;
};
using Node = NodeTemplate<CellPointerT>;


// Storage for nodes.
Node* h_nodes;
__device__ Node* d_nodes;


float random_float() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}


void create_network_structure() {
  srand(kSeed);

  Node* node_data = (Node*) malloc(sizeof(Node)*kNumIntersections);

  // Create nodes.
  for (int i = 0; i < kNumIntersections; ++i) {
    node_data[i].num_outgoing = rand() % kMaxDegree + 1;
    node_data[i].num_incoming = 0;
    node_data[i].x = random_float();
    node_data[i].y = random_float();
  }

  // Create edges.
  for (int i = 0; i < kNumIntersections; ++i) {
    for (int k = 0; k < node_data[i].num_outgoing; ++k) {
      int target = -1;
      while (true) {
        target = rand() % kNumIntersections;

        if (target != i) {
          if (node_data[target].num_incoming < kMaxDegree) {
            node_data[i].node_out[k] = target;
            node_data[i].node_out_pos[k] = node_data[target].num_incoming;
            ++node_data[target].num_incoming;
            break;
          }
        }
      }
    }
  }

  // Copy data to GPU.
  cudaMemcpy(h_nodes, node_data, sizeof(Node)*kNumIntersections,
             cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());
}

#endif  // EXAMPLE_TRAFFIC_DATASET_H
