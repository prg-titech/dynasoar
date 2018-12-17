#ifndef EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
#define EXAMPLE_STRUCTURE_SOA_STRUCTURE_H

#include <curand_kernel.h>

#include "allocator_config.h"

// Pre-declare all classes.
class NodeBase;
class AnchorNode;
class Node;
class Spring;

using AllocatorT = SoaAllocator<64*64*64*64, NodeBase, AnchorNode, Node, Spring>;


class NodeBase : public SoaBase<AllocatorT> {
 public:
  static const bool kIsAbstract = true;
  using FieldTypes = std::tuple<
      float,          // pos_x_
      float,          // pos_y_
      int,            // start_edge_
      int>;           // num_edges_

 private:
  SoaField<NodeBase, 0> pos_x_;
  SoaField<NodeBase, 1> pos_y_;
  SoaField<NodeBase, 2> start_edge_;
  SoaField<NodeBase, 3> num_edges_;

 public:
  __DEV__ NodeBase(float pos_x, float pos_y);
};


class AnchorNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<>;

  __DEV__ AnchorNode(float pos_x, float pos_y);
};


class Node : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<
      float,          // vel_x_
      float,          // vel_y_
      float>;         // mass_

 private:
  SoaField<Node, 0> vel_x_;
  SoaField<Node, 1> vel_y_;
  SoaField<Node, 2> mass_;
};


class Spring : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      NodeBase*,      // p1_
      NodeBase*,      // p2_
      float,          // spring_factor_
      float,          // initial_length_
      float,          // force_x_
      float>;         // force_y_

 private:
  SoaField<Spring, 0> p1_;
  SoaField<Spring, 1> p2_;
  SoaField<Spring, 2> spring_factor_;
  SoaField<Spring, 3> initial_length_;
  SoaField<Spring, 4> force_x_;
  SoaField<Spring, 5> force_y_;
};

}
#endif  // EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
