#ifndef EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
#define EXAMPLE_STRUCTURE_SOA_STRUCTURE_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "configuration.h"

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
      DeviceArray<Spring*, kMaxDegree>,   // springs_
      float,                              // pos_x_
      float,                              // pos_y_
      int>;                               // num_springs_

 protected:
  SoaField<NodeBase, 0> springs_;
  SoaField<NodeBase, 1> pos_x_;
  SoaField<NodeBase, 2> pos_y_;
  SoaField<NodeBase, 3> num_springs_;

 public:
  __device__ NodeBase(float pos_x, float pos_y);

  __device__ float distance_to(NodeBase* other) const;

  __device__ float pos_x() const { return pos_x_; }

  __device__ float pos_y() const { return pos_y_; }
};


class AnchorNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<>;

  __device__ AnchorNode(float pos_x, float pos_y);

  __device__ void pull();
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

 public:
  __device__ Node(float pos_x, float pos_y);

  __device__ void move();
};


class Spring : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      NodeBase*,      // p1_
      NodeBase*,      // p2_
      float,          // spring_factor_
      float,          // initial_length_
      float>;         // force_

 private:
  SoaField<Spring, 0> p1_;
  SoaField<Spring, 1> p2_;
  SoaField<Spring, 2> spring_factor_;
  SoaField<Spring, 3> initial_length_;
  SoaField<Spring, 4> force_;

 public:
  __device__ Spring(NodeBase* p1, NodeBase* p2, float spring_factor);

  __device__ void compute_force();

  __device__ NodeBase* p1() const { return p1_; }

  __device__ NodeBase* p2() const { return p2_; }

  __device__ float force() const { return force_; }
};

#endif  // EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
