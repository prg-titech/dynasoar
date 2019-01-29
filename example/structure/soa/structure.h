#ifndef EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
#define EXAMPLE_STRUCTURE_SOA_STRUCTURE_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"

// Pre-declare all classes.
class NodeBase;
class AnchorNode;
class AnchorPullNode;
class Node;
class Spring;

using AllocatorT = SoaAllocator<2*64*64*64*64, NodeBase, AnchorNode,
                                AnchorPullNode, Node, Spring>;


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

  __device__ void add_spring(Spring* spring);

  __device__ int num_springs() const { return num_springs_; }

  __device__ void remove_spring(Spring* spring);
};


class AnchorNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<>;

  __device__ AnchorNode(float pos_x, float pos_y);
};


class AnchorPullNode : public AnchorNode {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = AnchorNode;

  using FieldTypes = std::tuple<
      float,          // vel_x_
      float>;         // vel_y_

 private:
  SoaField<AnchorPullNode, 0> vel_x_;
  SoaField<AnchorPullNode, 1> vel_y_;

 public:
  __device__ AnchorPullNode(float pos_x, float pos_y,
                            float vel_x, float vel_y);

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
  __device__ Node(float pos_x, float pos_y, float mass);

  __device__ void move();
};


class Spring : public SoaBase<AllocatorT> {
 public:
  using FieldTypes = std::tuple<
      NodeBase*,      // p1_
      NodeBase*,      // p2_
      float,          // spring_factor_
      float,          // initial_length_
      float,          // force_
      float>;         // max_force_

 private:
  SoaField<Spring, 0> p1_;
  SoaField<Spring, 1> p2_;
  SoaField<Spring, 2> spring_factor_;
  SoaField<Spring, 3> initial_length_;
  SoaField<Spring, 4> force_;
  SoaField<Spring, 5> max_force_;

 public:
  __device__ Spring(NodeBase* p1, NodeBase* p2, float spring_factor,
                    float max_force);

  __device__ void compute_force();

  __device__ NodeBase* p1() const { return p1_; }

  __device__ NodeBase* p2() const { return p2_; }

  __device__ float force() const { return force_; }

  __device__ float max_force() const { return max_force_; }

  // For rendering purposes.
  __device__ void add_to_rendering_array();
};

#endif  // EXAMPLE_STRUCTURE_SOA_STRUCTURE_H
