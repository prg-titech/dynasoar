#ifndef EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
#define EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"

// Pre-declare all classes.
class NodeBase;
class BodyNode;
class TreeNode;

using AllocatorT = SoaAllocator<kNumObjects, NodeBase, BodyNode, TreeNode>;


class NodeBase : public AllocatorT::Base {
 public:
  static const bool kIsAbstract = true;

  declare_field_types(
      NodeBase,
      TreeNode*,      // parent_
      double,         // pos_x_
      double,         // pos_y_
      double,         // mass_
      int)            // child_index_

// TODO: Fix visibility.
// protected:
  Field<NodeBase, 0> parent_;
  Field<NodeBase, 1> pos_x_;
  Field<NodeBase, 2> pos_y_;
  Field<NodeBase, 3> mass_;
  Field<NodeBase, 4> child_index_;

// public:
  __device__ NodeBase(TreeNode* parent, double pos_x, double pos_y,
                      double mass);

  __device__ TreeNode* parent() const { return parent_; }

  __device__ void set_parent(TreeNode* parent) { parent_ = parent; }

  __device__ __host__ double pos_x() const { return pos_x_; }

  __device__ __host__ double pos_y() const { return pos_y_; }

  __device__ __host__ double mass() const { return mass_; }

  __device__ void apply_force(BodyNode* body);

  __device__ void check_apply_force(BodyNode* body);

  __device__ double distance_to(NodeBase* other);
};


class BodyNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  declare_field_types(
      BodyNode,
      double,          // vel_x_
      double,          // vel_y_
      double,          // force_x_
      double)          // force_y_

 protected:
  Field<BodyNode, 0> vel_x_;
  Field<BodyNode, 1> vel_y_;
  Field<BodyNode, 2> force_x_;
  Field<BodyNode, 3> force_y_;

 public:
  __device__ BodyNode(double pos_x, double pos_y, double vel_x, double vel_y,
                      double mass);

  __device__ void add_force(double x, double y) {
    force_x_ += x;
    force_y_ += y;
  }

  __device__ void add_to_tree();

  __device__ void clear_node();

  __device__ void compute_force();

  __device__ void update();

  __device__ void check_apply_force(BodyNode* body);

  __device__ void add_checksum();
};


class TreeNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  declare_field_types(
      TreeNode,
      DeviceArray<NodeBase*, 4>,                  // children_
      double,                                     // p1_x_
      double,                                     // p1_y_
      double,                                     // p2_x_
      double,                                     // p2_y_
      bool,                                       // bfs_frontier_
      bool)                                       // bfs_done_

// private:
  Field<TreeNode, 0> children_;
  Field<TreeNode, 1> p1_x_;
  Field<TreeNode, 2> p1_y_;
  Field<TreeNode, 3> p2_x_;
  Field<TreeNode, 4> p2_y_;
  Field<TreeNode, 5> bfs_frontier_;
  Field<TreeNode, 6> bfs_done_;

 public:
  __device__ TreeNode(TreeNode* parent, double p1_x, double p1_y, double p2_x,
                      double p2_y);

  __device__ void check_apply_force(BodyNode* body);

  __device__ int child_index(BodyNode* body);

  __device__ void collapse_tree();

  __device__ bool contains(NodeBase* node);

  __device__ bool contains(BodyNode* body);

  __device__ bool contains(TreeNode* node);

  __device__ TreeNode* make_child_tree_node(int c_idx);

  __device__ void remove(NodeBase* body);

  __device__ bool is_leaf();

  __device__ int num_direct_children();

  __device__ void initialize_frontier();

  __device__ void bfs_step();

  __device__ void update_frontier();

  __device__ void update_frontier_delete();

  __device__ void bfs_delete();

  // Only for debugging.
  __device__ void check_consistency();

#ifdef OPTION_RENDER
  __device__ __host__ float p1_x() { return p1_x_; }
  __device__ __host__ float p1_y() { return p1_y_; }
  __device__ __host__ float p2_x() { return p2_x_; }
  __device__ __host__ float p2_y() { return p2_y_; }
#endif  // OPTION_RENDER
};

#endif  // EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
