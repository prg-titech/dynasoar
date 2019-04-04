#ifndef EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
#define EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "../configuration.h"

// Pre-declare all classes.
class NodeBase;
class BodyNode;
class TreeNode;

using AllocatorT = SoaAllocator<64*64*64*64, NodeBase, BodyNode, TreeNode>;


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
  SoaField<NodeBase, 0> parent_;
  SoaField<NodeBase, 1> pos_x_;
  SoaField<NodeBase, 2> pos_y_;
  SoaField<NodeBase, 3> mass_;
  SoaField<NodeBase, 4> child_index_;

// public:
  __DEV__ NodeBase(TreeNode* parent, double pos_x, double pos_y, double mass);

  __DEV__ TreeNode* parent() const { return parent_; }

  __DEV__ void set_parent(TreeNode* parent) { parent_ = parent; }

  __DEV__ double pos_x() const { return pos_x_; }

  __DEV__ double pos_y() const { return pos_y_; }

  __DEV__ double mass() const { return mass_; }

  __DEV__ void apply_force(BodyNode* body);

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ double distance_to(NodeBase* other);
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
  SoaField<BodyNode, 0> vel_x_;
  SoaField<BodyNode, 1> vel_y_;
  SoaField<BodyNode, 2> force_x_;
  SoaField<BodyNode, 3> force_y_;

 public:
  __DEV__ BodyNode(double pos_x, double pos_y, double vel_x, double vel_y,
                   double mass);

  __DEV__ void add_force(double x, double y) { force_x_ += x; force_y_ += y; }

  __DEV__ void add_to_tree();

  __DEV__ void clear_node();

  __DEV__ void compute_force();

  __DEV__ void update();

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ void add_checksum();

  __DEV__ void add_to_draw_array();
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
  SoaField<TreeNode, 0> children_;
  SoaField<TreeNode, 1> p1_x_;
  SoaField<TreeNode, 2> p1_y_;
  SoaField<TreeNode, 3> p2_x_;
  SoaField<TreeNode, 4> p2_y_;
  SoaField<TreeNode, 5> bfs_frontier_;
  SoaField<TreeNode, 6> bfs_done_;

 public:
  __DEV__ TreeNode(TreeNode* parent, double p1_x, double p1_y, double p2_x,
                   double p2_y);

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ int child_index(BodyNode* body);

  __DEV__ void collapse_tree();

  __DEV__ bool contains(NodeBase* node);

  __DEV__ bool contains(BodyNode* body);

  __DEV__ bool contains(TreeNode* node);

  __DEV__ TreeNode* make_child_tree_node(int c_idx);

  __DEV__ void remove(NodeBase* body);

  __DEV__ bool is_leaf();

  __DEV__ int num_direct_children();

  __DEV__ void initialize_frontier();

  __DEV__ void bfs_step();

  __DEV__ void update_frontier();

  __DEV__ void update_frontier_delete();

  __DEV__ void bfs_delete();

  __DEV__ void check_consistency();
};

#endif  // EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
