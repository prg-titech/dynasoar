#ifndef EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
#define EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H

#include <curand_kernel.h>

#include "allocator_config.h"
#include "configuration.h"

// Pre-declare all classes.
class NodeBase;
class BodyNode;
class TreeNode;

using AllocatorT = SoaAllocator<64*64*64*64, NodeBase, BodyNode, TreeNode>;


class NodeBase : public SoaBase<AllocatorT> {
 public:
  static const bool kIsAbstract = true;
  using FieldTypes = std::tuple<
      TreeNode*,      // parent_
      float,          // pos_x_
      float,          // pos_y_
      float>;         // mass_

 protected:
  SoaField<NodeBase, 0> parent_;
  SoaField<NodeBase, 1> pos_x_;
  SoaField<NodeBase, 2> pos_y_;
  SoaField<NodeBase, 3> mass_;

 public:
  __DEV__ NodeBase(TreeNode* parent, float pos_x, float pos_y, float mass);

  __DEV__ TreeNode* parent() const { return parent_; }

  __DEV__ void set_parent(TreeNode* parent) { parent_ = parent; }

  __DEV__ void cas_parent_retry(TreeNode* assumed, TreeNode* value);

  __DEV__ float pos_x() const { return pos_x_; }

  __DEV__ float pos_y() const { return pos_y_; }

  __DEV__ float mass() const { return mass_; }

  __DEV__ void apply_force(BodyNode* body);

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ float distance_to(NodeBase* other);
};


class BodyNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<
      float,          // vel_x_
      float,          // vel_y_
      float,          // force_x_
      float>;         // force_y_

 protected:
  SoaField<BodyNode, 0> vel_x_;
  SoaField<BodyNode, 1> vel_y_;
  SoaField<BodyNode, 2> force_x_;
  SoaField<BodyNode, 3> force_y_;

 public:
  __DEV__ BodyNode(float pos_x, float pos_y, float vel_x, float vel_y,
                   float mass);

  __DEV__ void add_force(float x, float y) { force_x_ += x; force_y_ += y; }

  __DEV__ void add_to_tree();

  __DEV__ void clear_node();

  __DEV__ void compute_force();

  __DEV__ void update();

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ void add_checksum();

  __DEV__ void add_to_draw_array();

   // Only for debugging.
  __DEV__ void sanity_check();
};


class TreeNode : public NodeBase {
 public:
  static const bool kIsAbstract = false;
  using BaseClass = NodeBase;

  using FieldTypes = std::tuple<
      DeviceArray<NodeBase*, 4>,                  // children_
      float,                                      // p1_x_
      float,                                      // p1_y_
      float,                                      // p2_x_
      float,                                      // p2_y_
      bool,                                       // frontier_
      bool,                                       // next_frontier_
      bool>;                                      // visited_

 private:
  //volatile SoaField<TreeNode, 0> volatile_children_;
  SoaField<TreeNode, 0> children_;

  SoaField<TreeNode, 1> p1_x_;
  SoaField<TreeNode, 2> p1_y_;
  SoaField<TreeNode, 3> p2_x_;
  SoaField<TreeNode, 4> p2_y_;
  SoaField<TreeNode, 5> frontier_;
  SoaField<TreeNode, 6> next_frontier_;
  SoaField<TreeNode, 7> visited_;

 public:
  __DEV__ TreeNode(TreeNode* parent, float p1_x, float p1_y, float p2_x,
                   float p2_y);

  __DEV__ void check_apply_force(BodyNode* body);

  __DEV__ int child_index(NodeBase* body);

  __DEV__ int compute_index(BodyNode* body);

  __DEV__ NodeBase* child(int idx) { return children_[idx]; }

  __DEV__ void collapse_tree();

  __DEV__ void insert(BodyNode* body);

  __DEV__ bool contains(BodyNode* body);

  __DEV__ void remove(NodeBase* body);

  __DEV__ bool remove_child(int c_idx, TreeNode* node);

  __DEV__ void remove_unvisited();

  __DEV__ bool is_leaf();

  __DEV__ void initialize_frontier();

  // TODO: Use iteration counter argument instead of second kernel when
  // supported by API.
  __DEV__ void bfs_step();

  __DEV__ void update_frontier();

   // Only for debugging.
  __DEV__ void sanity_check();
};

#endif  // EXAMPLE_BARNES_HUT_DYNAMIC_SOA_BARNES_HUT_H
