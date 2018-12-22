#include <chrono>
#include <curand_kernel.h>

#include "barnes_hut.h"
#include "configuration.h"


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Root of the quad tree.
__DEV__ TreeNode* tree;


template<typename T>
__DEV__ T* pointerCAS(T** addr, T* assumed, T* value) {
  auto* i_addr = reinterpret_cast<unsigned long long int*>(addr);
  auto i_assumed = reinterpret_cast<unsigned long long int>(assumed);
  auto i_value = reinterpret_cast<unsigned long long int>(value);
  return reinterpret_cast<T*>(atomicCAS(i_addr, i_assumed, i_value));
}


__DEV__ NodeBase::NodeBase(TreeNode* parent, float pos_x, float pos_y,
                           float mass)
    : parent_(parent), pos_x_(pos_x), pos_y_(pos_y), mass_(mass) {}


__DEV__ BodyNode::BodyNode(float pos_x, float pos_y, float vel_x, float vel_y,
                           float mass)
    : NodeBase(/*parent=*/ nullptr, pos_x, pos_y, mass),
      vel_x_(vel_x), vel_y_(vel_y) {}


__DEV__ TreeNode::TreeNode(TreeNode* parent, float p1_x, float p1_y,
                           float p2_x, float p2_y)
    : NodeBase(parent, 0.0f, 0.0f, 0.0f),
      p1_x_(p1_x), p1_y_(p1_y), p2_x_(p2_x), p2_y_(p2_y) {}


__DEV__ float NodeBase::distance_to(NodeBase* other) {
  float dx = other->pos_x() - pos_x_;
  float dy = other->pos_y() - pos_y_;
  return sqrt(dx*dx + dy*dy);
}


__DEV__ void NodeBase::apply_force(BodyNode* body) {
  // Update `body`.
  if (body != this) {
    float dx = body->pos_x() - pos_x_;
    float dy = body->pos_y() - pos_y_;
    float dist = sqrt(dx*dx + dy*dy);
    float F = kGravityConstant * mass_ * body->mass()
        / (dist * dist + kDampeningFactor);
    body->add_force(F*dx / dist, F*dy / dist);
  }
}


__DEV__ void BodyNode::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;

  // TODO: We may need a while loop here instead of recursion.
  tree->check_apply_force(this);
}


__DEV__ void NodeBase::check_apply_force(BodyNode* body) {
  // TODO: This function should be virtual but we do not have native support
  // for virtual functions in SoaAlloc yet.
  TreeNode* tree_node = this->cast<TreeNode>();
  if (tree_node != nullptr) {
    tree_node->check_apply_force(body);
  } else {
    BodyNode* body_node = this->cast<BodyNode>();
    if (body_node != nullptr) {
      body_node->check_apply_force(body);
    } else {
      assert(false);
    }
  }
}


__DEV__ void TreeNode::check_apply_force(BodyNode* body) {
  if (distance_to(body) <= kDistThreshold) {
    // Too close. Recurse.
    for (int i = 0; i < 4; ++i) {
      if (children_[i] != nullptr) {
        children_[i]->check_apply_force(body);
      }
    }
  } else {
    // Far enough away to use approximation.
    apply_force(body);
  }
}


__DEV__ void BodyNode::check_apply_force(BodyNode* body) {
  apply_force(body);
}


__DEV__ void BodyNode::update() {
  vel_x_ += force_x_*kDt / mass_;
  vel_y_ += force_y_*kDt / mass_;
  pos_x_ += vel_x_*kDt;
  pos_y_ += vel_y_*kDt;

  if (pos_x_ < -1 || pos_x_ > 1) {
    vel_x_ = -vel_x_;
  }

  if (pos_y_ < -1 || pos_y_ > 1) {
    vel_y_ = -vel_y_;
  }
}


__DEV__ void BodyNode::clear_node() {
  assert(parent_ != nullptr);

  if (!parent_->is_inside(this)) {
    parent_->remove(this);
    parent_ = nullptr;
  }
}


__DEV__ void TreeNode::remove(BodyNode* body) {
  for (int i = 0; i < 4; ++i) {
    if (children_[i] == body) {
      children_[i] = nullptr;
      return;
    }
  }

  // BodyNode not found.
  assert(false);
}


__DEV__ void BodyNode::add_to_tree() {
  if (parent_ == nullptr) {
    tree->insert(this);
  }
}


__DEV__ int TreeNode::child_index(BodyNode* body) {
  assert(is_inside(body));

  int c_idx = 0;
  if (body->pos_x() > (p1_x_ + p2_x_) / 2) c_idx = 1;
  if (body->pos_y() > (p1_y_ + p2_y_) / 2) c_idx += 2;
  return c_idx;
}


__DEV__ void TreeNode::insert(BodyNode* body) {
  assert(is_inside(body));
  TreeNode* current = this;

  while (true) {
    // TODO: Need a threadfence here to see updates of current?
    // Probably not because atomic operation should update cache.

    // Check where to insert in this node.
    int c_idx = current->child_index(body);
    auto*& child = current->children_[c_idx];

    if (child == nullptr) {
      if (pointerCAS<NodeBase>(&child, nullptr, body) == nullptr) {
        return;
      }
    } else if (child->cast<TreeNode>() != nullptr) {
      current = child->cast<TreeNode>();
    } else {
      // TODO: Maybe threadfence here or atomic read to avoid false reads?
      BodyNode* other = child->cast<BodyNode>();
      assert(other != nullptr);

      // Replace BodyNode with TreeNode.
      float new_p1_x = c_idx == 0 || c_idx == 2
          ? current->p1_x_ : (current->p1_x_ + current->p2_x_) / 2;
      float new_p2_x = c_idx == 0 || c_idx == 2
          ? (current->p1_x_ + current->p2_x_) / 2 : current->p2_x_;
      float new_p1_y = c_idx == 0 || c_idx == 1
          ? current->p1_y_ : (current->p1_y_ + current->p2_y_) / 2;
      float new_p2_y = c_idx == 0 || c_idx == 1
          ? (current->p1_y_ + current->p2_y_) / 2 : current->p2_y_;

      auto* new_node = device_allocator->make_new<TreeNode>(
          /*parent=*/ current, new_p1_x, new_p1_y, new_p2_x, new_p2_y);
      assert(new_node->is_inside(other));
      assert(new_node->is_inside(body));

      // Insert other into new node.
      new_node->children_[new_node->child_index(other)] = other;

      // Try to install this node.
      // TODO: Ensure that references work as expected here.
      if (pointerCAS<NodeBase>(&child, other, new_node) == other) {
        other->set_parent(new_node);

        // Now insert body.
        current = new_node;
      } else {
        device_allocator->free(new_node);
      }
    }
  }
}


__DEV__ bool TreeNode::remove_child(int c_idx, TreeNode* node) {
  return pointerCAS<NodeBase>(&children_[c_idx], node, nullptr) == node;
}


__DEV__ void TreeNode::collapse_tree() {
  // Collapse bottom-up.
  // Leaf = Only BodyNode objects as children. Or no children at all.
  // No threadfence needed between kernel launches because of cudaDeviceSync.

  if (is_leaf()) {
    TreeNode* current = this;

    while (current != tree) {
      TreeNode* parent = current->parent_;
      assert(parent != nullptr);

      int num_children = 0;
      NodeBase* single_child = nullptr;

      for (int i = 0; i < 4; ++i) {
        // TODO: There could be cases where we do not see a concurrent delete
        // due to missing threadfence.
        // Dangerous: Multiple threads may be deleting stuff at the same time.
        if (current->children_[i] != nullptr) {
          ++num_children;
          single_child = current->children_[i];
        }
      }

      if (num_children < 2) {
        // Find index of current node in parent.
        int c_idx = -1;
        for (int i = 0; i < 4; ++i) {
          if (parent->children_[i] == current) {
            c_idx = i;
            break;
          }
        }

        if (c_idx != -1) {
          if (num_children == 0) {
            // Node is empty. Remove.
            if (parent->remove_child(c_idx, current)) {
              current = parent;
            }
          } else if (num_children == 1) {
            assert(single_child != nullptr);
            // Node has only one child. Merge with parent.
            if (pointerCAS<NodeBase>(&parent->children_[c_idx], current,
                                     single_child) == current) {
              assert(single_child->parent() == this);
              // TODO: Use pointerCAS here?
              single_child->set_parent(parent);
              device_allocator->free(current);
              current = parent;
            }
          }
        } else {
          // Node not found in parent.
          break;
        }
      } else {
         // Retain node.
        break;
      }
    }
  }
}


__DEV__ bool TreeNode::is_leaf() {
  for (int i = 0; i < 4; ++i) {
    if (children_[i]->cast<TreeNode>() != nullptr) {
      return false;
    }
  }

  return true;
}


__DEV__ bool TreeNode::is_inside(BodyNode* body) {
  float x = body->pos_x();
  float y = body->pos_y();
  return x >= p1_x_ && x < p2_x_ && y >= p1_y_ && y < p2_y_;
}


__DEV__ void TreeNode::initialize_frontier() {
  frontier_ = is_leaf();
}


__DEV__ void TreeNode::update_frontier() {
  frontier_ = next_frontier_;
  next_frontier_ = false;
}


__DEV__ void TreeNode::bfs_step() {
  if (frontier_) {
    frontier_ = false;

    // Update pos_x and pos_y: gravitational center
    float total_mass = 0.0f;
    float sum_pos_x = 0.0f;
    float sum_pos_y = 0.0f;

    for (int i = 0; i < 4; ++i) {
      if (children_[i] != nullptr) {
        total_mass += children_[i]->mass();
        sum_pos_x += children_[i]->mass()*children_[i]->pos_x();
        sum_pos_y += children_[i]->mass()*children_[i]->pos_y();
      }
    }

    mass_ = total_mass;
    pos_x_ = sum_pos_x/total_mass;
    pos_y_ = sum_pos_y/total_mass;

    // Add parent to printier.
    if (parent_ != nullptr) {
      parent_->next_frontier_ = true;
    }
  }
}


void step() {
  allocator_handle->parallel_do<BodyNode, &BodyNode::compute_force>();
  allocator_handle->parallel_do<BodyNode, &BodyNode::update>();
  allocator_handle->parallel_do<BodyNode, &BodyNode::clear_node>();
  allocator_handle->parallel_do<BodyNode, &BodyNode::add_to_tree>();
  allocator_handle->parallel_do<TreeNode, &TreeNode::collapse_tree>();

  // BFS steps to update tree.
  allocator_handle->parallel_do<TreeNode, &TreeNode::initialize_frontier>();
  for (int i = 0; i < 10; ++i) {
    allocator_handle->parallel_do<TreeNode, &TreeNode::bfs_step>();
    allocator_handle->parallel_do<TreeNode, &TreeNode::update_frontier>();
  }
}


__global__ void kernel_init_tree() {
  tree = device_allocator->make_new<TreeNode>(
      /*parent=*/ nullptr,
      /*p1_x=*/ -1.0f,
      /*p1_y=*/ -1.0f,
      /*p2_x=*/ 1.0f,
      /*p2_y=*/ 1.0f);
}


__global__ void kernel_init_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
    device_allocator->make_new<BodyNode>(
        /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
        /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
        /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


void initialize_simulation() {
  kernel_init_tree<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_init_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}


int main(int /*argc*/, char** /*argv*/) {
  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  initialize_simulation();

  for (int i = 0; i < kIterations; ++i) {
    step();
  }
}
