#pragma once
// variable.h — Var / VarPtr.
//
// Phase 2 redesign: backward() does a topological sort of the graph, then
// runs each node's back_fn in reverse. Shared nodes are visited once, so
// a node used in two branches gets its gradient accumulated exactly once.

#include "autograd/tensor.h"
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>
#include <utility>
#include <cassert>

namespace ag {

struct Var;
using VarPtr = std::shared_ptr<Var>;

struct Var {
    Mat data;
    Mat grad;

    // Inputs to this node (for topo traversal).
    std::vector<VarPtr> parents;

    // Captured by apply<Fn>. Reads this->grad, accumulates into parents.
    std::function<void()> back_fn;

    explicit Var(Mat d)
        : data(std::move(d)),
          grad(Mat::Zero(data.rows(), data.cols())) {}

    static VarPtr make(Mat d) { return std::make_shared<Var>(std::move(d)); }

    // Call on a scalar (1x1) loss node. Builds topo order, then runs
    // back_fn in reverse. Safe to call once per forward pass.
    void backward();

    // Zeroes grad on every node reachable from this node.
    void zero_grad();

private:
    void _build_topo(std::vector<Var*>& order,
                     std::unordered_set<Var*>& visited);
};

} // namespace ag
