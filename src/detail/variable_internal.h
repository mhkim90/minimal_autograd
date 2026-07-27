#pragma once

#include "autograd/core/variable.h"
#include "autograd/extension/custom_op.h"

#include <memory>
#include <utility>
#include <vector>

namespace ag {
namespace detail {

enum class OpKind {
    Leaf,
    Add,
    Mul,
    Scale,
    Sum,
    Custom,
};

struct VariableNode {
    Tensor value;
    bool requires_grad = false;
    bool has_grad = false;
    Tensor grad;

    OpKind kind = OpKind::Leaf;
    float scalar = 0.f;
    std::vector<std::shared_ptr<VariableNode>> parents;
    std::vector<Tensor> saved;
    BackwardFunction custom_backward;

    VariableNode(Tensor v, bool requires)
        : value(std::move(v)), requires_grad(requires) {}
};

struct VariableAccess {
    static const std::shared_ptr<VariableNode>& node(const Variable& v) {
        return v.node_;
    }

    static Variable make(std::shared_ptr<VariableNode> node) {
        return Variable(std::move(node));
    }
};

}  // namespace detail
}  // namespace ag
