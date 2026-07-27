#pragma once

#include "autograd/core/variable.h"
#include "autograd/extension/custom_op.h"

#include <cstdint>
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
    SumAxes,
    MatMul,
    ReLU,
    BroadcastAdd,
    Softmax,
    LogSoftmax,
    Transpose,
    Reshape,
    Concat,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    SiLU,
    Softplus,
    Sub,
    Div,
    Cumsum,
    Flip,
    Sin,
    Cos,
    Clamp,
    Slice,
    Custom,
};

struct VariableNode {
    Tensor value;
    bool requires_grad = false;
    bool has_grad = false;
    Tensor grad;

    OpKind kind = OpKind::Leaf;

    // Per-operation scalar metadata.
    float scalar = 0.f;

    // Per-op extras. Field meanings by kind:
    //   scalar          : Scale factor
    //   extra_f0        : Clamp lo
    //   extra_f1        : Clamp hi
    //   axis            : Single-axis axis (Cumsum / Flip / Softmax /
    //                     LogSoftmax / Transpose first axis / Slice /
    //                     Concat / Split).
    //   extra_i0        : Slice start / Transpose second axis
    //   extra_i1        : Slice len
    //   axes            : Multi-axis axes (sum / mean with axes)
    //   keep_dims       : sum / mean with axes keep_dims flag
    float extra_f0 = 0.f;
    float extra_f1 = 0.f;
    int   axis    = 1;
    int64_t extra_i0 = 0;
    int64_t extra_i1 = 0;
    std::vector<int> axes;
    bool keep_dims = false;

    std::vector<std::shared_ptr<VariableNode>> parents;
    std::vector<Tensor> saved;
    BackwardFunction custom_backward;

    VariableNode(Tensor v, bool needs_grad)
        : value(std::move(v)), requires_grad(needs_grad) {}
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
