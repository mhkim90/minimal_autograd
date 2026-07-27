// Tensor-based core autograd operations.
//
// Each free function validates operand shapes/devices, dispatches the
// forward computation through detail::tensor_*, and constructs a private
// graph node. When no input requires gradients, no graph node is
// installed; the call becomes a pure forward (with the public Variable
// value but no recorded backward path).

#include "autograd/core/ops.h"

#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ag {
namespace {

void validate_binary(const char* op,
                     const Variable& a,
                     const Variable& b,
                     bool exact_shape = true) {
    if (exact_shape && a.value().shape() != b.value().shape()) {
        std::ostringstream os;
        os << op << ": shape mismatch (" << a.value().shape() << " vs "
           << b.value().shape() << ")";
        throw std::invalid_argument(os.str());
    }
    if (a.device() != b.device()) {
        std::ostringstream os;
        os << op << ": device mismatch (" << a.device() << " vs "
           << b.device() << ")";
        throw std::invalid_argument(os.str());
    }
}

void validate_unary(const char* op, const Variable& a) {
    (void)op;
    (void)a;
}

void require_rank2(const char* op, const Variable& v) {
    if (v.value().shape().rank() != 2) {
        std::ostringstream os;
        os << op << ": expected rank-2 tensor, got shape " << v.value().shape();
        throw std::invalid_argument(os.str());
    }
}

using detail::OpKind;

struct Extras {
    float scalar  = 0.f;
    float extra_f0 = 0.f;
    float extra_f1 = 0.f;
    int   axis    = 0;
    int64_t extra_i0 = 0;
    int64_t extra_i1 = 0;
};

Variable make_result(Tensor value,
                     OpKind kind,
                     std::vector<std::shared_ptr<detail::VariableNode>> parents,
                     std::vector<Tensor> saved = {},
                     Extras extras = {}) {
    bool requires_grad = false;
    for (const auto& parent : parents) {
        requires_grad = requires_grad || parent->requires_grad;
    }

    auto node =
        std::make_shared<detail::VariableNode>(std::move(value), requires_grad);
    if (requires_grad) {
        node->kind = kind;
        node->parents = std::move(parents);
        node->saved = std::move(saved);
        node->scalar   = extras.scalar;
        node->extra_f0 = extras.extra_f0;
        node->extra_f1 = extras.extra_f1;
        node->axis     = extras.axis;
        node->extra_i0 = extras.extra_i0;
        node->extra_i1 = extras.extra_i1;
    }
    return detail::VariableAccess::make(std::move(node));
}

}  // namespace

// ── Core arithmetic ────────────────────────────────────────────────────

Variable add(const Variable& a, const Variable& b) {
    validate_binary("add", a, b);
    return make_result(
        detail::tensor_add(a.value(), b.value()),
        OpKind::Add,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable mul(const Variable& a, const Variable& b) {
    validate_binary("mul", a, b);
    return make_result(
        detail::tensor_mul(a.value(), b.value()),
        OpKind::Mul,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable scale(const Variable& a, float scalar) {
    return make_result(
        detail::tensor_scale(a.value(), scalar),
        OpKind::Scale,
        {detail::VariableAccess::node(a)},
        {},
        Extras{scalar, 0.f, 0.f, 0, 0, 0});
}

Variable sum(const Variable& a) {
    return make_result(
        detail::tensor_sum(a.value()),
        OpKind::Sum,
        {detail::VariableAccess::node(a)});
}

// ── Linear algebra, reductions, activations, and layout ────────────────

Variable matmul(const Variable& a, const Variable& b) {
    validate_binary("matmul", a, b, /*exact_shape=*/false);
    require_rank2("matmul", a);
    require_rank2("matmul", b);
    return make_result(
        detail::tensor_matmul(a.value(), b.value()),
        OpKind::MatMul,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable mean(const Variable& a) {
    const std::size_t n = a.value().elements();
    if (n == 0) {
        return sum(a);
    }
    return scale(sum(a), 1.f / static_cast<float>(n));
}

Variable broadcast_add(const Variable& a, const Variable& b) {
    validate_binary("broadcast_add", a, b, /*exact_shape=*/false);
    require_rank2("broadcast_add", a);
    require_rank2("broadcast_add", b);
    return make_result(
        detail::tensor_broadcast_add(a.value(), b.value()),
        OpKind::BroadcastAdd,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable sub(const Variable& a, const Variable& b) {
    validate_binary("sub", a, b);
    return make_result(
        detail::tensor_sub(a.value(), b.value()),
        OpKind::Sub,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable div_op(const Variable& a, const Variable& b) {
    validate_binary("div_op", a, b);
    return make_result(
        detail::tensor_div(a.value(), b.value()),
        OpKind::Div,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable relu(const Variable& a) {
    validate_unary("relu", a);
    return make_result(
        detail::tensor_relu(a.value()),
        OpKind::ReLU,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sigmoid(const Variable& a) {
    validate_unary("sigmoid", a);
    Tensor out = detail::tensor_sigmoid(a.value());
    Tensor saved = out.clone();
    return make_result(
        std::move(out),
        OpKind::Sigmoid,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable tanh_op(const Variable& a) {
    validate_unary("tanh_op", a);
    Tensor out = detail::tensor_tanh(a.value());
    Tensor saved = out.clone();
    return make_result(
        std::move(out),
        OpKind::Tanh,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable exp_op(const Variable& a) {
    validate_unary("exp_op", a);
    Tensor out = detail::tensor_exp(a.value());
    Tensor saved = out.clone();
    return make_result(
        std::move(out),
        OpKind::Exp,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable log_op(const Variable& a) {
    validate_unary("log_op", a);
    return make_result(
        detail::tensor_log(a.value()),
        OpKind::Log,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sqrt_op(const Variable& a) {
    validate_unary("sqrt_op", a);
    Tensor out = detail::tensor_sqrt(a.value());
    Tensor saved = out.clone();
    return make_result(
        std::move(out),
        OpKind::Sqrt,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable silu(const Variable& a) {
    validate_unary("silu", a);
    Tensor sig;
    Tensor out = detail::tensor_silu_forward(a.value(), sig);
    return make_result(
        std::move(out),
        OpKind::SiLU,
        {detail::VariableAccess::node(a)},
        {a.value().clone(), std::move(sig)});
}

Variable softplus(const Variable& a) {
    validate_unary("softplus", a);
    return make_result(
        detail::tensor_softplus(a.value()),
        OpKind::Softplus,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sin_op(const Variable& a) {
    validate_unary("sin_op", a);
    return make_result(
        detail::tensor_sin(a.value()),
        OpKind::Sin,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable cos_op(const Variable& a) {
    validate_unary("cos_op", a);
    return make_result(
        detail::tensor_cos(a.value()),
        OpKind::Cos,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable clamp(const Variable& a, float lo, float hi) {
    validate_unary("clamp", a);
    if (lo > hi) {
        throw std::invalid_argument("clamp: lo must not exceed hi");
    }
    return make_result(
        detail::tensor_clamp(a.value(), lo, hi),
        OpKind::Clamp,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, lo, hi, 0, 0, 0});
}

Variable softmax(const Variable& a) {
    validate_unary("softmax", a);
    require_rank2("softmax", a);
    Tensor saved;
    Tensor out = detail::tensor_softmax(a.value(), saved);
    return make_result(
        std::move(out),
        OpKind::Softmax,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable log_softmax(const Variable& a) {
    validate_unary("log_softmax", a);
    require_rank2("log_softmax", a);
    Tensor saved;
    Tensor out = detail::tensor_log_softmax(a.value(), saved);
    return make_result(
        std::move(out),
        OpKind::LogSoftmax,
        {detail::VariableAccess::node(a)},
        {std::move(saved)});
}

Variable transpose(const Variable& a) {
    validate_unary("transpose", a);
    require_rank2("transpose", a);
    return make_result(
        detail::tensor_transpose(a.value()),
        OpKind::Transpose,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable reshape(const Variable& a, int64_t rows, int64_t cols) {
    validate_unary("reshape", a);
    require_rank2("reshape", a);
    return make_result(
        detail::tensor_reshape_view(a.value(), rows, cols),
        OpKind::Reshape,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable concat(std::vector<Variable> inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat: requires at least one input");
    }
    std::vector<std::shared_ptr<detail::VariableNode>> parents;
    parents.reserve(inputs.size());
    std::vector<Tensor> values;
    values.reserve(inputs.size());
    for (auto& v : inputs) {
        if (v.device() != inputs[0].device()) {
            throw std::invalid_argument("concat: device mismatch");
        }
        parents.push_back(detail::VariableAccess::node(v));
        values.push_back(v.value());
    }
    Tensor out = detail::tensor_concat(values);
    return make_result(
        std::move(out),
        OpKind::Concat,
        std::move(parents),
        std::move(values));
}

Variable hcat(std::vector<Variable> inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("hcat: requires at least one input");
    }
    std::vector<std::shared_ptr<detail::VariableNode>> parents;
    parents.reserve(inputs.size());
    std::vector<Tensor> values;
    values.reserve(inputs.size());
    for (auto& v : inputs) {
        if (v.device() != inputs[0].device()) {
            throw std::invalid_argument("hcat: device mismatch");
        }
        parents.push_back(detail::VariableAccess::node(v));
        values.push_back(v.value());
    }
    Tensor out = detail::tensor_hcat(values);
    return make_result(
        std::move(out),
        OpKind::HCat,
        std::move(parents),
        std::move(values));
}

Variable col_slice(const Variable& a, int64_t start, int64_t len) {
    validate_unary("col_slice", a);
    return make_result(
        detail::tensor_col_slice(a.value(), start, len),
        OpKind::ColSlice,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, 0.f, 0.f, 0, start, len});
}

Variable row_slice(const Variable& a, int64_t start, int64_t len) {
    validate_unary("row_slice", a);
    return make_result(
        detail::tensor_row_slice(a.value(), start, len),
        OpKind::RowSlice,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, 0.f, 0.f, 0, start, len});
}

std::pair<Variable, Variable> split(const Variable& a) {
    validate_unary("split", a);
    require_rank2("split", a);
    const int64_t total_cols = a.value().shape()[1];
    if (total_cols % 2 != 0) {
        throw std::invalid_argument(
            "split: requires an even number of columns");
    }
    const int64_t half = total_cols / 2;
    return {col_slice(a, 0, half), col_slice(a, half, half)};
}

Variable cumsum(const Variable& a, int axis) {
    validate_unary("cumsum", a);
    require_rank2("cumsum", a);
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("cumsum: axis must be 0 or 1");
    }
    return make_result(
        detail::tensor_cumsum(a.value(), axis),
        OpKind::Cumsum,
        {detail::VariableAccess::node(a)},
        {},
        Extras{0.f, 0.f, 0.f, axis, 0, 0});
}

Variable flip(const Variable& a, int axis) {
    validate_unary("flip", a);
    require_rank2("flip", a);
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("flip: axis must be 0 or 1");
    }
    return make_result(
        detail::tensor_flip(a.value(), axis),
        OpKind::Flip,
        {detail::VariableAccess::node(a)},
        {},
        Extras{0.f, 0.f, 0.f, axis, 0, 0});
}

}  // namespace ag
