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

// Explicit CUDA rejection at the public boundary of every math op.
// The new OOP math path is CPU-only: a CUDA Variable falls through
// the public validate_binary path's device-mismatch check only when
// the operands disagree, which means a CUDA Tensor on both sides
// would slip into the CPU kernels via detail::tensor_* below.
// Rejecting here keeps the error message site-specific and stops a
// CUDA Variable from silently copying through copy_to_host in the
// dispatch path.
void validate_cpu(const char* op, const Variable& a, const Variable& b) {
    if (a.device().is_cuda() || b.device().is_cuda()) {
        std::ostringstream os;
        os << op << ": CUDA tensors are not supported in this build "
              "(got " << a.device().to_string() << " and "
           << b.device().to_string() << ")";
        throw std::runtime_error(os.str());
    }
}

void validate_cpu(const char* op, const Variable& a) {
    if (a.device().is_cuda()) {
        std::ostringstream os;
        os << op << ": CUDA tensors are not supported in this build "
              "(got " << a.device().to_string() << ")";
        throw std::runtime_error(os.str());
    }
}

void validate_cpu(const char* op, const std::vector<Variable>& inputs) {
    for (const auto& v : inputs) {
        if (v.device().is_cuda()) {
            std::ostringstream os;
            os << op << ": CUDA tensors are not supported in this build "
                  "(got " << v.device().to_string() << ")";
            throw std::runtime_error(os.str());
        }
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
    std::vector<int> axes;
    bool keep_dims = false;
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
        node->axes     = std::move(extras.axes);
        node->keep_dims = extras.keep_dims;
    }
    return detail::VariableAccess::make(std::move(node));
}

}  // namespace

// ── Core arithmetic ────────────────────────────────────────────────────

Variable add(const Variable& a, const Variable& b) {
    validate_binary("add", a, b);
    validate_cpu("add", a, b);
    return make_result(
        detail::tensor_add(a.value(), b.value()),
        OpKind::Add,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable mul(const Variable& a, const Variable& b) {
    validate_binary("mul", a, b);
    validate_cpu("mul", a, b);
    return make_result(
        detail::tensor_mul(a.value(), b.value()),
        OpKind::Mul,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable scale(const Variable& a, float scalar) {
    validate_cpu("scale", a);
    return make_result(
        detail::tensor_scale(a.value(), scalar),
        OpKind::Scale,
        {detail::VariableAccess::node(a)},
        {},
        Extras{scalar, 0.f, 0.f, 0, 0, 0, {}, false});
}

Variable sum(const Variable& a) {
    validate_cpu("sum", a);
    return make_result(
        detail::tensor_sum(a.value()),
        OpKind::Sum,
        {detail::VariableAccess::node(a)});
}

Variable sum(const Variable& a, const std::vector<int>& axes, bool keep_dims) {
    validate_unary("sum", a);
    validate_cpu("sum", a);
    Tensor out = detail::tensor_sum_axes_nd(a.value(), axes, keep_dims);
    return make_result(
        std::move(out),
        OpKind::SumAxes,
        {detail::VariableAccess::node(a)},
        {},
        Extras{0.f, 0.f, 0.f, 0, 0, 0,
               detail::normalize_axes(axes,
                                      static_cast<int>(a.value().shape().rank()),
                                      "sum"),
               keep_dims});
}

// ── Linear algebra, reductions, activations, and layout ────────────────

Variable matmul(const Variable& a, const Variable& b) {
    validate_binary("matmul", a, b, /*exact_shape=*/false);
    validate_cpu("matmul", a, b);
    return make_result(
        detail::tensor_matmul_nd(a.value(), b.value()),
        OpKind::MatMul,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable mean(const Variable& a) {
    validate_cpu("mean", a);
    const std::size_t n = a.value().elements();
    if (n == 0) {
        return sum(a);
    }
    return scale(sum(a), 1.f / static_cast<float>(n));
}

Variable mean(const Variable& a, const std::vector<int>& axes, bool keep_dims) {
    validate_unary("mean", a);
    validate_cpu("mean", a);
    const int64_t n = [&] {
        const auto& s = a.value().shape();
        const auto norm =
            detail::normalize_axes(axes, static_cast<int>(s.rank()), "mean");
        int64_t prod = 1;
        for (int axis : norm) {
            prod = detail::mul_check_overflow(prod, s[axis], "mean");
        }
        return prod;
    }();
    if (n == 0) {
        return sum(a, axes, keep_dims);
    }
    return scale(sum(a, axes, keep_dims), 1.f / static_cast<float>(n));
}

Variable broadcast_add(const Variable& a, const Variable& b) {
    validate_binary("broadcast_add", a, b, /*exact_shape=*/false);
    validate_cpu("broadcast_add", a, b);
    return make_result(
        detail::tensor_broadcast_add_nd(a.value(), b.value()),
        OpKind::BroadcastAdd,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable sub(const Variable& a, const Variable& b) {
    validate_binary("sub", a, b);
    validate_cpu("sub", a, b);
    return make_result(
        detail::tensor_sub(a.value(), b.value()),
        OpKind::Sub,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable div_op(const Variable& a, const Variable& b) {
    validate_binary("div_op", a, b);
    validate_cpu("div_op", a, b);
    return make_result(
        detail::tensor_div(a.value(), b.value()),
        OpKind::Div,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable relu(const Variable& a) {
    validate_unary("relu", a);
    validate_cpu("relu", a);
    return make_result(
        detail::tensor_relu(a.value()),
        OpKind::ReLU,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sigmoid(const Variable& a) {
    validate_unary("sigmoid", a);
    validate_cpu("sigmoid", a);
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
    validate_cpu("tanh_op", a);
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
    validate_cpu("exp_op", a);
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
    validate_cpu("log_op", a);
    return make_result(
        detail::tensor_log(a.value()),
        OpKind::Log,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sqrt_op(const Variable& a) {
    validate_unary("sqrt_op", a);
    validate_cpu("sqrt_op", a);
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
    validate_cpu("silu", a);
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
    validate_cpu("softplus", a);
    return make_result(
        detail::tensor_softplus(a.value()),
        OpKind::Softplus,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable sin_op(const Variable& a) {
    validate_unary("sin_op", a);
    validate_cpu("sin_op", a);
    return make_result(
        detail::tensor_sin(a.value()),
        OpKind::Sin,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable cos_op(const Variable& a) {
    validate_unary("cos_op", a);
    validate_cpu("cos_op", a);
    return make_result(
        detail::tensor_cos(a.value()),
        OpKind::Cos,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable clamp(const Variable& a, float lo, float hi) {
    validate_unary("clamp", a);
    validate_cpu("clamp", a);
    if (lo > hi) {
        throw std::invalid_argument("clamp: lo must not exceed hi");
    }
    return make_result(
        detail::tensor_clamp(a.value(), lo, hi),
        OpKind::Clamp,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, lo, hi, 0, 0, 0, {}, false});
}

Variable softmax(const Variable& a, int axis) {
    validate_unary("softmax", a);
    validate_cpu("softmax", a);
    Tensor saved;
    Tensor out = detail::tensor_softmax_nd(a.value(), axis, saved);
    return make_result(
        std::move(out),
        OpKind::Softmax,
        {detail::VariableAccess::node(a)},
        {std::move(saved)},
        Extras{0.f, 0.f, 0.f, axis, 0, 0, {}, false});
}

Variable log_softmax(const Variable& a, int axis) {
    validate_unary("log_softmax", a);
    validate_cpu("log_softmax", a);
    Tensor saved;
    Tensor out = detail::tensor_log_softmax_nd(a.value(), axis, saved);
    return make_result(
        std::move(out),
        OpKind::LogSoftmax,
        {detail::VariableAccess::node(a)},
        {std::move(saved)},
        Extras{0.f, 0.f, 0.f, axis, 0, 0, {}, false});
}

Variable transpose(const Variable& a) {
    validate_unary("transpose", a);
    validate_cpu("transpose", a);
    const int rank = static_cast<int>(a.value().shape().rank());
    if (rank < 2) {
        throw std::invalid_argument("transpose: requires rank >= 2");
    }
    const int ax0 = rank - 2;
    const int ax1 = rank - 1;
    return make_result(
        detail::tensor_transpose_nd(a.value(), ax0, ax1),
        OpKind::Transpose,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, 0.f, 0.f, ax0, 0, ax1, {}, false});
}

Variable transpose(const Variable& a, int axis0, int axis1) {
    validate_unary("transpose", a);
    validate_cpu("transpose", a);
    return make_result(
        detail::tensor_transpose_nd(a.value(), axis0, axis1),
        OpKind::Transpose,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, 0.f, 0.f, axis0, 0, axis1, {}, false});
}

Variable reshape(const Variable& a, const Shape& shape) {
    validate_unary("reshape", a);
    validate_cpu("reshape", a);
    return make_result(
        detail::tensor_reshape_view(a.value(), shape),
        OpKind::Reshape,
        {detail::VariableAccess::node(a)},
        {a.value().clone()});
}

Variable reshape(const Variable& a, int64_t rows, int64_t cols) {
    return reshape(a, Shape{rows, cols});
}

Variable concat(std::vector<Variable> inputs, int axis) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat: requires at least one input");
    }
    validate_cpu("concat", inputs);
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
    const int normalized_axis = detail::normalize_axis(
        axis, static_cast<int>(inputs[0].value().shape().rank()), "concat");
    Tensor out = detail::tensor_concat_nd(values, normalized_axis);
    return make_result(
        std::move(out),
        OpKind::Concat,
        std::move(parents),
        std::move(values),
        Extras{0.f, 0.f, 0.f, normalized_axis, 0, 0, {}, false});
}

Variable hcat(std::vector<Variable> inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("hcat: requires at least one input");
    }
    const int rank = static_cast<int>(inputs[0].value().shape().rank());
    const int axis = rank - 1;
    return concat(std::move(inputs), axis);
}

Variable slice(const Variable& a, int axis, int64_t start, int64_t len) {
    validate_unary("slice", a);
    validate_cpu("slice", a);
    return make_result(
        detail::tensor_slice_nd(a.value(), axis, start, len),
        OpKind::Slice,
        {detail::VariableAccess::node(a)},
        {a.value().clone()},
        Extras{0.f, 0.f, 0.f, axis, start, len, {}, false});
}

Variable col_slice(const Variable& a, int64_t start, int64_t len) {
    return slice(a, 1, start, len);
}

Variable row_slice(const Variable& a, int64_t start, int64_t len) {
    return slice(a, 0, start, len);
}

std::pair<Variable, Variable> split(const Variable& a, int axis) {
    validate_unary("split", a);
    validate_cpu("split", a);
    const int rank = static_cast<int>(a.value().shape().rank());
    const int ax = detail::normalize_axis(axis, rank, "split");
    const int64_t total = a.value().shape()[ax];
    if (total % 2 != 0) {
        std::ostringstream os;
        os << "split: requires an even length along axis " << ax
           << " (got " << total << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t half = total / 2;
    return {slice(a, ax, 0, half), slice(a, ax, half, half)};
}

Variable cumsum(const Variable& a, int axis) {
    validate_unary("cumsum", a);
    validate_cpu("cumsum", a);
    return make_result(
        detail::tensor_cumsum_nd(a.value(), axis),
        OpKind::Cumsum,
        {detail::VariableAccess::node(a)},
        {},
        Extras{0.f, 0.f, 0.f, axis, 0, 0, {}, false});
}

Variable flip(const Variable& a, int axis) {
    validate_unary("flip", a);
    validate_cpu("flip", a);
    return make_result(
        detail::tensor_flip_nd(a.value(), axis),
        OpKind::Flip,
        {detail::VariableAccess::node(a)},
        {},
        Extras{0.f, 0.f, 0.f, axis, 0, 0, {}, false});
}

}  // namespace ag
