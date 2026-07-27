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

void validate_binary(const char* op, const Variable& a, const Variable& b) {
    if (a.value().shape() != b.value().shape()) {
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

Variable make_result(Tensor value,
                     detail::OpKind kind,
                     std::vector<std::shared_ptr<detail::VariableNode>> parents,
                     std::vector<Tensor> saved = {},
                     float scalar = 0.f) {
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
        node->scalar = scalar;
    }
    return detail::VariableAccess::make(std::move(node));
}

}  // namespace

Variable add(const Variable& a, const Variable& b) {
    validate_binary("add", a, b);
    return make_result(
        detail::tensor_add(a.value(), b.value()),
        detail::OpKind::Add,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)});
}

Variable mul(const Variable& a, const Variable& b) {
    validate_binary("mul", a, b);
    return make_result(
        detail::tensor_mul(a.value(), b.value()),
        detail::OpKind::Mul,
        {detail::VariableAccess::node(a), detail::VariableAccess::node(b)},
        {a.value().clone(), b.value().clone()});
}

Variable scale(const Variable& a, float scalar) {
    return make_result(
        detail::tensor_scale(a.value(), scalar),
        detail::OpKind::Scale,
        {detail::VariableAccess::node(a)},
        {},
        scalar);
}

Variable sum(const Variable& a) {
    return make_result(
        detail::tensor_sum(a.value()),
        detail::OpKind::Sum,
        {detail::VariableAccess::node(a)});
}

}  // namespace ag
