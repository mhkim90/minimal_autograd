#pragma once
// Tensor-based autograd value handle.
//
// This transitional header is Eigen-free. It will become the canonical
// autograd/variable.h after the legacy Var API is removed.

#include "autograd/tensor.h"

#include <memory>

namespace ag {

namespace detail {
class VariableNode;
struct VariableAccess;
}  // namespace detail

class Variable {
public:
    Variable();
    explicit Variable(Tensor value, bool requires_grad = false);

    const Tensor& value() const noexcept;
    bool requires_grad() const noexcept;
    bool has_grad() const noexcept;
    const Tensor& grad() const;
    Device device() const noexcept;

    Variable to(Device target) const;
    Variable detach() const;

    void backward();
    void backward(const Tensor& upstream_gradient);

    // Clears this node only. It does not traverse the graph.
    void zero_grad();

private:
    explicit Variable(std::shared_ptr<detail::VariableNode> node);

    std::shared_ptr<detail::VariableNode> node_;

    friend struct detail::VariableAccess;
};

}  // namespace ag
