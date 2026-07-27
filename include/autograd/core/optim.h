#pragma once
// Phase 5a optim stack on the Tensor/Variable API.
//
// Declarations live in the ag::optim namespace to avoid colliding with
// the legacy ag::SGD / ag::Adam declarations in autograd/optim.h. The
// replacement stack will be exposed under flat ag::SGD / ag::Adam
// aliases after the legacy facade is removed in Phase 11.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No public mutable graph/storage handles.

#include "autograd/core/variable.h"

#include <vector>

namespace ag {
namespace optim {

// Vanilla stochastic gradient descent: p -= lr * grad for every
// eligible parameter.
//
// Constraints (Phase 5a):
//   * CPU-only. Non-CPU parameters cause std::runtime_error to be
//     thrown from step().
//   * Parameters without requires_grad, or with no current gradient
//     (has_grad == false), are skipped without error.
//   * The learning rate must be finite and non-negative. The
//     constructor rejects non-finite and negative values; step()
//     does not re-validate (the constructor's guarantee is preserved
//     unless the user mutates lr through back-channels, which is not
//     part of the public surface).
//   * Mutation is performed through the narrow internal
//     detail::VariableAccess::apply_to_storage helper. The Tensor
//     storage is mutated in place; the Variable's node Tensor is not
//     replaced. A Tensor alias taken before step() observes the
//     updated values.
class SGD {
public:
    SGD(std::vector<Variable> params, float lr);

    // Validates every eligible parameter before performing any update.
    // Parameters without requires_grad or with has_grad == false are skipped.
    void step();

    // Clears the gradient on every registered parameter.
    void zero_grad();

    float learning_rate() const noexcept { return lr_; }
    const std::vector<Variable>& parameters() const noexcept {
        return params_;
    }

private:
    std::vector<Variable> params_;
    float lr_;
};

}  // namespace optim
}  // namespace ag
