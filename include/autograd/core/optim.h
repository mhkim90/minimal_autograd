#pragma once
// Optimizer stack on the Tensor/Variable API.
//
// Declarations live in the ag::optim namespace to avoid colliding with
// the legacy ag::SGD / ag::Adam declarations in autograd/optim.h. The
// replacement stack will be exposed under flat ag::SGD / ag::Adam
// aliases after the legacy facade is removed in Phase 11.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No public graph fields, raw pointers, or writable element references.

#include "autograd/core/variable.h"
#include "autograd/tensor.h"

#include <cstdint>
#include <vector>

namespace ag {
namespace optim {

// Vanilla stochastic gradient descent: p -= lr * grad for every
// eligible parameter.
//
// Constraints:
//   * CPU-only. Non-CPU parameters cause std::runtime_error to be
//     thrown from step().
//   * Parameters without requires_grad, or with no current gradient
//     (has_grad == false), are skipped without error.
//   * The learning rate must be finite and non-negative. The
//     constructor rejects non-finite and negative values.
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

// AdamState is a portable snapshot of an Adam optimizer's internal
// state: the step counter, the hyperparameters, and one pair of moment
// tensors per registered parameter. Moments are deep-copied on the
// way in (state()) and on the way out (load_state()), so an
// AdamState value never aliases the live optimizer.
struct AdamState {
    int64_t t = 0;
    float lr = 0.f;
    float beta1 = 0.f;
    float beta2 = 0.f;
    float eps = 0.f;
    std::vector<Tensor> first_moments;
    std::vector<Tensor> second_moments;
};

// Adam: per-parameter adaptive moment estimation. Eligible parameters
// (requires_grad && has_grad) receive
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
//   p -= lr * m_hat / (sqrt(v_hat) + eps).
//
// Constraints:
//   * CPU-only. The constructor rejects every CUDA parameter up front.
//   * Constructor validates: lr is finite and non-negative; beta1 and
//     beta2 are finite and lie in [0, 1); eps is finite and positive.
//   * Parameters without requires_grad / has_grad are skipped by step();
//     their moments are unchanged. The optimizer-wide step count advances
//     once only when at least one parameter is eligible.
//   * Empty parameter lists are accepted (no-op step/zero_grad).
//   * step() prevalidates every eligible parameter before any state or
//     parameter mutation, and computes all new values into temporaries
//     so a failure cannot leave the optimizer or parameters partially
//     updated.
//   * Parameter Tensor storage is preserved during commit through the
//     internal optimizer mutation boundary.
//   * state() returns independent deep-cloned moments; load_state()
//     validates the entire snapshot before mutating any live state,
//     and a failed load leaves both optimizer state and parameter
//     values unchanged.
class Adam {
public:
    Adam(std::vector<Variable> params,
         float lr = 1e-3f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f);

    void step();
    void zero_grad();

    AdamState state() const;
    void load_state(const AdamState& s);

    float learning_rate() const noexcept { return lr_; }
    float beta1() const noexcept { return beta1_; }
    float beta2() const noexcept { return beta2_; }
    float eps() const noexcept { return eps_; }
    int64_t step_count() const noexcept { return t_; }
    const std::vector<Variable>& parameters() const noexcept {
        return params_;
    }

private:
    std::vector<Variable> params_;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    int64_t t_ = 0;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
};

}  // namespace optim
}  // namespace ag
