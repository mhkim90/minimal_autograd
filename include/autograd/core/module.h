#pragma once
// Phase 5a module stack on the Tensor/Variable API.
//
// Declarations live in the ag::nn namespace to avoid colliding with the
// legacy ag::Module / ag::Linear declarations in autograd/module.h.
// Once the legacy facade is removed in Phase 11, these declarations can
// be exposed under flat ag::Module / ag::Linear aliases. Until then,
// including autograd/core/module.h is the entry point for the new
// replacement stack.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No public graph fields, raw pointers, or writable element references.

#include "autograd/core/variable.h"

#include <string>
#include <utility>
#include <vector>

namespace ag {
namespace nn {

// A (name, parameter) record produced by Module::named_parameters().
// The order returned is the registration order, which must be
// deterministic across calls.
struct NamedParameter {
    std::string name;
    Variable parameter;
};

// Module is the OOP composition boundary. Concrete modules register
// their parameters via register_parameter(...) in their constructor and
// implement forward(...). Sequential composition and child-module
// registration are reserved for Phase 5b.
class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;
    virtual ~Module() = default;

    // Pure virtual forward. Phase 5a ships a single concrete subclass
    // (Linear); additional modules land in later phases.
    virtual Variable forward(const Variable& input) = 0;

    // Convenience callable that delegates to forward().
    Variable operator()(const Variable& input) { return forward(input); }

    // Returns every registered parameter in registration order. The
    // returned Variables share storage with the module's internal
    // copies (they are aliases through shared_ptr<VariableNode>), so
    // in-place mutation by an optimizer step is visible through
    // subsequent calls.
    std::vector<Variable> parameters() const;

    // Returns every registered parameter paired with its name, in
    // registration order.
    std::vector<NamedParameter> named_parameters() const;

    // Clears the gradient on every registered parameter. The leaves
    // themselves and their Tensor values are untouched.
    void zero_grad();

protected:
    // Registers a parameter under the given name. Throws
    // std::invalid_argument if `name` is empty, if a parameter with the
    // same name has already been registered in this module, or if
    // `parameter` does not require gradients.
    void register_parameter(std::string name, Variable parameter);

private:
    std::vector<NamedParameter> parameters_;
};

// Linear(in, out) computes y = matmul(x, W) + b, where W has shape
// (in, out) and b has shape (1, out). Parameters are owned privately
// and exposed through const accessors and the Module registration
// machinery. Initialization is a deterministic Xavier-like uniform
// distribution on CPU; no RNG state is shared between modules.
class Linear : public Module {
public:
    Linear(int in_features, int out_features);

    Variable forward(const Variable& input) override;

    const Variable& weight() const noexcept { return weight_; }
    const Variable& bias() const noexcept { return bias_; }

private:
    Variable weight_;
    Variable bias_;
};

}  // namespace nn
}  // namespace ag
