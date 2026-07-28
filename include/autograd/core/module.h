#pragma once
// Module stack on the Tensor/Variable API.
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

#include <memory>
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
// implement forward(...). Modules can also register child modules via
// register_module(...); parameters() and named_parameters() recurse
// depth-first into children, and zero_grad() recurses too.
class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;
    virtual ~Module() = default;

    virtual Variable forward(const Variable& input) = 0;

    Variable operator()(const Variable& input) { return forward(input); }

    // Returns every leaf parameter reachable through this module's direct
    // parameters first, then child modules depth-first; each registry keeps
    // insertion order. The returned Variables share storage with the module's
    // internal copies (they are aliases through
    // shared_ptr<VariableNode>), so in-place mutation by an optimizer
    // step is visible through subsequent calls.
    std::vector<Variable> parameters() const;

    // Same traversal as parameters(), but each leaf is paired with its
    // fully-qualified name. Direct-parameter names are returned
    // verbatim; child-module names are joined with '.' into the
    // leaves underneath.
    std::vector<NamedParameter> named_parameters() const;

    // Clears the gradient on every reachable leaf parameter (recurses
    // into child modules).
    void zero_grad();

protected:
    // A registered child module: its name (which the parent uses when
    // building dotted names) and a non-null shared_ptr.
    struct NamedChild {
        std::string name;
        std::shared_ptr<Module> module;
    };

    // Registers a parameter under the given name. Throws
    // std::invalid_argument if `name` is empty, if `name` is already
    // used by a parameter or a child module, or if `parameter` does
    // not require gradients.
    void register_parameter(std::string name, Variable parameter);

    // Registers a child module under the given name. Throws
    // std::invalid_argument if `name` is empty, if `module` is null,
    // if registration would create a cycle, or if `name` is already used
    // by a parameter or a child module.
    void register_module(std::string name, std::shared_ptr<Module> module);

    // Internal collection helper invoked by parameters(),
    // named_parameters(), and zero_grad(). Recurses depth-first.
    void collect_named(std::vector<NamedParameter>& out,
                       const std::string& prefix) const;

    std::vector<NamedParameter> parameters_;
    std::vector<NamedChild> children_;

private:
    bool contains(const Module* target) const;
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

// ReLU is a parameter-free module that forwards through the public
// ag::relu free function.
class ReLU : public Module {
public:
    Variable forward(const Variable& input) override;
};

// Sequential composes child modules in registration order and passes
// the input through each forward() in turn. Children receive numeric
// names ("0", "1", ...) so the resulting named_parameters() tree is
// deterministic and stable across repeated calls. Sequential does not
// expose a public mutable layer container.
class Sequential : public Module {
public:
    Sequential() = default;

    // Adds a child module. The child is assigned the next numeric
    // name based on the current child count.
    void add(std::shared_ptr<Module> module);

    Variable forward(const Variable& input) override;
};

// Conv2d(in_C, out_C, kH, kW, stride, pad) is the NCHW rank-4
// convolution module on the replacement Tensor/Variable API. It owns
// privately-registered "weight" of shape (out_C, in_C, kH, kW) and
// "bias" of shape (out_C,). Initialization is a deterministic
// Kaiming-like uniform distribution with bound sqrt(1 / fan_in).
// forward(...) routes through the public ag::conv2d free function.
class Conv2d : public Module {
public:
    Conv2d(int in_channels,
           int out_channels,
           int kH, int kW,
           int stride = 1,
           int pad = 0);

    Variable forward(const Variable& input) override;

    const Variable& weight() const noexcept { return weight_; }
    const Variable& bias() const noexcept { return bias_; }

    int in_channels() const noexcept { return in_channels_; }
    int out_channels() const noexcept { return out_channels_; }
    int kernel_h() const noexcept { return kH_; }
    int kernel_w() const noexcept { return kW_; }
    int stride() const noexcept { return stride_; }
    int pad() const noexcept { return pad_; }

private:
    int in_channels_;
    int out_channels_;
    int kH_;
    int kW_;
    int stride_;
    int pad_;

    Variable weight_;
    Variable bias_;
};

// MaxPool2d(kH, kW, stride) is the NCHW rank-4 max-pooling module.
// stride defaults to kH (i.e. kW when equal) when the constructor's
// stride argument is negative, matching the legacy default. stride
// must be positive otherwise. The module is parameter-free and routes
// forward through the public ag::max_pool2d free function.
class MaxPool2d : public Module {
public:
    MaxPool2d(int kH, int kW, int stride = -1);

    Variable forward(const Variable& input) override;

    int kernel_h() const noexcept { return kH_; }
    int kernel_w() const noexcept { return kW_; }
    int stride() const noexcept { return stride_; }

private:
    int kH_;
    int kW_;
    int stride_;
};

}  // namespace nn
}  // namespace ag
