// src/core/module.cpp — Phase 5a module implementation on the
// Tensor/Variable API.
//
// Linear owns private weight/bias Variables, registers them in the
// documented order, and computes y = matmul(x, W) + b. Initialization
// is a minimal deterministic Xavier-like CPU routine using
// std::mt19937. No Eigen is used here.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ag {
namespace nn {

namespace {

void validate_features(const char* what, int in_features, int out_features) {
    if (in_features <= 0) {
        std::ostringstream os;
        os << what << ": in_features must be positive (got " << in_features
           << ")";
        throw std::invalid_argument(os.str());
    }
    if (out_features <= 0) {
        std::ostringstream os;
        os << what << ": out_features must be positive (got " << out_features
           << ")";
        throw std::invalid_argument(os.str());
    }
}

// Deterministic Xavier-like uniform initialization. Uses a fixed-seed
// std::mt19937 so two Linear modules with the same shape produce
// identical initial parameters. No Eigen.
std::vector<float> xavier_uniform(int in_features, int out_features) {
    // Xavier uniform-style scale: sqrt(6 / (in + out)) gives a stable
    // magnitude for both forward and backward passes.
    const float scale =
        std::sqrt(6.f / static_cast<float>(in_features + out_features));
    std::mt19937 rng(0x5a3f'c1d7u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> values(static_cast<std::size_t>(in_features) *
                              static_cast<std::size_t>(out_features));
    for (float& v : values) v = dist(rng) * scale;
    return values;
}

}  // namespace

// ── Module base class ─────────────────────────────────────────────────

std::vector<Variable> Module::parameters() const {
    std::vector<Variable> out;
    out.reserve(parameters_.size());
    for (const auto& p : parameters_) out.push_back(p.parameter);
    return out;
}

std::vector<NamedParameter> Module::named_parameters() const {
    return parameters_;
}

void Module::zero_grad() {
    for (auto& p : parameters_) {
        p.parameter.zero_grad();
    }
}

void Module::register_parameter(std::string name, Variable parameter) {
    if (name.empty()) {
        throw std::invalid_argument(
            "nn::Module::register_parameter: name must not be empty");
    }
    if (!parameter.requires_grad()) {
        throw std::invalid_argument(
            "nn::Module::register_parameter: parameter must be trainable "
            "(requires_grad == true)");
    }
    for (const auto& existing : parameters_) {
        if (existing.name == name) {
            std::ostringstream os;
            os << "nn::Module::register_parameter: duplicate parameter "
                  "name '" << name << "'";
            throw std::invalid_argument(os.str());
        }
    }
    parameters_.push_back({std::move(name), std::move(parameter)});
}

// ── Linear ────────────────────────────────────────────────────────────

Linear::Linear(int in_features, int out_features) {
    validate_features("nn::Linear", in_features, out_features);

    const auto values = xavier_uniform(in_features, out_features);

    Tensor weight_tensor = Tensor::empty(Shape{static_cast<int64_t>(
                                                in_features),
                                            static_cast<int64_t>(
                                                out_features)});
    weight_tensor.copy_from_host(values.data(), values.size());

    Tensor bias_tensor = Tensor::zeros(Shape{1, static_cast<int64_t>(
                                                     out_features)});

    // requires_grad=true so that backward() accumulates gradients on
    // these leaves.
    Variable weight(std::move(weight_tensor), true);
    Variable bias(std::move(bias_tensor), true);

    // register_parameter takes Variables by value; copies share the
    // underlying VariableNode through shared_ptr, so weight_ and the
    // registered entry alias the same Tensor storage.
    register_parameter("weight", weight);
    register_parameter("bias", bias);
    weight_ = weight;
    bias_ = bias;
}

Variable Linear::forward(const Variable& input) {
    return broadcast_add(matmul(input, weight_), bias_);
}

}  // namespace nn
}  // namespace ag
