// src/core/module.cpp — module implementation on the Tensor/Variable API.
//
// Linear owns private weight/bias Variables, registers them in the
// documented order, and computes y = matmul(x, W) + b. ReLU wraps the
// public ag::relu free function. Sequential composes child modules in
// registration order and assigns numeric child names so the
// named_parameters() traversal is deterministic. Initialization is a
// minimal deterministic Xavier-like CPU routine using std::mt19937. No
// Eigen is used here.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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
    const float scale =
        std::sqrt(6.f / static_cast<float>(in_features + out_features));
    std::mt19937 rng(0x5a3f'c1d7u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> values(static_cast<std::size_t>(in_features) *
                              static_cast<std::size_t>(out_features));
    for (float& v : values) v = dist(rng) * scale;
    return values;
}

// Deterministic Kaiming-like uniform initialization for Conv2d. The
// fan_in is in_channels * kH * kW. The bound is sqrt(1/fan_in),
// matching the legacy Conv2d initializer. Same fixed seed as Linear so
// two Conv2d modules with identical shapes produce identical
// parameters.
std::vector<float> kaiming_uniform_conv(int in_channels, int out_channels,
                                         int kH, int kW) {
    const float fan_in = static_cast<float>(in_channels) *
                         static_cast<float>(kH) *
                         static_cast<float>(kW);
    const float bound = std::sqrt(1.f / fan_in);
    std::mt19937 rng(0x9e37'79b9u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> values(static_cast<std::size_t>(out_channels) *
                              static_cast<std::size_t>(in_channels) *
                              static_cast<std::size_t>(kH) *
                              static_cast<std::size_t>(kW));
    for (float& v : values) v = dist(rng) * bound;
    return values;
}

}  // namespace

// ── Module base class ─────────────────────────────────────────────────

void Module::collect_named(std::vector<NamedParameter>& out,
                            const std::string& prefix) const {
    for (const auto& p : parameters_) {
        NamedParameter np;
        np.name = prefix.empty() ? p.name : prefix + "." + p.name;
        np.parameter = p.parameter;
        out.push_back(std::move(np));
    }
    for (const auto& c : children_) {
        const std::string child_prefix =
            prefix.empty() ? c.name : prefix + "." + c.name;
        c.module->collect_named(out, child_prefix);
    }
}

std::vector<Variable> Module::parameters() const {
    std::vector<NamedParameter> named;
    collect_named(named, "");
    std::vector<Variable> out;
    out.reserve(named.size());
    for (auto& np : named) out.push_back(np.parameter);
    return out;
}

std::vector<NamedParameter> Module::named_parameters() const {
    std::vector<NamedParameter> out;
    collect_named(out, "");
    return out;
}

void Module::zero_grad() {
    for (auto& p : parameters_) p.parameter.zero_grad();
    for (auto& c : children_) c.module->zero_grad();
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
    for (const auto& existing : children_) {
        if (existing.name == name) {
            std::ostringstream os;
            os << "nn::Module::register_parameter: parameter name '"
               << name << "' collides with a child module";
            throw std::invalid_argument(os.str());
        }
    }
    parameters_.push_back({std::move(name), std::move(parameter)});
}

void Module::register_module(std::string name,
                             std::shared_ptr<Module> module) {
    if (name.empty()) {
        throw std::invalid_argument(
            "nn::Module::register_module: name must not be empty");
    }
    if (!module) {
        throw std::invalid_argument(
            "nn::Module::register_module: child module must not be null");
    }
    if (module->contains(this)) {
        throw std::invalid_argument(
            "nn::Module::register_module: module cycle detected");
    }
    for (const auto& existing : parameters_) {
        if (existing.name == name) {
            std::ostringstream os;
            os << "nn::Module::register_module: child name '" << name
               << "' collides with a parameter";
            throw std::invalid_argument(os.str());
        }
    }
    for (const auto& existing : children_) {
        if (existing.name == name) {
            std::ostringstream os;
            os << "nn::Module::register_module: duplicate child name '"
               << name << "'";
            throw std::invalid_argument(os.str());
        }
    }
    children_.push_back({std::move(name), std::move(module)});
}

bool Module::contains(const Module* target) const {
    if (this == target) return true;
    for (const auto& child : children_) {
        if (child.module->contains(target)) return true;
    }
    return false;
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

    Variable weight(std::move(weight_tensor), true);
    Variable bias(std::move(bias_tensor), true);

    register_parameter("weight", weight);
    register_parameter("bias", bias);
    weight_ = weight;
    bias_ = bias;
}

Variable Linear::forward(const Variable& input) {
    return broadcast_add(matmul(input, weight_), bias_);
}

// ── ReLU ──────────────────────────────────────────────────────────────

Variable ReLU::forward(const Variable& input) {
    return relu(input);
}

// ── Sequential ────────────────────────────────────────────────────────

void Sequential::add(std::shared_ptr<Module> module) {
    const std::string name = std::to_string(children_.size());
    register_module(name, std::move(module));
}

Variable Sequential::forward(const Variable& input) {
    Variable x = input;
    for (auto& c : children_) {
        x = c.module->forward(x);
    }
    return x;
}

// ── Conv2d ────────────────────────────────────────────────────────────

Conv2d::Conv2d(int in_channels,
               int out_channels,
               int kH, int kW,
               int stride,
               int pad)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kH_(kH),
      kW_(kW),
      stride_(stride),
      pad_(pad) {
    if (in_channels_ <= 0) {
        std::ostringstream os;
        os << "nn::Conv2d: in_channels must be positive (got "
           << in_channels_ << ")";
        throw std::invalid_argument(os.str());
    }
    if (out_channels_ <= 0) {
        std::ostringstream os;
        os << "nn::Conv2d: out_channels must be positive (got "
           << out_channels_ << ")";
        throw std::invalid_argument(os.str());
    }
    if (kH_ <= 0 || kW_ <= 0) {
        std::ostringstream os;
        os << "nn::Conv2d: kernel must be positive (got "
           << kH_ << " x " << kW_ << ")";
        throw std::invalid_argument(os.str());
    }
    if (stride_ <= 0) {
        std::ostringstream os;
        os << "nn::Conv2d: stride must be positive (got " << stride_ << ")";
        throw std::invalid_argument(os.str());
    }
    if (pad_ < 0) {
        std::ostringstream os;
        os << "nn::Conv2d: pad must be non-negative (got " << pad_ << ")";
        throw std::invalid_argument(os.str());
    }

    const auto weight_values = kaiming_uniform_conv(
        in_channels_, out_channels_, kH_, kW_);

    Tensor weight_tensor = Tensor::empty(
        Shape{static_cast<int64_t>(out_channels_),
              static_cast<int64_t>(in_channels_),
              static_cast<int64_t>(kH_),
              static_cast<int64_t>(kW_)});
    weight_tensor.copy_from_host(weight_values.data(),
                                 weight_values.size());

    Tensor bias_tensor = Tensor::zeros(Shape{static_cast<int64_t>(
                                                    out_channels_)});

    Variable weight_var(std::move(weight_tensor), true);
    Variable bias_var(std::move(bias_tensor), true);

    register_parameter("weight", weight_var);
    register_parameter("bias", bias_var);
    weight_ = weight_var;
    bias_ = bias_var;
}

Variable Conv2d::forward(const Variable& input) {
    const Shape& s = input.value().shape();
    if (s.rank() != 4) {
        std::ostringstream os;
        os << "nn::Conv2d::forward: input must be rank-4 (N, C, H, W); got "
           << s;
        throw std::invalid_argument(os.str());
    }
    if (static_cast<int>(s[1]) != in_channels_) {
        std::ostringstream os;
        os << "nn::Conv2d::forward: input channel mismatch (input C="
           << s[1] << ", module in_channels=" << in_channels_ << ")";
        throw std::invalid_argument(os.str());
    }
    return conv2d(input, weight_, bias_, stride_, pad_);
}

// ── MaxPool2d ─────────────────────────────────────────────────────────

MaxPool2d::MaxPool2d(int kH, int kW, int stride)
    : kH_(kH), kW_(kW),
      stride_(stride < 0 ? kH : stride) {
    if (kH_ <= 0 || kW_ <= 0) {
        std::ostringstream os;
        os << "nn::MaxPool2d: kernel must be positive (got "
           << kH_ << " x " << kW_ << ")";
        throw std::invalid_argument(os.str());
    }
    if (stride_ <= 0) {
        std::ostringstream os;
        os << "nn::MaxPool2d: stride must be positive (got "
           << stride_ << ")";
        throw std::invalid_argument(os.str());
    }
}

Variable MaxPool2d::forward(const Variable& input) {
    return max_pool2d(input, kH_, kW_, stride_);
}

}  // namespace nn
}  // namespace ag
