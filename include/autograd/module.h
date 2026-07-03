#pragma once
// module.h — Module, Linear, Sequential, ReLUModule.
//
// A Module owns learnable parameters (VarPtr leaves) and exposes a forward()
// that takes a VarPtr and returns a VarPtr. Sequential composes them.

#include "autograd/ops.h"
#include <vector>
#include <memory>
#include <cmath>
#include <utility>
#include <cassert>

namespace ag {

struct Module {
    virtual VarPtr forward(VarPtr x) = 0;
    virtual ~Module() = default;

    // Leaf parameters — what the optimizer updates.
    virtual std::vector<VarPtr> parameters() = 0;

    void zero_grad() {
        for (auto& p : parameters()) {
            p->grad.setZero();
            p->cuda_zero_grad();
        }
    }
};

// Linear(in, out): y = x @ W + b
//
// W: (in, out) — Xavier init: scale = sqrt(2/in)
// b: (1, out)  — zero
struct Linear : Module {
    VarPtr W;
    VarPtr b;

    Linear(int in_features, int out_features);

    VarPtr forward(VarPtr x) override {
        return broadcast_add(matmul(x, W), b);
    }

    std::vector<VarPtr> parameters() override { return {W, b}; }
};

struct Sequential : Module {
    std::vector<std::shared_ptr<Module>> layers;

    void add(std::shared_ptr<Module> m) { layers.push_back(std::move(m)); }

    VarPtr forward(VarPtr x) override {
        for (auto& l : layers) x = l->forward(x);
        return x;
    }

    std::vector<VarPtr> parameters() override {
        std::vector<VarPtr> params;
        for (auto& l : layers)
            for (auto& p : l->parameters())
                params.push_back(p);
        return params;
    }
};

// Module wrapper around the relu() free function — for use inside Sequential.
struct ReLUModule : Module {
    VarPtr forward(VarPtr x) override { return relu(x); }
    std::vector<VarPtr> parameters() override { return {}; }
};

struct SiLUModule : Module {
    VarPtr forward(VarPtr x) override { return silu(x); }
    std::vector<VarPtr> parameters() override { return {}; }
};

struct SigmoidModule : Module {
    VarPtr forward(VarPtr x) override { return sigmoid(x); }
    std::vector<VarPtr> parameters() override { return {}; }
};

} // namespace ag
