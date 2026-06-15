#pragma once
// norm.h — GroupNorm.
//
// GroupNorm normalizes over (C/G, H, W) for G groups per sample.
// Input layout: (N, C*H*W), shape parameters passed explicitly.
//
// Backward is not implemented — inference-only. Training use requires
// implementing the full Jacobian before removing the assert in backward().

#include "autograd/module.h"
#include <cmath>
#include <cassert>

namespace ag {

struct GroupNorm : Module {
    int num_groups;
    int channels;
    float eps;
    VarPtr gamma;  // (1, channels) — scale per channel
    VarPtr beta;   // (1, channels) — shift per channel

    GroupNorm(int num_groups, int channels, float eps = 1e-5f);

    // x:  (N, C*H*W)
    // C:  number of channels (must equal this->channels)
    // HW: H*W spatial size
    VarPtr forward(VarPtr x, int C, int HW);

    VarPtr forward(VarPtr x) override {
        assert(false && "GroupNorm requires forward(x, C, HW)");
        return nullptr;
    }

    std::vector<VarPtr> parameters() override { return {gamma, beta}; }
};

} // namespace ag
