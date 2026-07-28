// src/core/norm.cpp — NCHW rank-4 GroupNorm free function on the
// Tensor/Variable API.
//
// The forward + backward kernels live in src/detail/tensor_kernels.h.
// This file wires them into the public ag::group_norm free function:
// each call validates shapes / geometry, runs the forward kernel,
// records the saved tensors needed for backward (xhat pre-affine
// normalized values and per-(sample, group) inv_std), and registers a
// private graph node whose extras metadata encodes num_groups and
// eps.
//
// Shape contract (rank-4 NCHW; first-axis-contiguous storage):
//   input : (N, C, H, W)
//   gamma : (C,)  rank-1
//   beta  : (C,)  rank-1
//   output: (N, C, H, W)
//
// num_groups must be a positive integer and must divide C evenly.
// eps must be finite and strictly positive.

#include "autograd/core/ops.h"
#include "autograd/core/variable.h"
#include "autograd/tensor.h"
#include "autograd/shape.h"

#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ag {
namespace {

void validate_eps(const char* op, float eps) {
    // std::isfinite is false for NaN and +-inf, so a single check
    // covers both. eps must also be strictly positive.
    if (!std::isfinite(eps) || eps <= 0.f) {
        std::ostringstream os;
        os << op << ": eps must be finite and positive (got " << eps << ")";
        throw std::invalid_argument(os.str());
    }
}

void validate_affine_channel(const Tensor& t, int C, const char* what) {
    const Shape& s = t.shape();
    if (s.rank() != 1 || static_cast<int>(s[0]) != C) {
        std::ostringstream os;
        os << what << ": must be rank-1 with length C=" << C
           << "; got " << s;
        throw std::invalid_argument(os.str());
    }
    if (t.device() != Device::cpu()) {
        std::ostringstream os;
        os << what << ": device must be CPU";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

Variable group_norm(const Variable& input,
                    const Variable& gamma,
                    const Variable& beta,
                    int num_groups,
                    float eps) {
    if (input.device().is_cuda() || gamma.device().is_cuda() ||
        beta.device().is_cuda()) {
        throw std::runtime_error(
            "ag::group_norm: CUDA tensors are not supported in this build");
    }
    validate_eps("ag::group_norm", eps);

    const Shape& in_s = input.value().shape();
    if (in_s.rank() != 4) {
        std::ostringstream os;
        os << "ag::group_norm: input must be rank-4 (N, C, H, W); got "
           << in_s;
        throw std::invalid_argument(os.str());
    }
    if (num_groups <= 0) {
        std::ostringstream os;
        os << "ag::group_norm: num_groups must be positive (got "
           << num_groups << ")";
        throw std::invalid_argument(os.str());
    }
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    if (C % num_groups != 0) {
        std::ostringstream os;
        os << "ag::group_norm: channel count C=" << C
           << " not divisible by num_groups=" << num_groups;
        throw std::invalid_argument(os.str());
    }
    validate_affine_channel(gamma.value(), C, "ag::group_norm: gamma");
    validate_affine_channel(beta.value(), C, "ag::group_norm: beta");

    Tensor saved_xhat;
    Tensor saved_inv_std;
    Tensor out = detail::tensor_group_norm_nchw_forward(
        input.value(), gamma.value(), beta.value(),
        num_groups, eps, saved_xhat, saved_inv_std);

    const bool any_requires_grad =
        input.requires_grad() || gamma.requires_grad() ||
        beta.requires_grad();
    auto node = std::make_shared<detail::VariableNode>(
        std::move(out), any_requires_grad);
    if (any_requires_grad) {
        node->kind = detail::OpKind::GroupNorm;
        node->parents = {
            detail::VariableAccess::node(input),
            detail::VariableAccess::node(gamma),
            detail::VariableAccess::node(beta),
        };
        // Snapshot gamma so backward-input uses the forward-time
        // affine weights even if the live gamma is mutated (by an
        // optimizer step or in-place edit) between forward and
        // backward. d_gamma and d_beta are unaffected because they
        // accumulate against the upstream gradient, not gamma.
        node->saved = {
            std::move(saved_xhat),
            std::move(saved_inv_std),
            gamma.value().clone(),
        };
        node->extra_i2 = num_groups;
        (void)N; (void)H; (void)W;
    }
    return detail::VariableAccess::make(std::move(node));
}

}  // namespace ag
