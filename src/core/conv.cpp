// src/core/conv.cpp — NCHW rank-4 Conv2d and MaxPool2d free functions on
// the Tensor/Variable API.
//
// The kernels live in src/detail/tensor_kernels.h. This file wires
// them into the public ag::conv2d and ag::max_pool2d free functions:
// each validates shapes / geometry, runs the forward kernel, records
// the saved tensors needed for backward (im2col matrix and weight for
// Conv2d; argmax mask for MaxPool2d), and registers a private graph
// node whose saved metadata encodes the geometry.
//
// Shape contract (rank-4 NCHW; first-axis-contiguous storage):
//   input  : (N, C, H, W)
//   weight : (OC, C, kH, kW)
//   bias   : (OC,)  rank-1
//   output : (N, OC, oH, oW)
//
// stride must be a positive integer, pad must be a non-negative
// integer such that oH = (H + 2*pad - kH)/stride + 1 (and similarly
// for oW) is a positive integer. Bias is rank-1 (OC,) only; any other
// shape (including rank-2 (1, OC)) is rejected so backward returns a
// gradient with the same shape as the bias parent.

#include "autograd/core/ops.h"
#include "autograd/core/variable.h"
#include "autograd/tensor.h"
#include "autograd/shape.h"

#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ag {
namespace {

void validate_conv2d_geometry(const char* op, int H, int W,
                              int kH, int kW, int stride, int pad) {
    if (stride <= 0) {
        std::ostringstream os;
        os << op << ": stride must be positive (got " << stride << ")";
        throw std::invalid_argument(os.str());
    }
    if (pad < 0) {
        std::ostringstream os;
        os << op << ": pad must be non-negative (got " << pad << ")";
        throw std::invalid_argument(os.str());
    }
    if (kH <= 0 || kW <= 0) {
        std::ostringstream os;
        os << op << ": kernel must be positive (got "
           << kH << " x " << kW << ")";
        throw std::invalid_argument(os.str());
    }
    if (H + 2 * pad < kH || W + 2 * pad < kW) {
        std::ostringstream os;
        os << op << ": kernel larger than input plus 2*pad ("
           << H << "+2*" << pad << " vs " << kH << ", "
           << W << "+2*" << pad << " vs " << kW << ")";
        throw std::invalid_argument(os.str());
    }
    const int span_h = H + 2 * pad - kH;
    const int span_w = W + 2 * pad - kW;
    if (span_h % stride != 0 || span_w % stride != 0) {
        std::ostringstream os;
        os << op << ": output extent non-integer (spans "
           << span_h << ", " << span_w << " not divisible by stride "
           << stride << ")";
        throw std::invalid_argument(os.str());
    }
}

void validate_pool_geometry(const char* op, int H, int W,
                            int kH, int kW, int stride) {
    if (kH <= 0 || kW <= 0) {
        std::ostringstream os;
        os << op << ": kernel must be positive (got "
           << kH << " x " << kW << ")";
        throw std::invalid_argument(os.str());
    }
    if (stride <= 0) {
        std::ostringstream os;
        os << op << ": stride must be positive (got " << stride << ")";
        throw std::invalid_argument(os.str());
    }
    if (H < kH || W < kW) {
        std::ostringstream os;
        os << op << ": kernel larger than input ("
           << H << " vs " << kH << ", " << W << " vs " << kW << ")";
        throw std::invalid_argument(os.str());
    }
}

// Rank-1 (OC,) bias is the only accepted bias shape. Anything else is
// rejected so the backward contract returns a gradient with the same
// shape as the bias parent.
void validate_bias(const Tensor& bias, int OC, const char* op) {
    const Shape& s = bias.shape();
    if (s.rank() != 1 || static_cast<int>(s[0]) != OC) {
        std::ostringstream os;
        os << op << ": bias must be rank-1 with length OC="
           << OC << "; got " << s;
        throw std::invalid_argument(os.str());
    }
    if (bias.device() != Device::cpu()) {
        std::ostringstream os;
        os << op << ": bias device must be CPU";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

Variable conv2d(const Variable& input,
               const Variable& weight,
               const Variable& bias,
               int stride,
               int pad) {
    if (input.device().is_cuda() || weight.device().is_cuda() ||
        bias.device().is_cuda()) {
        throw std::runtime_error(
            "ag::conv2d: CUDA tensors are not supported in this build");
    }
    const Shape& in_s = input.value().shape();
    if (in_s.rank() != 4) {
        std::ostringstream os;
        os << "ag::conv2d: input must be rank-4 (N, C, H, W); got "
           << in_s;
        throw std::invalid_argument(os.str());
    }
    const Shape& w_s = weight.value().shape();
    if (w_s.rank() != 4) {
        std::ostringstream os;
        os << "ag::conv2d: weight must be rank-4 (OC, C, kH, kW); got "
           << w_s;
        throw std::invalid_argument(os.str());
    }
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    const int OC = static_cast<int>(w_s[0]);
    const int w_in_C = static_cast<int>(w_s[1]);
    const int kH = static_cast<int>(w_s[2]);
    const int kW = static_cast<int>(w_s[3]);
    if (C != w_in_C) {
        std::ostringstream os;
        os << "ag::conv2d: input channel mismatch (input C=" << C
           << ", weight in_C=" << w_in_C << ")";
        throw std::invalid_argument(os.str());
    }
    validate_conv2d_geometry("ag::conv2d", H, W, kH, kW, stride, pad);

    validate_bias(bias.value(), OC, "ag::conv2d");

    Tensor saved_col;
    Tensor out = detail::tensor_conv2d_nchw_forward(
        input.value(), weight.value(), bias.value(), stride, pad, saved_col);

    const bool any_requires_grad =
        input.requires_grad() || weight.requires_grad() ||
        bias.requires_grad();
    auto node = std::make_shared<detail::VariableNode>(
        std::move(out), any_requires_grad);
    if (any_requires_grad) {
        node->kind = detail::OpKind::Conv2d;
        node->parents = {
            detail::VariableAccess::node(input),
            detail::VariableAccess::node(weight),
            detail::VariableAccess::node(bias),
        };
        node->saved = {std::move(saved_col), weight.value().clone()};
        node->extra_i0 = kH;
        node->extra_i1 = kW;
        node->axis = stride;
        node->extra_f0 = static_cast<float>(pad);
        (void)N;
    }
    return detail::VariableAccess::make(std::move(node));
}

Variable max_pool2d(const Variable& input,
                    int kH, int kW,
                    int stride) {
    if (input.device().is_cuda()) {
        throw std::runtime_error(
            "ag::max_pool2d: CUDA tensors are not supported in this build");
    }
    const Shape& in_s = input.value().shape();
    if (in_s.rank() != 4) {
        std::ostringstream os;
        os << "ag::max_pool2d: input must be rank-4 (N, C, H, W); got "
           << in_s;
        throw std::invalid_argument(os.str());
    }
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    validate_pool_geometry("ag::max_pool2d", H, W, kH, kW, stride);

    Tensor saved_mask;
    Tensor out = detail::tensor_maxpool2d_nchw_forward(
        input.value(), kH, kW, stride, saved_mask);

    if (input.requires_grad()) {
        auto node = std::make_shared<detail::VariableNode>(
            std::move(out), true);
        node->kind = detail::OpKind::MaxPool2d;
        node->parents = {detail::VariableAccess::node(input)};
        node->saved = {std::move(saved_mask)};
        node->extra_i0 = kH;
        node->extra_i1 = kW;
        node->axis = stride;
        (void)N;
        return detail::VariableAccess::make(std::move(node));
    }
    auto leaf = std::make_shared<detail::VariableNode>(
        std::move(out), false);
    return detail::VariableAccess::make(std::move(leaf));
}

}  // namespace ag
