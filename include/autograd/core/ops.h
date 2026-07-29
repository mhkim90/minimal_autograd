#pragma once
// Tensor-based autograd operations.
//
// Conventions:
//   * All operations are rank-agnostic. Storage is last-axis contiguous
//     (canonical row-major): stride[n-1] = 1 and
//     stride[i] = stride[i+1] * dim[i+1]. For rank-2, flat index is
//     row * columns + col.
//   * Elementwise binary operations require identical shapes and
//     devices. Matrix multiplication validates its inner dimensions
//     and requires identical leading batch dims with rank >= 2.
//     broadcast_add supports symmetric trailing-dimension broadcasting.
//   * Negative axes are normalized against the input rank and
//     out-of-range or duplicate axes throw std::invalid_argument.
//   * Backward closures are private; only forward Variable values,
//     gradients, and shapes are visible to callers.
//   * These headers are Eigen- and CUDA-free.

#include "autograd/core/variable.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ag {

Variable add(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable scale(const Variable& a, float scalar);

// Full reduction to a scalar (Shape{}). Any rank.
Variable sum(const Variable& a);
// Reduce along the given axes. keep_dims=false removes the reduced axes
// from the output shape; keep_dims=true keeps them as 1s.
Variable sum(const Variable& a, const std::vector<int>& axes,
             bool keep_dims = false);

// --- Linear algebra ---

// matmul(a, b) requires rank >= 2 for both inputs. The last two axes
// are the matrix dimensions; the leading dimensions must match
// (batched matmul). Returns (..., M, N). Both inputs must share a
// device; CPU and CUDA are both supported and the result stays on
// that device, including the operand gradients.
Variable matmul(const Variable& a, const Variable& b);

// --- Reductions ---

// Full reduction to a scalar (Shape{}). Any rank.
Variable mean(const Variable& a);
// Reduce along the given axes. Equivalent to sum / scale.
Variable mean(const Variable& a, const std::vector<int>& axes,
              bool keep_dims = false);

// --- Elementwise broadcast / arithmetic ---

// broadcast_add: symmetric trailing-dimension broadcasting. For rank-2
// this preserves the legacy (N, D) + (1, D) contract.
Variable broadcast_add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable div_op(const Variable& a, const Variable& b);

// --- Activations / scalar functions ---

Variable relu(const Variable& a);
Variable sigmoid(const Variable& a);
Variable tanh_op(const Variable& a);
Variable exp_op(const Variable& a);
Variable log_op(const Variable& a);
Variable sqrt_op(const Variable& a);
Variable silu(const Variable& a);
Variable softplus(const Variable& a);
Variable sin_op(const Variable& a);
Variable cos_op(const Variable& a);

// --- Numerical / normalized ---

// softmax and log_softmax along an axis (default last axis). Any rank.
Variable softmax(const Variable& a, int axis = -1);
Variable log_softmax(const Variable& a, int axis = -1);

// --- Shape / layout ---

// transpose swaps the final two axes (rank-2 compatibility form).
Variable transpose(const Variable& a);
// transpose swaps two arbitrary axes.
Variable transpose(const Variable& a, int axis0, int axis1);

// Reshape to a Shape (any rank with matching numel).
Variable reshape(const Variable& a, const Shape& shape);
// Rank-2 compatibility: reshape to (rows, cols).
Variable reshape(const Variable& a, int64_t rows, int64_t cols);

// Concat along an axis (default 0). Any rank, distinct non-axis dims.
Variable concat(std::vector<Variable> inputs, int axis = 0);
// hcat stacks along the last axis. Any rank, distinct non-last dims.
Variable hcat(std::vector<Variable> inputs);

// --- Slicing / accumulation ---

// Generic slice along an axis: keep axis[start : start + len].
Variable slice(const Variable& a, int axis, int64_t start, int64_t len);
// col_slice / row_slice are compatibility wrappers for slice along
// axes 1 / 0 and also work on higher-rank inputs.
Variable col_slice(const Variable& a, int64_t start, int64_t len);
Variable row_slice(const Variable& a, int64_t start, int64_t len);
// split(x) splits along the last axis by default. Requires an even
// length along that axis.
std::pair<Variable, Variable> split(const Variable& a, int axis = -1);

// cumsum / flip along an axis (default last axis). Any rank.
Variable cumsum(const Variable& a, int axis = -1);
Variable flip(const Variable& a, int axis = -1);

// --- Clamp ---

Variable clamp(const Variable& a, float lo, float hi);

// --- N-D spatial ops (NCHW rank-4) ---
//
// Inputs to conv2d / max_pool2d / depthwise_conv2d / avg_pool2d /
// nearest_upsample2d must be rank-4 NCHW tensors. Weights for
// conv2d are rank-4 (out_C, in_C, kH, kW); weights for
// depthwise_conv2d are rank-3 (C, kH, kW). Bias (where accepted)
// must be a rank-1 tensor of length equal to the channel axis. Stride
// must be a positive integer; pad must be a non-negative integer such
// that the resulting output extent is well-defined (i.e. (H + 2*pad -
// kH) is a non-negative multiple of stride). All operations are
// CPU-only: a non-CPU operand raises std::runtime_error; shape,
// geometry, and channel-mismatch errors raise std::invalid_argument.

// 2D convolution. Computes the standard cross-correlation:
//
//   out[n, oc, oh, ow] = bias[oc]
//                       + sum_{c, kh, kw} weight[oc, c, kh, kw]
//                         * input[n, c, oh*stride + kh - pad,
//                                 ow*stride + kw - pad]
//
// with implicit zero padding outside the input plane. The forward
// graph records the im2col matrix and the weight tensor for backward.
Variable conv2d(const Variable& input,
               const Variable& weight,
               const Variable& bias,
               int stride,
               int pad);

// 2D max pooling. stride defaults to kernel size (non-overlapping).
// The forward graph records the per-output argmax position as a
// rank-4 (N, C, oH, oW) float tensor of kernel-flat indices; backward
// routes the upstream gradient to the recorded argmax (with
// accumulation across overlapping windows) and zeros elsewhere.
Variable max_pool2d(const Variable& input,
                    int kH, int kW,
                    int stride);

// 2D depthwise convolution. Each input channel c is convolved with
// its own (kH, kW) filter weight[c, :, :] and shifted by bias[c].
// The forward graph records the im2col matrix and the weight tensor
// for backward, mirroring ag::conv2d.
Variable depthwise_conv2d(const Variable& input,
                          const Variable& weight,
                          const Variable& bias,
                          int stride,
                          int pad);

// 2D average pooling. Output is the mean over each kH x kW window.
// Backward distributes (1 / (kH*kW)) of the upstream gradient across
// every window that contains each input pixel, with overlap
// accumulation across windows.
Variable avg_pool2d(const Variable& input,
                    int kH, int kW,
                    int stride);

// 2D nearest-neighbor upsampling. Each input pixel is replicated to
// a scale x scale block of output positions. scale must be >= 1.
// Backward sums the upstream gradient over the scale x scale block
// per input pixel.
Variable nearest_upsample2d(const Variable& input,
                            int scale);

// Group normalization. Normalizes over (channels_per_group * H * W)
// elements per (sample, group) where channels are partitioned into
// num_groups contiguous groups of channels_per_group = C / num_groups
// channels each. Returns the per-channel affine transform
//   y[n, c, h, w] = gamma[c] * xhat[n, c, h, w] + beta[c]
// where xhat is the normalized value with zero mean and unit variance
// over the group. num_groups must be a positive integer and must
// divide C. eps must be finite and positive.
Variable group_norm(const Variable& input,
                    const Variable& gamma,
                    const Variable& beta,
                    int num_groups,
                    float eps);

}  // namespace ag
