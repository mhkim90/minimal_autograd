#pragma once
// Tensor-based autograd operations.
//
// Conventions:
//   * All operations are rank-agnostic. Storage is first-axis contiguous
//     (stride 0 = 1; stride[i] = stride[i-1] * dim[i-1]). For rank-2
//     this matches the legacy Eigen column-major layout (flat index
//     = row + rows * col).
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
// (batched matmul). Returns (..., M, N).
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

}  // namespace ag
