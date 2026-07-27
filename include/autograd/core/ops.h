#pragma once
// Tensor-based autograd operations.
//
// Conventions:
//   * Elementwise binary operations require identical shapes and devices.
//     Matrix multiplication validates its inner dimensions; broadcast_add
//     expects (N, D) + (1, D).
//   * Matrix-specific operations are rank-2 only and treat the
//     underlying float32 storage as column-major to match the legacy
//     Eigen layout (flat index = row + rows * col).
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
Variable sum(const Variable& a);

// All rank-2-requiring ops throw std::invalid_argument with a message
// naming the offending shape when applied to a non-rank-2 input.

// --- Linear algebra ---

// matmul(a, b) requires a (M, K) and b (K, N); returns (M, N).
Variable matmul(const Variable& a, const Variable& b);

// --- Reductions ---

// Full reduction to a scalar (Shape{}).
Variable mean(const Variable& a);

// --- Elementwise broadcast / arithmetic ---

// broadcast_add: a (N, D) + b (1, D). Per-row bias add.
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

// softmax and log_softmax are per-row on a rank-2 input.
Variable softmax(const Variable& a);
Variable log_softmax(const Variable& a);

// --- Shape / layout ---

Variable transpose(const Variable& a);
Variable reshape(const Variable& a, int64_t rows, int64_t cols);
// concat stacks inputs vertically (all inputs share the same cols).
Variable concat(std::vector<Variable> inputs);
// hcat stacks inputs horizontally (all inputs share the same rows).
Variable hcat(std::vector<Variable> inputs);

// --- Slicing / accumulation ---

// col_slice(a, start, len) and row_slice(a, start, len) take a contiguous
// range. Both throw on out-of-range indices.
Variable col_slice(const Variable& a, int64_t start, int64_t len);
Variable row_slice(const Variable& a, int64_t start, int64_t len);
// split(x) splits cols evenly in two; requires an even number of cols.
std::pair<Variable, Variable> split(const Variable& a);

// cumsum(a, axis) and flip(a, axis): axis 0 = rows, axis 1 = cols.
Variable cumsum(const Variable& a, int axis = 1);
Variable flip(const Variable& a, int axis = 1);

// --- Clamp ---

Variable clamp(const Variable& a, float lo, float hi);

}  // namespace ag
