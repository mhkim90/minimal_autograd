#pragma once
// loss.h — Eigen-free losses on the Tensor-based autograd API.
//
// Both losses return a scalar Variable (Shape{}). The target tensor is
// supplied separately; it is consumed for forward computation only and is
// not part of the autograd graph.

#include "autograd/core/variable.h"

namespace ag {

// mse_loss(pred, target) = mean((pred - target)^2), elementwise.
Variable mse_loss(const Variable& pred, const Tensor& target);

// cross_entropy(pred, target) = -mean(sum(target * log_softmax(pred),
// axis=1)); target is one-hot (N, C).
Variable cross_entropy(const Variable& pred, const Tensor& target);

}  // namespace ag
