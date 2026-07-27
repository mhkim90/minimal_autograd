#pragma once
// loss.h — Eigen-free losses on the Tensor-based autograd API.
//
// Both losses return a scalar Variable (Shape{}). The target tensor is
// supplied separately; it is consumed for forward computation only and is
// not part of the autograd graph.

#include "autograd/core/variable.h"

namespace ag {

// mse_loss(pred, target) = mean((pred - target)^2) over all elements.
// Any rank.
Variable mse_loss(const Variable& pred, const Tensor& target);

// cross_entropy(pred, target) treats the last axis as classes and
// averages over all leading sample axes. target is one-hot on the
// last axis. Any rank.
Variable cross_entropy(const Variable& pred, const Tensor& target);

}  // namespace ag
