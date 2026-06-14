#pragma once
// loss.h — MSELoss, CrossEntropyLoss.
//
// Both are composable from existing ops; no new Function subclass is needed.

#include "autograd/ops.h"
#include "autograd/variable.h"

namespace ag {

// mse_loss(pred, target) = mean((pred - target)^2)
VarPtr mse_loss(VarPtr pred, const Mat& target);

// cross_entropy(pred, target) = -mean(sum(target * log_softmax(pred), axis=1))
// target must be one-hot (N, C).
VarPtr cross_entropy(VarPtr pred, const Mat& target);

} // namespace ag
