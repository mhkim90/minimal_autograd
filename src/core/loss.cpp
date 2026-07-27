// src/core/loss.cpp — losses on the Tensor-based autograd API.
//
// mse_loss and cross_entropy are each composed from a small graph of
// Tensor-based ops (see autograd/core/ops.h). The target tensor is
// supplied by value and never enters the autograd graph; it is wrapped
// in a non-grad Variable for the binary ops it participates in.

#include "autograd/core/loss.h"
#include "autograd/core/ops.h"

#include <cstddef>

namespace ag {

Variable mse_loss(const Variable& pred, const Tensor& target) {
    Variable target_v = Variable(target.clone(), /*requires_grad=*/false);
    Variable diff = sub(pred, target_v);
    Variable sq = mul(diff, diff);
    const std::size_t n = pred.value().elements();
    if (n == 0) return sum(sq);
    return scale(sum(sq), 1.f / static_cast<float>(n));
}

Variable cross_entropy(const Variable& pred, const Tensor& target) {
    Variable target_v = Variable(target.clone(), /*requires_grad=*/false);
    Variable lsm = log_softmax(pred);
    Variable prod = mul(lsm, target_v);
    Variable s = sum(prod);
    const int64_t batch = pred.value().shape()[0];
    if (batch == 0) {
        return scale(s, -1.f);
    }
    return scale(s, -1.f / static_cast<float>(batch));
}

}  // namespace ag
