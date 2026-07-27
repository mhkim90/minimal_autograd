// src/core/loss.cpp — losses on the Tensor-based autograd API.
//
// mse_loss and cross_entropy are each composed from a small graph of
// Tensor-based ops (see autograd/core/ops.h). The target tensor is
// supplied by value and never enters the autograd graph; it is wrapped
// in a non-grad Variable for the binary ops it participates in.
//
// cross_entropy treats the last axis as classes and averages over all
// leading sample axes. For shape (N, C) this is equivalent to the old
// 2-D contract; for rank-N inputs the loss is the mean over all
// leading dims.

#include "autograd/core/loss.h"
#include "autograd/core/ops.h"

#include <cstddef>
#include <stdexcept>

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
    const std::size_t rank = pred.value().shape().rank();
    if (rank == 0 || pred.value().shape()[static_cast<int>(rank) - 1] == 0) {
        throw std::invalid_argument(
            "cross_entropy: requires a non-empty class axis");
    }
    Variable target_v = Variable(target.clone(), /*requires_grad=*/false);
    // Classes live on the last axis (-1).
    Variable lsm = log_softmax(pred, -1);
    Variable prod = mul(lsm, target_v);
    // Sum over the last axis (classes), then mean over all leading dims.
    Variable total = sum(prod);
    int64_t samples = 1;
    for (std::size_t i = 0; i + 1 < rank; ++i) {
        samples = detail::mul_check_overflow(
            samples, pred.value().shape()[static_cast<int>(i)],
            "cross_entropy");
    }
    if (samples == 0) return scale(total, -1.f);
    return scale(total, -1.f / static_cast<float>(samples));
}

}  // namespace ag
