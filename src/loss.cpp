#include "autograd/loss.h"

namespace ag {

VarPtr mse_loss(VarPtr pred, const Mat& target) {
    // diff = pred - target  (elementwise add, NOT broadcast_add — target has
    // the same shape as pred, not the (1, D) bias shape that broadcast_add
    // expects).
    auto diff = add(pred, Var::make(-target));
    auto sq   = mul(diff, diff);
    int n = pred->data.rows() * pred->data.cols();
    return scale(sum(sq), 1.f / static_cast<float>(n));
}

VarPtr cross_entropy(VarPtr pred, const Mat& target) {
    auto lsm  = log_softmax(pred);                          // (N, C)
    auto prod = mul(lsm, Var::make(target));                 // element-wise
    auto s    = sum(prod);                                   // scalar
    int n     = pred->data.rows();
    return scale(s, -1.f / static_cast<float>(n));
}

} // namespace ag
