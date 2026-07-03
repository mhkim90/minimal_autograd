#include "autograd/loss.h"

namespace ag {

VarPtr mse_loss(VarPtr pred, const Mat& target) {
    // diff = pred - target  (elementwise add, NOT broadcast_add — target has
    // the same shape as pred, not the (1, D) bias shape that broadcast_add
    // expects).
    auto target_var = Var::make(-target);
#ifdef AUTOGRAD_USE_CUDA
    if (pred->is_cuda()) target_var = target_var->cuda(pred->cuda_device());
#endif
    auto diff = add(pred, target_var);
    auto sq   = mul(diff, diff);
    int n = pred->data.rows() * pred->data.cols();
    return scale(sum(sq), 1.f / static_cast<float>(n));
}

VarPtr cross_entropy(VarPtr pred, const Mat& target) {
    auto lsm  = log_softmax(pred);                          // (N, C)
    auto target_var = Var::make(target);
#ifdef AUTOGRAD_USE_CUDA
    if (pred->is_cuda()) target_var = target_var->cuda(pred->cuda_device());
#endif
    auto prod = mul(lsm, target_var);                        // element-wise
    auto s    = sum(prod);                                   // scalar
    int n     = pred->data.rows();
    return scale(s, -1.f / static_cast<float>(n));
}

} // namespace ag
