#pragma once
// optim.h — SGD, Adam.

#include "autograd/cuda_core.h"
#include "autograd/variable.h"
#include <vector>

namespace ag {

struct SGD {
    std::vector<VarPtr> params;
    float lr;

    SGD(std::vector<VarPtr> params, float lr)
        : params(std::move(params)), lr(lr) {}

    void step() {
        for (auto& p : params) {
#ifdef AUTOGRAD_USE_CUDA
            if (p->is_cuda()) {
                cuda_sgd_step(*p, lr);
                continue;
            }
#endif
            p->data -= lr * p->grad;
        }
    }

    void zero_grad() {
        for (auto& p : params) p->grad.setZero();
    }
};

struct Adam {
    std::vector<VarPtr> params;
    float lr;
    float beta1, beta2, eps;
    int t = 0;
    std::vector<Mat> m;   // first moment
    std::vector<Mat> v;   // second moment

    Adam(std::vector<VarPtr> params,
         float lr    = 1e-3f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps   = 1e-8f);

    void step();
    void zero_grad();
};

} // namespace ag
