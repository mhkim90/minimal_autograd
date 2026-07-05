#include "autograd/optim.h"
#include <cmath>
#include <stdexcept>

namespace ag {

Adam::Adam(std::vector<VarPtr> params,
           float lr, float beta1, float beta2, float eps)
    : params(std::move(params)), lr(lr),
      beta1(beta1), beta2(beta2), eps(eps) {
    m.reserve(this->params.size());
    v.reserve(this->params.size());
#ifdef AUTOGRAD_USE_CUDA
    cuda_m.reserve(this->params.size());
    cuda_v.reserve(this->params.size());
#endif
    for (auto& p : this->params) {
        m.push_back(Mat::Zero(p->data.rows(), p->data.cols()));
        v.push_back(Mat::Zero(p->data.rows(), p->data.cols()));
#ifdef AUTOGRAD_USE_CUDA
        if (p->is_cuda()) {
            cuda_m.push_back(Var::make(Mat::Zero(p->data.rows(), p->data.cols()))
                                 ->cuda(p->cuda_device()));
            cuda_v.push_back(Var::make(Mat::Zero(p->data.rows(), p->data.cols()))
                                 ->cuda(p->cuda_device()));
        } else {
            cuda_m.push_back(nullptr);
            cuda_v.push_back(nullptr);
        }
#endif
    }
}

void Adam::step() {
    t += 1;
    const float bc1 = 1.f - std::pow(beta1, t);
    const float bc2 = 1.f - std::pow(beta2, t);
    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
#ifdef AUTOGRAD_USE_CUDA
        if (p->is_cuda()) {
            cuda_adam_step(*p, cuda_m[i]->cuda_data(), cuda_v[i]->cuda_data(),
                           lr, beta1, beta2, eps, bc1, bc2);
            continue;
        }
#endif
        m[i] = beta1 * m[i] + (1.f - beta1) * p->grad;
        v[i] = beta2 * v[i] + (1.f - beta2) * p->grad.cwiseProduct(p->grad);

        Mat m_hat = m[i] / bc1;
        Mat v_hat = v[i] / bc2;
        Mat denom = v_hat.cwiseSqrt() + Mat::Constant(v_hat.rows(), v_hat.cols(), eps);
        p->data -= lr * m_hat.cwiseQuotient(denom);
    }
}

void Adam::zero_grad() {
    for (auto& p : params) {
        p->clear_grad();
    }
}

} // namespace ag
