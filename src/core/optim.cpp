// src/core/optim.cpp — optimizer implementation on the Tensor/Variable API.
//
// SGD applies p <- p - lr * grad in place through the narrow
// detail::VariableAccess::apply_to_storage helper. The Tensor storage
// identity is preserved; a Tensor alias taken before step() observes
// the post-step values.
//
// Adam prevalidates every eligible parameter before mutation, then
// computes all new moments and parameter values into temporaries so a
// failure cannot leave any state partially updated. Parameter
// mutation is committed through the same internal optimizer mutation
// boundary; moment mutation is committed into the optimizer's own
// moment Tensors (Tensor::copy_from_host) so the moments stored in an
// AdamState snapshot never alias the live optimizer's storage.

#include "autograd/core/optim.h"
#include "detail/variable_internal.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ag {
namespace optim {

namespace {

void validate_lr(const char* what, float lr) {
    if (!std::isfinite(lr)) {
        std::ostringstream os;
        os << what << ": learning rate must be finite (got " << lr << ")";
        throw std::invalid_argument(os.str());
    }
    if (lr < 0.f) {
        std::ostringstream os;
        os << what << ": learning rate must be non-negative (got " << lr
           << ")";
        throw std::invalid_argument(os.str());
    }
}

void validate_beta(const char* what, const char* which, float b) {
    if (!std::isfinite(b)) {
        std::ostringstream os;
        os << what << ": " << which << " must be finite (got " << b << ")";
        throw std::invalid_argument(os.str());
    }
    if (b < 0.f || b >= 1.f) {
        std::ostringstream os;
        os << what << ": " << which << " must lie in [0, 1) (got " << b
           << ")";
        throw std::invalid_argument(os.str());
    }
}

void validate_eps(const char* what, float eps) {
    if (!std::isfinite(eps)) {
        std::ostringstream os;
        os << what << ": eps must be finite (got " << eps << ")";
        throw std::invalid_argument(os.str());
    }
    if (eps <= 0.f) {
        std::ostringstream os;
        os << what << ": eps must be positive (got " << eps << ")";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

SGD::SGD(std::vector<Variable> params, float lr)
    : params_(std::move(params)), lr_(lr) {
    validate_lr("optim::SGD", lr_);
}

void SGD::step() {
    for (const auto& p : params_) {
        if (!p.requires_grad() || !p.has_grad()) {
            continue;
        }
        if (p.device().is_cuda()) {
            std::ostringstream os;
            os << "optim::SGD::step: CUDA tensors are not supported "
                  "(got device " << p.device().to_string() << ")";
            throw std::runtime_error(os.str());
        }
        if (p.grad().shape() != p.value().shape() ||
            p.grad().device() != p.device()) {
            throw std::invalid_argument(
                "optim::SGD::step: gradient does not match parameter");
        }
    }

    for (auto& p : params_) {
        if (!p.requires_grad() || !p.has_grad()) {
            continue;
        }

        const Tensor& grad = p.grad();
        if (grad.empty()) continue;

        std::vector<float> grad_data(grad.elements());
        grad.copy_to_host(grad_data.empty() ? nullptr : grad_data.data(),
                          grad_data.size());

        detail::VariableAccess::apply_to_storage(
            p, [lr = lr_, &grad_data](float& element, std::size_t i) {
                element -= lr * grad_data[i];
            });
    }
}

void SGD::zero_grad() {
    for (auto& p : params_) {
        p.zero_grad();
    }
}

// ── Adam ──────────────────────────────────────────────────────────────

Adam::Adam(std::vector<Variable> params,
           float lr, float beta1, float beta2, float eps)
    : params_(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      t_(0) {
    validate_lr("optim::Adam", lr_);
    validate_beta("optim::Adam", "beta1", beta1_);
    validate_beta("optim::Adam", "beta2", beta2_);
    validate_eps("optim::Adam", eps_);

    first_moments_.reserve(params_.size());
    second_moments_.reserve(params_.size());
    for (const auto& p : params_) {
        if (p.device().is_cuda()) {
            std::ostringstream os;
            os << "optim::Adam: CUDA tensors are not supported "
                  "(got device " << p.device().to_string() << ")";
            throw std::runtime_error(os.str());
        }
        first_moments_.push_back(
            Tensor::zeros(p.value().shape(), p.device()));
        second_moments_.push_back(
            Tensor::zeros(p.value().shape(), p.device()));
    }
}

void Adam::step() {
    // Phase 1: prevalidate every eligible parameter before any
    // mutation. If any check fails, no state or parameter is touched.
    for (const auto& p : params_) {
        if (!p.requires_grad() || !p.has_grad()) continue;
        if (p.device().is_cuda()) {
            std::ostringstream os;
            os << "optim::Adam::step: CUDA tensors are not supported "
                  "(got device " << p.device().to_string() << ")";
            throw std::runtime_error(os.str());
        }
        const Tensor& g = p.grad();
        if (g.shape() != p.value().shape() ||
            g.device() != p.device()) {
            throw std::invalid_argument(
                "optim::Adam::step: gradient does not match parameter");
        }
    }

    const int64_t new_t = t_ + 1;
    const float bc1 = 1.f - std::pow(beta1_, static_cast<float>(new_t));
    const float bc2 = 1.f - std::pow(beta2_, static_cast<float>(new_t));

    // Phase 2: compute every new (m, v, p) into temporaries so any
    // future error before commit cannot leave state half-updated.
    std::vector<std::vector<float>> new_m_buf(params_.size());
    std::vector<std::vector<float>> new_v_buf(params_.size());
    std::vector<std::vector<float>> new_p_buf(params_.size());
    std::vector<bool> eligible(params_.size(), false);
    bool any_eligible = false;

    for (std::size_t i = 0; i < params_.size(); ++i) {
        const auto& p = params_[i];
        if (!p.requires_grad() || !p.has_grad()) continue;
        eligible[i] = true;
        any_eligible = true;

        const Tensor& g = p.grad();
        const std::size_t n = g.elements();
        if (n == 0) continue;

        std::vector<float> g_buf(n);
        g.copy_to_host(g_buf.data(), n);
        std::vector<float> m_buf(n);
        first_moments_[i].copy_to_host(m_buf.data(), n);
        std::vector<float> v_buf(n);
        second_moments_[i].copy_to_host(v_buf.data(), n);
        std::vector<float> p_buf(n);
        p.value().copy_to_host(p_buf.data(), n);

        new_m_buf[i].resize(n);
        new_v_buf[i].resize(n);
        new_p_buf[i].resize(n);

        for (std::size_t j = 0; j < n; ++j) {
            new_m_buf[i][j] = beta1_ * m_buf[j] + (1.f - beta1_) * g_buf[j];
            new_v_buf[i][j] = beta2_ * v_buf[j]
                              + (1.f - beta2_) * g_buf[j] * g_buf[j];
            const float m_hat = new_m_buf[i][j] / bc1;
            const float v_hat = new_v_buf[i][j] / bc2;
            const float denom = std::sqrt(v_hat) + eps_;
            new_p_buf[i][j] = p_buf[j] - lr_ * m_hat / denom;
        }
    }

    // Phase 3: commit. t_ advances only when at least one parameter
    // is eligible, so empty/no-grad step() calls leave t_ unchanged.
    // Moments and parameters are updated only after every eligible
    // index has its new values computed.
    if (!any_eligible) return;
    t_ = new_t;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!eligible[i]) continue;
        first_moments_[i].copy_from_host(new_m_buf[i].data(),
                                         new_m_buf[i].size());
        second_moments_[i].copy_from_host(new_v_buf[i].data(),
                                          new_v_buf[i].size());
        detail::VariableAccess::copy_to_storage(params_[i], new_p_buf[i]);
    }
}

void Adam::zero_grad() {
    for (auto& p : params_) p.zero_grad();
}

AdamState Adam::state() const {
    AdamState s;
    s.t = t_;
    s.lr = lr_;
    s.beta1 = beta1_;
    s.beta2 = beta2_;
    s.eps = eps_;
    s.first_moments.reserve(first_moments_.size());
    s.second_moments.reserve(second_moments_.size());
    for (const auto& m : first_moments_) s.first_moments.push_back(m.clone());
    for (const auto& v : second_moments_) s.second_moments.push_back(v.clone());
    return s;
}

void Adam::load_state(const AdamState& s) {
    // Validate the entire snapshot before mutating any live state.
    if (s.t < 0) {
        throw std::invalid_argument(
            "optim::Adam::load_state: step count must be non-negative");
    }
    validate_lr("optim::Adam::load_state", s.lr);
    validate_beta("optim::Adam::load_state", "beta1", s.beta1);
    validate_beta("optim::Adam::load_state", "beta2", s.beta2);
    validate_eps("optim::Adam::load_state", s.eps);
    if (s.first_moments.size() != params_.size() ||
        s.second_moments.size() != params_.size()) {
        std::ostringstream os;
        os << "optim::Adam::load_state: moment count mismatch "
              "(got " << s.first_moments.size() << ", want "
           << params_.size() << ")";
        throw std::invalid_argument(os.str());
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        const auto& mp = s.first_moments[i];
        const auto& mv = s.second_moments[i];
        if (mp.shape() != params_[i].value().shape() ||
            mv.shape() != params_[i].value().shape()) {
            std::ostringstream os;
            os << "optim::Adam::load_state: moment shape mismatch at "
                  "index " << i;
            throw std::invalid_argument(os.str());
        }
        if (mp.device() != params_[i].device() ||
            mv.device() != params_[i].device()) {
            std::ostringstream os;
            os << "optim::Adam::load_state: moment device mismatch at "
                  "index " << i;
            throw std::invalid_argument(os.str());
        }
    }

    // Prepare deep copies of every moment before swapping anything.
    std::vector<Tensor> new_m;
    std::vector<Tensor> new_v;
    new_m.reserve(s.first_moments.size());
    new_v.reserve(s.second_moments.size());
    for (std::size_t i = 0; i < s.first_moments.size(); ++i) {
        new_m.push_back(s.first_moments[i].clone());
        new_v.push_back(s.second_moments[i].clone());
    }

    t_ = s.t;
    lr_ = s.lr;
    beta1_ = s.beta1;
    beta2_ = s.beta2;
    eps_ = s.eps;
    first_moments_ = std::move(new_m);
    second_moments_ = std::move(new_v);
}

}  // namespace optim
}  // namespace ag
