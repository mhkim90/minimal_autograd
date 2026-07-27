// src/core/optim.cpp — Phase 5a optim implementation on the
// Tensor/Variable API.
//
// SGD applies p <- p - lr * grad in place through the narrow
// detail::VariableAccess::apply_to_storage helper. The Tensor storage
// identity is preserved; a Tensor alias taken before step() observes
// the post-step values.

#include "autograd/core/optim.h"
#include "detail/variable_internal.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
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
            os << "optim::SGD::step: CUDA tensors are not supported in "
                  "Phase 5a (got device " << p.device().to_string() << ")";
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

}  // namespace optim
}  // namespace ag
