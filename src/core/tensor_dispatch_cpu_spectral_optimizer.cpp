#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

#include "tensor_dispatch_internal.h"

namespace ag {
namespace detail {

TensorDFT2Result tensor_dft2_last2(const Tensor& real_in,
                                   const Tensor& imag_in,
                                   bool inverse, bool scale_output) {
    validate_dft2(real_in, imag_in);
    return cpu_ops::tensor_dft2_last2(
        real_in, imag_in, inverse, scale_output);
}

void optimizer_sgd_step(Variable& parameter, float lr) {
    const Tensor& grad = parameter.grad();
    std::vector<float> grad_data(grad.elements());
    grad.copy_to_host(grad_data.empty() ? nullptr : grad_data.data(),
                      grad_data.size());
    VariableAccess::apply_to_storage(
        parameter, [lr, &grad_data](float& element, std::size_t i) {
            element -= lr * grad_data[i];
        });
}

void optimizer_adam_step(Variable& parameter, Tensor& m, Tensor& v,
                         float lr, float beta1, float beta2, float eps,
                         float bias_correction1, float bias_correction2) {
    const Tensor& grad = parameter.grad();
    const std::size_t n = grad.elements();
    if (n == 0) return;

    std::vector<float> g_buf(n), m_buf(n), v_buf(n), p_buf(n);
    grad.copy_to_host(g_buf.data(), n);
    m.copy_to_host(m_buf.data(), n);
    v.copy_to_host(v_buf.data(), n);
    parameter.value().copy_to_host(p_buf.data(), n);

    std::vector<float> new_m(n), new_v(n), new_p(n);
    for (std::size_t i = 0; i < n; ++i) {
        new_m[i] = beta1 * m_buf[i] + (1.f - beta1) * g_buf[i];
        new_v[i] = beta2 * v_buf[i]
                   + (1.f - beta2) * g_buf[i] * g_buf[i];
        const float m_hat = new_m[i] / bias_correction1;
        const float v_hat = new_v[i] / bias_correction2;
        new_p[i] = p_buf[i] - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
    m.copy_from_host(new_m.data(), new_m.size());
    v.copy_from_host(new_v.data(), new_v.size());
    VariableAccess::copy_to_storage(parameter, new_p);
}

}  // namespace detail
}  // namespace ag
