#include "detail/tensor_kernels.h"

#include "detail/tensor_cuda_ops.h"
#include "detail/variable_internal.h"
#include "tensor_dispatch_internal.h"

namespace ag {
namespace detail {

TensorDFT2Result tensor_dft2_last2(const Tensor& real_in,
                                   const Tensor& imag_in,
                                   bool inverse, bool scale_output) {
    validate_dft2(real_in, imag_in);
    if (real_in.device().is_cuda()) {
        const CudaTensorDFT2Result cuda_out = cuda_tensor_dft2_last2(
            real_in, imag_in, inverse, scale_output);
        return TensorDFT2Result{cuda_out.real, cuda_out.imag};
    }
    return cpu_ops::tensor_dft2_last2(
        real_in, imag_in, inverse, scale_output);
}

void optimizer_sgd_step(Variable& parameter, float lr) {
    if (parameter.device().is_cuda()) {
        VariableAccess::apply_to_storage_cuda(
            parameter, [lr](float* p_data, const float* g_data,
                            int device, std::size_t n) {
                cuda_sgd_step(p_data, g_data, lr, n, device);
            });
        return;
    }

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
    if (parameter.device().is_cuda()) {
        VariableAccess::apply_to_storage_cuda(
            parameter,
            [&m, &v, lr, beta1, beta2, eps, bias_correction1,
             bias_correction2](float* p_data, const float* g_data,
                               int device, std::size_t n) {
                float* m_data = CudaTensorAccess::cuda_data_mutable(m);
                float* v_data = CudaTensorAccess::cuda_data_mutable(v);
                cuda_adam_step(p_data, m_data, v_data, g_data, lr, beta1,
                               beta2, eps, bias_correction1,
                               bias_correction2, n, device);
            });
        return;
    }

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
