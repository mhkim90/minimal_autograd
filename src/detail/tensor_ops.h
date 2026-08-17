#pragma once

#include "autograd/tensor.h"

namespace ag {
namespace detail {

namespace cpu_ops {

Tensor tensor_add(const Tensor& a, const Tensor& b);
Tensor tensor_mul(const Tensor& a, const Tensor& b);
Tensor tensor_scale(const Tensor& a, float scalar);

Tensor tensor_relu(const Tensor& a);
Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_sigmoid(const Tensor& a);
Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_tanh(const Tensor& a);
Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_exp(const Tensor& a);
Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_log(const Tensor& a);
Tensor tensor_log_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_sqrt(const Tensor& a);
Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out);
Tensor tensor_silu_backward(const Tensor& g,
                            const Tensor& x,
                            const Tensor& sig);
Tensor tensor_softplus(const Tensor& a);
Tensor tensor_softplus_backward(const Tensor& g, const Tensor& x);

Tensor tensor_sub(const Tensor& a, const Tensor& b);
Tensor tensor_sub_backward_a(const Tensor& g);
Tensor tensor_sub_backward_b(const Tensor& g);
Tensor tensor_div(const Tensor& a, const Tensor& b);
Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b);
Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b);

}  // namespace cpu_ops

Tensor tensor_add(const Tensor& a, const Tensor& b);
Tensor tensor_mul(const Tensor& a, const Tensor& b);
Tensor tensor_scale(const Tensor& a, float scalar);

Tensor tensor_relu(const Tensor& a);
Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_sigmoid(const Tensor& a);
Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_tanh(const Tensor& a);
Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_exp(const Tensor& a);
Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_log(const Tensor& a);
Tensor tensor_log_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_sqrt(const Tensor& a);
Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved);
Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out);
Tensor tensor_silu_backward(const Tensor& g,
                            const Tensor& x,
                            const Tensor& sig);
Tensor tensor_softplus(const Tensor& a);
Tensor tensor_softplus_backward(const Tensor& g, const Tensor& x);

Tensor tensor_sub(const Tensor& a, const Tensor& b);
Tensor tensor_sub_backward_a(const Tensor& g);
Tensor tensor_sub_backward_b(const Tensor& g);
Tensor tensor_div(const Tensor& a, const Tensor& b);
Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b);
Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b);

}  // namespace detail
}  // namespace ag
