#pragma once

#include "autograd/tensor.h"

#include <vector>

namespace ag {
namespace detail {

Tensor cuda_tensor_add(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_mul(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_scale(const Tensor& a, float scalar);
Tensor cuda_tensor_ones(const Shape& shape, Device device);
Tensor cuda_tensor_sum(const Tensor& a);
Tensor cuda_tensor_broadcast_scalar(const Tensor& scalar,
                                    const Shape& target);
Tensor cuda_tensor_relu(const Tensor& a);
Tensor cuda_tensor_relu_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_sigmoid(const Tensor& a);
Tensor cuda_tensor_sigmoid_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_tanh(const Tensor& a);
Tensor cuda_tensor_tanh_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_exp(const Tensor& a);
Tensor cuda_tensor_exp_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_log(const Tensor& a);
Tensor cuda_tensor_log_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_sqrt(const Tensor& a);
Tensor cuda_tensor_sqrt_backward(const Tensor& g, const Tensor& saved);
Tensor cuda_tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out);
Tensor cuda_tensor_silu_backward(const Tensor& g,
                                 const Tensor& x,
                                 const Tensor& sig);
Tensor cuda_tensor_softplus(const Tensor& a);
Tensor cuda_tensor_softplus_backward(const Tensor& g, const Tensor& x);
Tensor cuda_tensor_sub(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_sub_backward_b(const Tensor& g);
Tensor cuda_tensor_div(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_div_backward_a(const Tensor& g, const Tensor& b);
Tensor cuda_tensor_div_backward_b(const Tensor& g,
                                  const Tensor& a,
                                  const Tensor& b);

Tensor cuda_tensor_broadcast_add(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_broadcast_add_backward(const Tensor& g,
                                          const Shape& input_shape);
Tensor cuda_tensor_sum_axes(const Tensor& a,
                            const std::vector<int>& axes,
                            bool keep_dims);
Tensor cuda_tensor_sum_axes_backward(const Tensor& g,
                                     const Shape& input_shape,
                                     const std::vector<int>& axes,
                                     bool keep_dims);
Tensor cuda_tensor_softmax(const Tensor& a, int axis, Tensor& saved_softmax);
Tensor cuda_tensor_softmax_backward(const Tensor& g,
                                    const Tensor& saved_softmax,
                                    int axis);
Tensor cuda_tensor_log_softmax(const Tensor& a,
                               int axis,
                               Tensor& saved_log_softmax);
Tensor cuda_tensor_log_softmax_backward(const Tensor& g,
                                        const Tensor& saved_log_softmax,
                                        int axis);

}  // namespace detail
}  // namespace ag
