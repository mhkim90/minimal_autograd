#pragma once

#include "autograd/tensor.h"

#include <cstdint>
#include <vector>

namespace ag {
namespace detail {

namespace cpu_ops {

Tensor tensor_add(const Tensor& a, const Tensor& b);
Tensor tensor_mul(const Tensor& a, const Tensor& b);
Tensor tensor_scale(const Tensor& a, float scalar);
Tensor tensor_sum(const Tensor& a);
Tensor tensor_broadcast_scalar(const Tensor& scalar, const Shape& target);
Tensor tensor_reshape_view(const Tensor& a, const Shape& target_shape);
Tensor tensor_concat_nd(const std::vector<Tensor>& inputs, int axis);
Tensor tensor_slice_nd(const Tensor& a, int axis, int64_t start, int64_t len);
Tensor tensor_slice_backward_nd(const Tensor& g,
                                const Shape& input_shape,
                                int axis, int64_t start, int64_t len);
std::vector<Tensor> tensor_concat_backward_nd(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis);
Tensor tensor_sum_axes_nd(const Tensor& a,
                          const std::vector<int>& axes, bool keep_dims);
Tensor tensor_sum_axes_backward_nd(const Tensor& g,
                                   const Shape& input_shape,
                                   const std::vector<int>& axes,
                                   bool keep_dims);
Tensor tensor_broadcast_add_nd(const Tensor& a, const Tensor& b);
Tensor tensor_broadcast_add_backward_nd(const Tensor& g,
                                        const Shape& input_shape);
Tensor tensor_matmul_nd(const Tensor& a, const Tensor& b);
Tensor tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b);
Tensor tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g);

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

Tensor tensor_sum(const Tensor& a);
Tensor tensor_broadcast_scalar(const Tensor& scalar, const Shape& target);
Tensor tensor_reshape_view(const Tensor& a, const Shape& target_shape);
Tensor tensor_concat_nd(const std::vector<Tensor>& inputs, int axis);
Tensor tensor_slice_nd(const Tensor& a, int axis, int64_t start, int64_t len);
Tensor tensor_slice_backward_nd(const Tensor& g,
                                const Shape& input_shape,
                                int axis, int64_t start, int64_t len);
std::vector<Tensor> tensor_concat_backward_nd(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis);
Tensor tensor_sum_axes_nd(const Tensor& a,
                          const std::vector<int>& axes, bool keep_dims);
Tensor tensor_sum_axes_backward_nd(const Tensor& g,
                                   const Shape& input_shape,
                                   const std::vector<int>& axes,
                                   bool keep_dims);
Tensor tensor_broadcast_add_nd(const Tensor& a, const Tensor& b);
Tensor tensor_broadcast_add_backward_nd(const Tensor& g,
                                        const Shape& input_shape);
Tensor tensor_matmul_nd(const Tensor& a, const Tensor& b);
Tensor tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b);
Tensor tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g);

}  // namespace detail
}  // namespace ag
