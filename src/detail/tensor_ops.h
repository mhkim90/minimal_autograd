#pragma once

#include "autograd/tensor.h"

#include <cstdint>
#include <vector>

namespace ag {

class Variable;

namespace detail {

struct TensorDFT2Result {
    Tensor real;
    Tensor imag;
};

namespace cpu_ops {

Tensor tensor_ones(const Shape& shape, Device device);
Tensor tensor_add(const Tensor& a, const Tensor& b);
Tensor tensor_mul(const Tensor& a, const Tensor& b);
Tensor tensor_less_equal(const Tensor& a, const Tensor& b);
Tensor tensor_where(const Tensor& condition,
                    const Tensor& when_true,
                    const Tensor& when_false);
bool tensor_all_true(const Tensor& a);
bool tensor_all_finite(const Tensor& a);
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

TensorDFT2Result tensor_dft2_last2(const Tensor& real_in,
                                   const Tensor& imag_in,
                                   bool inverse, bool scale_output);

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
Tensor tensor_softmax_nd(const Tensor& a, int axis, Tensor& saved_softmax);
Tensor tensor_softmax_backward_nd(const Tensor& g,
                                  const Tensor& saved_softmax,
                                  int axis);
Tensor tensor_log_softmax_nd(const Tensor& a, int axis, Tensor& saved_lsm);
Tensor tensor_log_softmax_backward_nd(const Tensor& g,
                                      const Tensor& saved_lsm,
                                      int axis);

Tensor tensor_sub(const Tensor& a, const Tensor& b);
Tensor tensor_sub_backward_a(const Tensor& g);
Tensor tensor_sub_backward_b(const Tensor& g);
Tensor tensor_div(const Tensor& a, const Tensor& b);
Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b);
Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b);

Tensor tensor_conv2d_nchw_forward(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int stride, int pad,
                                  Tensor& saved_col);
Tensor tensor_conv2d_nchw_backward_input(
    const Tensor& g, const Tensor& weight,
    int N, int C, int H, int W,
    int kH, int kW, int stride, int pad);
Tensor tensor_conv2d_nchw_backward_weight(
    const Tensor& g, const Tensor& col, const Shape& w_shape);
Tensor tensor_conv2d_nchw_backward_bias(const Tensor& g);
Tensor tensor_maxpool2d_nchw_forward(const Tensor& input,
                                     int kH, int kW, int stride,
                                     Tensor& saved_mask);
Tensor tensor_maxpool2d_nchw_backward(
    const Tensor& g, const Tensor& mask,
    int N, int C, int H, int W,
    int kH, int kW, int stride);

}  // namespace cpu_ops

Tensor tensor_ones(const Shape& shape, Device device);
Tensor tensor_add(const Tensor& a, const Tensor& b);
Tensor tensor_mul(const Tensor& a, const Tensor& b);
Tensor tensor_less_equal(const Tensor& a, const Tensor& b);
Tensor tensor_where(const Tensor& condition,
                    const Tensor& when_true,
                    const Tensor& when_false);
bool tensor_all_true(const Tensor& a);
bool tensor_all_finite(const Tensor& a);
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
Tensor tensor_softmax_nd(const Tensor& a, int axis, Tensor& saved_softmax);
Tensor tensor_softmax_backward_nd(const Tensor& g,
                                  const Tensor& saved_softmax,
                                  int axis);
Tensor tensor_log_softmax_nd(const Tensor& a, int axis, Tensor& saved_lsm);
Tensor tensor_log_softmax_backward_nd(const Tensor& g,
                                      const Tensor& saved_lsm,
                                      int axis);

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

Tensor tensor_conv2d_nchw_forward(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int stride, int pad,
                                  Tensor& saved_col);
Tensor tensor_conv2d_nchw_backward_input(
    const Tensor& g, const Tensor& weight,
    int N, int C, int H, int W,
    int kH, int kW, int stride, int pad);
Tensor tensor_conv2d_nchw_backward_weight(
    const Tensor& g, const Tensor& col, const Shape& w_shape);
Tensor tensor_conv2d_nchw_backward_bias(const Tensor& g);
Tensor tensor_maxpool2d_nchw_forward(const Tensor& input,
                                     int kH, int kW, int stride,
                                     Tensor& saved_mask);
Tensor tensor_maxpool2d_nchw_backward(
    const Tensor& g, const Tensor& mask,
    int N, int C, int H, int W,
    int kH, int kW, int stride);

Tensor tensor_zeros(const Shape& shape, Device device);
TensorDFT2Result tensor_dft2_last2(const Tensor& real_in,
                                   const Tensor& imag_in,
                                   bool inverse, bool scale_output);

void optimizer_sgd_step(Variable& parameter, float lr);
void optimizer_adam_step(Variable& parameter, Tensor& m, Tensor& v,
                         float lr, float beta1, float beta2, float eps,
                         float bias_correction1, float bias_correction2);

}  // namespace detail
}  // namespace ag
