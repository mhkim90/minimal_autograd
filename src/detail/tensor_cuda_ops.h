#pragma once

#include "autograd/tensor.h"

#include <vector>

namespace ag {
namespace detail {

Tensor cuda_tensor_add(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_mul(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_less_equal(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_where(const Tensor& condition,
                         const Tensor& when_true,
                         const Tensor& when_false);
bool cuda_tensor_all_true(const Tensor& a);
bool cuda_tensor_all_finite(const Tensor& a);
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

// matmul forward + backward for the OOP N-D matmul free function.
// Inputs are required to satisfy the same rank/batch/inner contract
// as the CPU tensor_matmul_nd family; validation runs on the host
// before the kernels are launched.
Tensor cuda_tensor_matmul_nd(const Tensor& a, const Tensor& b);
Tensor cuda_tensor_matmul_backward_a_nd(const Tensor& g,
                                        const Tensor& b);
Tensor cuda_tensor_matmul_backward_b_nd(const Tensor& a,
                                        const Tensor& g);

Tensor cuda_tensor_conv2d_nchw_forward(const Tensor& input,
                                       const Tensor& weight,
                                       const Tensor& bias,
                                       int stride,
                                       int pad,
                                       Tensor& saved_col);
Tensor cuda_tensor_conv2d_nchw_backward_input(const Tensor& g,
                                              const Tensor& weight,
                                              int N, int C, int H, int W,
                                              int kH, int kW,
                                              int stride, int pad);
Tensor cuda_tensor_conv2d_nchw_backward_weight(const Tensor& g,
                                               const Tensor& col,
                                               const Shape& weight_shape);
Tensor cuda_tensor_conv2d_nchw_backward_bias(const Tensor& g);

Tensor cuda_tensor_maxpool2d_nchw_forward(const Tensor& input,
                                          int kH, int kW, int stride,
                                          Tensor& saved_mask);
Tensor cuda_tensor_maxpool2d_nchw_backward(const Tensor& g,
                                           const Tensor& mask,
                                           int N, int C, int H, int W,
                                           int kH, int kW, int stride);

Tensor cuda_tensor_zeros(const Shape& shape, Device device);
Tensor cuda_tensor_slice(const Tensor& a, int axis,
                         int64_t start, int64_t len);
Tensor cuda_tensor_slice_backward(const Tensor& g,
                                  const Shape& input_shape,
                                  int axis, int64_t start, int64_t len);
Tensor cuda_tensor_concat(const std::vector<Tensor>& inputs, int axis);
std::vector<Tensor> cuda_tensor_concat_backward(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis);

struct CudaTensorDFT2Result {
    Tensor real;
    Tensor imag;
};

CudaTensorDFT2Result cuda_tensor_dft2_last2(const Tensor& real,
                                            const Tensor& imag,
                                            bool inverse,
                                            bool scale_output);

// In-place parameter updates on CUDA for the OOP optim::SGD /
// optim::Adam free path. They operate directly on the device
// pointers supplied through the private CudaTensorAccess bridge so
// the optimizer hot path does not pay a host round-trip.
void cuda_sgd_step(float* p, const float* grad, float lr,
                   std::size_t n, int device);
void cuda_adam_step(float* p, float* m, float* v,
                    const float* grad,
                    float lr, float beta1, float beta2, float eps,
                    float bias_correction1, float bias_correction2,
                    std::size_t n, int device);

}  // namespace detail
}  // namespace ag
