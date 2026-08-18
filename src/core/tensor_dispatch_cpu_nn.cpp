#include "detail/tensor_kernels.h"

namespace ag {
namespace detail {

Tensor tensor_conv2d_nchw_forward(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int stride, int pad,
                                  Tensor& saved_col) {
    return cpu_ops::tensor_conv2d_nchw_forward(
        input, weight, bias, stride, pad, saved_col);
}

Tensor tensor_conv2d_nchw_backward_input(
    const Tensor& g, const Tensor& weight,
    int N, int C, int H, int W,
    int kH, int kW, int stride, int pad) {
    return cpu_ops::tensor_conv2d_nchw_backward_input(
        g, weight, N, C, H, W, kH, kW, stride, pad);
}

Tensor tensor_conv2d_nchw_backward_weight(
    const Tensor& g, const Tensor& col, const Shape& w_shape) {
    return cpu_ops::tensor_conv2d_nchw_backward_weight(g, col, w_shape);
}

Tensor tensor_conv2d_nchw_backward_bias(const Tensor& g) {
    return cpu_ops::tensor_conv2d_nchw_backward_bias(g);
}

Tensor tensor_maxpool2d_nchw_forward(const Tensor& input,
                                     int kH, int kW, int stride,
                                     Tensor& saved_mask) {
    return cpu_ops::tensor_maxpool2d_nchw_forward(
        input, kH, kW, stride, saved_mask);
}

Tensor tensor_maxpool2d_nchw_backward(
    const Tensor& g, const Tensor& mask,
    int N, int C, int H, int W,
    int kH, int kW, int stride) {
    return cpu_ops::tensor_maxpool2d_nchw_backward(
        g, mask, N, C, H, W, kH, kW, stride);
}

}  // namespace detail
}  // namespace ag
