#include "detail/tensor_kernels.h"

#include "tensor_dispatch_internal.h"

namespace ag {
namespace detail {

Tensor tensor_zeros(const Shape& shape, Device device) {
    return Tensor::zeros(shape, device);
}

Tensor tensor_ones(const Shape& shape, Device device) {
    return cpu_ops::tensor_ones(shape, device);
}

Tensor tensor_add(const Tensor& a, const Tensor& b) {
    require_same_shape("add", a, b);
    require_same_device("add", a, b);
    return cpu_ops::tensor_add(a, b);
}

Tensor tensor_mul(const Tensor& a, const Tensor& b) {
    require_same_shape("mul", a, b);
    require_same_device("mul", a, b);
    return cpu_ops::tensor_mul(a, b);
}

Tensor tensor_scale(const Tensor& a, float scalar) {
    return cpu_ops::tensor_scale(a, scalar);
}

Tensor tensor_softmax_nd(const Tensor& a, int axis,
                         Tensor& saved_softmax) {
    const int ax = normalize_axis(
        axis, static_cast<int>(a.shape().rank()), "softmax");
    return cpu_ops::tensor_softmax_nd(a, ax, saved_softmax);
}

Tensor tensor_softmax_backward_nd(const Tensor& g,
                                  const Tensor& saved_softmax,
                                  int axis) {
    require_same_shape("softmax_backward", g, saved_softmax);
    require_same_device("softmax_backward", g, saved_softmax);
    const int ax = normalize_axis(
        axis, static_cast<int>(g.shape().rank()), "softmax_backward");
    return cpu_ops::tensor_softmax_backward_nd(g, saved_softmax, ax);
}

Tensor tensor_log_softmax_nd(const Tensor& a, int axis, Tensor& saved_lsm) {
    const int ax = normalize_axis(
        axis, static_cast<int>(a.shape().rank()), "log_softmax");
    return cpu_ops::tensor_log_softmax_nd(a, ax, saved_lsm);
}

Tensor tensor_log_softmax_backward_nd(const Tensor& g,
                                      const Tensor& saved_lsm,
                                      int axis) {
    require_same_shape("log_softmax_backward", g, saved_lsm);
    require_same_device("log_softmax_backward", g, saved_lsm);
    const int ax = normalize_axis(
        axis, static_cast<int>(g.shape().rank()), "log_softmax_backward");
    return cpu_ops::tensor_log_softmax_backward_nd(g, saved_lsm, ax);
}

Tensor tensor_relu(const Tensor& a) {
    require_same_device("relu", a, a);
    return cpu_ops::tensor_relu(a);
}

Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("relu_backward", g, saved);
    require_same_device("relu_backward", g, saved);
    return cpu_ops::tensor_relu_backward(g, saved);
}

Tensor tensor_sigmoid(const Tensor& a) {
    return cpu_ops::tensor_sigmoid(a);
}

Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("sigmoid_backward", g, saved);
    require_same_device("sigmoid_backward", g, saved);
    return cpu_ops::tensor_sigmoid_backward(g, saved);
}

Tensor tensor_tanh(const Tensor& a) {
    return cpu_ops::tensor_tanh(a);
}

Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("tanh_backward", g, saved);
    require_same_device("tanh_backward", g, saved);
    return cpu_ops::tensor_tanh_backward(g, saved);
}

Tensor tensor_exp(const Tensor& a) {
    return cpu_ops::tensor_exp(a);
}

Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("exp_backward", g, saved);
    require_same_device("exp_backward", g, saved);
    return cpu_ops::tensor_exp_backward(g, saved);
}

Tensor tensor_log(const Tensor& a) {
    return cpu_ops::tensor_log(a);
}

Tensor tensor_log_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("log_backward", g, saved);
    require_same_device("log_backward", g, saved);
    return cpu_ops::tensor_log_backward(g, saved);
}

Tensor tensor_sqrt(const Tensor& a) {
    return cpu_ops::tensor_sqrt(a);
}

Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    require_same_shape("sqrt_backward", g, saved);
    require_same_device("sqrt_backward", g, saved);
    return cpu_ops::tensor_sqrt_backward(g, saved);
}

Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    return cpu_ops::tensor_silu_forward(a, sigmoid_out);
}

Tensor tensor_silu_backward(const Tensor& g,
                            const Tensor& x,
                            const Tensor& sig) {
    require_same_shape("silu_backward", g, x);
    require_same_shape("silu_backward", g, sig);
    require_same_device("silu_backward", g, x);
    require_same_device("silu_backward", g, sig);
    return cpu_ops::tensor_silu_backward(g, x, sig);
}

Tensor tensor_softplus(const Tensor& a) {
    return cpu_ops::tensor_softplus(a);
}

Tensor tensor_softplus_backward(const Tensor& g, const Tensor& x) {
    require_same_shape("softplus_backward", g, x);
    require_same_device("softplus_backward", g, x);
    return cpu_ops::tensor_softplus_backward(g, x);
}

Tensor tensor_sub(const Tensor& a, const Tensor& b) {
    require_same_shape("sub", a, b);
    require_same_device("sub", a, b);
    return cpu_ops::tensor_sub(a, b);
}

Tensor tensor_sub_backward_a(const Tensor& g) {
    return cpu_ops::tensor_sub_backward_a(g);
}

Tensor tensor_sub_backward_b(const Tensor& g) {
    return cpu_ops::tensor_sub_backward_b(g);
}

Tensor tensor_div(const Tensor& a, const Tensor& b) {
    require_same_shape("div", a, b);
    require_same_device("div", a, b);
    return cpu_ops::tensor_div(a, b);
}

Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    require_same_shape("div_backward_a", g, b);
    require_same_device("div_backward_a", g, b);
    return cpu_ops::tensor_div_backward_a(g, b);
}

Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b) {
    require_same_shape("div_backward_b", g, a);
    require_same_shape("div_backward_b", g, b);
    require_same_device("div_backward_b", g, a);
    require_same_device("div_backward_b", g, b);
    return cpu_ops::tensor_div_backward_b(g, a, b);
}

}  // namespace detail
}  // namespace ag
