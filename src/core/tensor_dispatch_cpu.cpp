#include "detail/tensor_kernels.h"
#include "detail/variable_internal.h"

namespace ag {
namespace detail {

namespace {

inline void validate_reshape(const Tensor& a, const Shape& target_shape) {
    const int64_t expected = static_cast<int64_t>(a.elements());
    const int64_t target = target_shape.numel();
    if (expected != target) {
        std::ostringstream os;
        os << "reshape: numel mismatch (have " << expected
           << ", want " << target << ")";
        throw std::invalid_argument(os.str());
    }
}

inline void validate_concat(const std::vector<Tensor>& inputs, int axis) {
    if (inputs.empty()) {
        throw std::invalid_argument("concat: requires at least one input");
    }
    const Shape& first = inputs[0].shape();
    const int rank = static_cast<int>(first.rank());
    const int ax = normalize_axis(axis, rank, "concat");
    int64_t total = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].device() != inputs[0].device()) {
            throw std::invalid_argument("concat: device mismatch");
        }
        if (inputs[i].shape().rank() != static_cast<std::size_t>(rank)) {
            std::ostringstream os;
            os << "concat: rank mismatch at input " << i
               << " (input rank " << inputs[i].shape().rank()
               << " vs expected " << rank << ")";
            throw std::invalid_argument(os.str());
        }
        for (int d = 0; d < rank; ++d) {
            if (d == ax) continue;
            if (inputs[i].shape()[d] != first[d]) {
                std::ostringstream os;
                os << "concat: dim " << d << " mismatch at input " << i
                   << " (have " << inputs[i].shape()[d]
                   << ", want " << first[d] << ")";
                throw std::invalid_argument(os.str());
            }
        }
        total = checked_dim_sum("concat", total, inputs[i].shape()[ax]);
    }
}

inline void validate_matmul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("matmul: device mismatch");
    }
    const Shape& sa = a.shape();
    const Shape& sb = b.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_b = static_cast<int>(sb.rank());
    if (rank_a < 2 || rank_b < 2) {
        std::ostringstream os;
        os << "matmul: requires rank >= 2 (got " << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    if (rank_a != rank_b) {
        std::ostringstream os;
        os << "matmul: rank mismatch (" << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    if (sa[rank_a - 1] != sb[rank_b - 2]) {
        std::ostringstream os;
        os << "matmul: inner dimensions mismatch ("
           << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sb[d]) {
            std::ostringstream os;
            os << "matmul: batch dim " << d << " mismatch ("
               << sa[d] << " vs " << sb[d] << ")";
            throw std::invalid_argument(os.str());
        }
    }
}

inline void validate_matmul_backward_a(const Tensor& g, const Tensor& b) {
    const Shape& sg = g.shape();
    const Shape& sb = b.shape();
    const int rank_g = static_cast<int>(sg.rank());
    const int rank_b = static_cast<int>(sb.rank());
    if (rank_g < 2 || rank_b < 2) {
        throw std::invalid_argument(
            "matmul_backward_a: requires rank >= 2");
    }
    if (rank_g != rank_b) {
        throw std::invalid_argument("matmul_backward_a: rank mismatch");
    }
    if (sg[rank_g - 1] != sb[rank_b - 1]) {
        throw std::invalid_argument(
            "matmul_backward_a: g/b inner mismatch");
    }
    for (int d = 0; d < rank_g - 2; ++d) {
        if (sg[d] != sb[d]) {
            throw std::invalid_argument("matmul_backward_a: batch mismatch");
        }
    }
}

inline void validate_matmul_backward_b(const Tensor& a, const Tensor& g) {
    const Shape& sa = a.shape();
    const Shape& sg = g.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_g = static_cast<int>(sg.rank());
    if (rank_a < 2 || rank_g < 2) {
        throw std::invalid_argument(
            "matmul_backward_b: requires rank >= 2");
    }
    if (rank_a != rank_g) {
        throw std::invalid_argument("matmul_backward_b: rank mismatch");
    }
    if (sg[rank_g - 2] != sa[rank_a - 2]) {
        throw std::invalid_argument(
            "matmul_backward_b: a/g inner mismatch");
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sg[d]) {
            throw std::invalid_argument("matmul_backward_b: batch mismatch");
        }
    }
}

inline void validate_dft2(const Tensor& real_in, const Tensor& imag_in) {
    const Shape& s = real_in.shape();
    if (s != imag_in.shape()) {
        std::ostringstream os;
        os << "tensor_dft2_last2: real/imag shape mismatch ("
           << s << " vs " << imag_in.shape() << ")";
        throw std::invalid_argument(os.str());
    }
    if (real_in.device() != imag_in.device()) {
        throw std::invalid_argument(
            "tensor_dft2_last2: real/imag device mismatch");
    }
    const int rank = static_cast<int>(s.rank());
    if (rank < 2) {
        std::ostringstream os;
        os << "tensor_dft2_last2: input must have rank >= 2 (got "
           << s << ")";
        throw std::invalid_argument(os.str());
    }
    const int H = static_cast<int>(s[rank - 2]);
    const int W = static_cast<int>(s[rank - 1]);
    if (H <= 0 || W <= 0) {
        std::ostringstream os;
        os << "tensor_dft2_last2: last two dimensions must be positive (got "
           << H << " x " << W << ")";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

Tensor tensor_zeros(const Shape& shape, Device device) {
    return Tensor::zeros(shape, device);
}

Tensor tensor_ones(const Shape& shape, Device device) {
    return cpu_ops::tensor_ones(shape, device);
}

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

Tensor tensor_sum(const Tensor& a) {
    return cpu_ops::tensor_sum(a);
}

Tensor tensor_broadcast_scalar(const Tensor& scalar, const Shape& target) {
    return cpu_ops::tensor_broadcast_scalar(scalar, target);
}

Tensor tensor_reshape_view(const Tensor& a, const Shape& target_shape) {
    validate_reshape(a, target_shape);
    return cpu_ops::tensor_reshape_view(a, target_shape);
}

Tensor tensor_concat_nd(const std::vector<Tensor>& inputs, int axis) {
    validate_concat(inputs, axis);
    return cpu_ops::tensor_concat_nd(inputs, axis);
}

Tensor tensor_slice_nd(const Tensor& a, int axis, int64_t start, int64_t len) {
    const int ax = normalize_axis(axis, static_cast<int>(a.shape().rank()), "slice");
    if (start < 0 || len <= 0 || start > a.shape()[ax] ||
        len > a.shape()[ax] - start) {
        std::ostringstream os;
        os << "slice: out of range (axis " << ax << " dim " << a.shape()[ax]
           << ", start " << start << ", len " << len << ")";
        throw std::invalid_argument(os.str());
    }
    return cpu_ops::tensor_slice_nd(a, axis, start, len);
}

Tensor tensor_slice_backward_nd(const Tensor& g, const Shape& input_shape,
                                int axis, int64_t start, int64_t len) {
    normalize_axis(axis, static_cast<int>(input_shape.rank()), "slice_backward");
    return cpu_ops::tensor_slice_backward_nd(g, input_shape, axis, start, len);
}

std::vector<Tensor> tensor_concat_backward_nd(
    const Tensor& g, const std::vector<int64_t>& along_per_input,
    const std::vector<Shape>& input_shapes, int axis) {
    if (input_shapes.empty()) {
        throw std::invalid_argument("concat_backward: empty inputs");
    }
    normalize_axis(axis, static_cast<int>(input_shapes[0].rank()),
                   "concat_backward");
    if (along_per_input.size() != input_shapes.size()) {
        throw std::invalid_argument(
            "concat_backward: along/size count mismatch");
    }
    return cpu_ops::tensor_concat_backward_nd(
        g, along_per_input, input_shapes, axis);
}

Tensor tensor_sum_axes_nd(const Tensor& a, const std::vector<int>& axes,
                          bool keep_dims) {
    normalize_axes(axes, static_cast<int>(a.shape().rank()), "sum");
    return cpu_ops::tensor_sum_axes_nd(a, axes, keep_dims);
}

Tensor tensor_sum_axes_backward_nd(const Tensor& g, const Shape& input_shape,
                                   const std::vector<int>& axes,
                                   bool keep_dims) {
    normalize_axes(axes, static_cast<int>(input_shape.rank()), "sum_backward");
    return cpu_ops::tensor_sum_axes_backward_nd(
        g, input_shape, axes, keep_dims);
}

Tensor tensor_broadcast_add_nd(const Tensor& a, const Tensor& b) {
    require_same_device("broadcast_add", a, b);
    const int rank_a = static_cast<int>(a.shape().rank());
    const int rank_b = static_cast<int>(b.shape().rank());
    const int out_rank = std::max(rank_a, rank_b);
    for (int d = 0; d < out_rank; ++d) {
        const int ai = d - (out_rank - rank_a);
        const int bi = d - (out_rank - rank_b);
        const int64_t ad = ai < 0 ? 1 : a.shape()[ai];
        const int64_t bd = bi < 0 ? 1 : b.shape()[bi];
        if (ad != bd && ad != 1 && bd != 1) {
            std::ostringstream os;
            os << "broadcast_add: dimension mismatch (" << a.shape()
               << " vs " << b.shape() << ")";
            throw std::invalid_argument(os.str());
        }
    }
    return cpu_ops::tensor_broadcast_add_nd(a, b);
}

Tensor tensor_broadcast_add_backward_nd(const Tensor& g,
                                        const Shape& input_shape) {
    if (input_shape.rank() > g.shape().rank()) {
        throw std::invalid_argument("broadcast_add_backward: rank mismatch");
    }
    return cpu_ops::tensor_broadcast_add_backward_nd(g, input_shape);
}

Tensor tensor_matmul_nd(const Tensor& a, const Tensor& b) {
    validate_matmul(a, b);
    return cpu_ops::tensor_matmul_nd(a, b);
}

Tensor tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b) {
    validate_matmul_backward_a(g, b);
    return cpu_ops::tensor_matmul_backward_a_nd(g, b);
}

Tensor tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g) {
    validate_matmul_backward_b(a, g);
    return cpu_ops::tensor_matmul_backward_b_nd(a, g);
}

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
