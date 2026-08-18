#include "detail/tensor_cuda_ops.h"

#include "detail/tensor_storage.h"

#include <math_constants.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ag {
namespace {

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

inline int blocks(std::size_t n) {
    return static_cast<int>((n + 255) / 256);
}

inline void finish_kernel(const char* what) {
    check(cudaGetLastError(), what);
}

enum BinaryOp { Add, Mul, Sub, Div };
enum UnaryOp {
    ReLU,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
};

__device__ float sigmoid_value(float x) {
    if (x >= 0.f) return 1.f / (1.f + expf(-x));
    const float e = expf(x);
    return e / (1.f + e);
}

__device__ float unary_value(float x, int op) {
    switch (op) {
        case ReLU: return x > 0.f ? x : 0.f;
        case Sigmoid: return sigmoid_value(x);
        case Tanh: return tanhf(x);
        case Exp: return expf(x);
        case Log: return logf(x);
        case Sqrt: return sqrtf(x);
    }
    return 0.f;
}

__global__ void binary_forward_kernel(const float* a, const float* b,
                                      float* out, std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (op) {
        case Add: out[i] = a[i] + b[i]; break;
        case Mul: out[i] = a[i] * b[i]; break;
        case Sub: out[i] = a[i] - b[i]; break;
        case Div: out[i] = a[i] / b[i]; break;
    }
}

__global__ void scale_kernel(const float* a, float* out,
                             float scalar, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * scalar;
}

__global__ void fill_kernel(float* out, float value, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void unary_forward_kernel(const float* a, float* out,
                                     std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = unary_value(a[i], op);
}

__global__ void unary_backward_kernel(const float* g, const float* saved,
                                      float* out, std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = saved[i];
    float derivative = 0.f;
    switch (op) {
        case ReLU: derivative = s > 0.f ? 1.f : 0.f; break;
        case Sigmoid: derivative = s * (1.f - s); break;
        case Tanh: derivative = 1.f - s * s; break;
        case Exp: derivative = s; break;
        case Log: derivative = 1.f / s; break;
        case Sqrt: derivative = 1.f / (2.f * s); break;
    }
    out[i] = g[i] * derivative;
}

__global__ void silu_forward_kernel(const float* a, float* out,
                                    float* sigmoid_out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = sigmoid_value(a[i]);
    sigmoid_out[i] = s;
    out[i] = a[i] * s;
}

__global__ void silu_backward_kernel(const float* g, const float* x,
                                     const float* sigmoid, float* out,
                                     std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = sigmoid[i];
    out[i] = g[i] * (s + x[i] * s * (1.f - s));
}

__global__ void softplus_forward_kernel(const float* a, float* out,
                                         std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    out[i] = fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
}

__global__ void softplus_backward_kernel(const float* g, const float* x,
                                         float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g[i] * sigmoid_value(x[i]);
}

__global__ void div_backward_a_kernel(const float* g, const float* b,
                                      float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g[i] / b[i];
}

__global__ void div_backward_b_kernel(const float* g, const float* a,
                                      const float* b, float* out,
                                      std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -(g[i] * a[i]) / (b[i] * b[i]);
}

__global__ void negate_kernel(const float* g, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -g[i];
}

__global__ void softmax_forward_kernel(const float* input, float* out,
                                       float* saved, std::size_t n,
                                       int64_t axis_dim,
                                       int64_t axis_stride) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    const int64_t coord = (flat / axis_stride) % axis_dim;
    const int64_t base = static_cast<int64_t>(flat) - coord * axis_stride;
    float max_value = -CUDART_INF_F;
    for (int64_t k = 0; k < axis_dim; ++k) {
        max_value = fmaxf(max_value, input[base + k * axis_stride]);
    }
    float denom = 0.f;
    for (int64_t k = 0; k < axis_dim; ++k) {
        denom += expf(input[base + k * axis_stride] - max_value);
    }
    const float value = expf(input[flat] - max_value) / denom;
    out[flat] = value;
    saved[flat] = value;
}

__global__ void softmax_backward_kernel(const float* g, const float* saved,
                                        float* out, std::size_t n,
                                        int64_t axis_dim,
                                        int64_t axis_stride) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    const int64_t coord = (flat / axis_stride) % axis_dim;
    const int64_t base = static_cast<int64_t>(flat) - coord * axis_stride;
    float dot = 0.f;
    for (int64_t k = 0; k < axis_dim; ++k) {
        const int64_t offset = base + k * axis_stride;
        dot += g[offset] * saved[offset];
    }
    out[flat] = saved[flat] * (g[flat] - dot);
}

__global__ void log_softmax_forward_kernel(const float* input, float* out,
                                           float* saved, std::size_t n,
                                           int64_t axis_dim,
                                           int64_t axis_stride) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    const int64_t coord = (flat / axis_stride) % axis_dim;
    const int64_t base = static_cast<int64_t>(flat) - coord * axis_stride;
    float max_value = input[base];
    for (int64_t k = 1; k < axis_dim; ++k) {
        max_value = fmaxf(max_value, input[base + k * axis_stride]);
    }
    float denom = 0.f;
    for (int64_t k = 0; k < axis_dim; ++k) {
        denom += expf(input[base + k * axis_stride] - max_value);
    }
    const float value = input[flat] - (max_value + logf(denom));
    out[flat] = value;
    saved[flat] = value;
}

__global__ void log_softmax_backward_kernel(const float* g,
                                            const float* saved, float* out,
                                            std::size_t n, int64_t axis_dim,
                                            int64_t axis_stride) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    const int64_t coord = (flat / axis_stride) % axis_dim;
    const int64_t base = static_cast<int64_t>(flat) - coord * axis_stride;
    float row_sum = 0.f;
    for (int64_t k = 0; k < axis_dim; ++k) {
        row_sum += g[base + k * axis_stride];
    }
    out[flat] = g[flat] - expf(saved[flat]) * row_sum;
}

}  // namespace

namespace detail {
namespace {

inline void set_device(const Tensor& tensor, const char* what) {
    check(cudaSetDevice(tensor.device().index()), what);
}

inline const float* tensor_data(const Tensor& tensor) {
    return CudaTensorAccess::cuda_data_const(tensor);
}

inline float* tensor_data(Tensor& tensor) {
    return CudaTensorAccess::cuda_data_mutable(tensor);
}

Tensor binary(const Tensor& a, const Tensor& b, int op, const char* name) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    set_device(a, name);
    binary_forward_kernel<<<blocks(n), 256>>>(
        tensor_data(a), tensor_data(b), tensor_data(out), n, op);
    finish_kernel(name);
    return out;
}

Tensor unary(const Tensor& a, int op, const char* name) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    set_device(a, name);
    unary_forward_kernel<<<blocks(n), 256>>>(
        tensor_data(a), tensor_data(out), n, op);
    finish_kernel(name);
    return out;
}

Tensor unary_backward(const Tensor& g, const Tensor& saved, int op,
                      const char* name) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, name);
    unary_backward_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(saved), tensor_data(out), n, op);
    finish_kernel(name);
    return out;
}

}  // namespace

Tensor cuda_tensor_add(const Tensor& a, const Tensor& b) {
    return binary(a, b, Add, "cuda_tensor_add");
}

Tensor cuda_tensor_mul(const Tensor& a, const Tensor& b) {
    return binary(a, b, Mul, "cuda_tensor_mul");
}

Tensor cuda_tensor_scale(const Tensor& a, float scalar) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    set_device(a, "cuda_tensor_scale");
    scale_kernel<<<blocks(n), 256>>>(tensor_data(a), tensor_data(out), scalar, n);
    finish_kernel("cuda_tensor_scale");
    return out;
}

Tensor cuda_tensor_ones(const Shape& shape, Device device) {
    Tensor out = Tensor::empty(shape, device);
    const std::size_t n = out.elements();
    if (n == 0) return out;
    set_device(out, "cuda_tensor_ones");
    fill_kernel<<<blocks(n), 256>>>(tensor_data(out), 1.f, n);
    finish_kernel("cuda_tensor_ones");
    return out;
}

Tensor cuda_tensor_relu(const Tensor& a) {
    return unary(a, ReLU, "cuda_tensor_relu");
}

Tensor cuda_tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, ReLU, "cuda_tensor_relu_backward");
}

Tensor cuda_tensor_sigmoid(const Tensor& a) {
    return unary(a, Sigmoid, "cuda_tensor_sigmoid");
}

Tensor cuda_tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Sigmoid, "cuda_tensor_sigmoid_backward");
}

Tensor cuda_tensor_tanh(const Tensor& a) {
    return unary(a, Tanh, "cuda_tensor_tanh");
}

Tensor cuda_tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Tanh, "cuda_tensor_tanh_backward");
}

Tensor cuda_tensor_exp(const Tensor& a) {
    return unary(a, Exp, "cuda_tensor_exp");
}

Tensor cuda_tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Exp, "cuda_tensor_exp_backward");
}

Tensor cuda_tensor_log(const Tensor& a) {
    return unary(a, Log, "cuda_tensor_log");
}

Tensor cuda_tensor_log_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Log, "cuda_tensor_log_backward");
}

Tensor cuda_tensor_sqrt(const Tensor& a) {
    return unary(a, Sqrt, "cuda_tensor_sqrt");
}

Tensor cuda_tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Sqrt, "cuda_tensor_sqrt_backward");
}

Tensor cuda_tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    sigmoid_out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    set_device(a, "cuda_tensor_silu_forward");
    silu_forward_kernel<<<blocks(n), 256>>>(
        tensor_data(a), tensor_data(out), tensor_data(sigmoid_out), n);
    finish_kernel("cuda_tensor_silu_forward");
    return out;
}

Tensor cuda_tensor_silu_backward(const Tensor& g, const Tensor& x,
                                 const Tensor& sig) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, "cuda_tensor_silu_backward");
    silu_backward_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(x), tensor_data(sig), tensor_data(out), n);
    finish_kernel("cuda_tensor_silu_backward");
    return out;
}

Tensor cuda_tensor_softplus(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    set_device(a, "cuda_tensor_softplus");
    softplus_forward_kernel<<<blocks(n), 256>>>(
        tensor_data(a), tensor_data(out), n);
    finish_kernel("cuda_tensor_softplus");
    return out;
}

Tensor cuda_tensor_softplus_backward(const Tensor& g, const Tensor& x) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, "cuda_tensor_softplus_backward");
    softplus_backward_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(x), tensor_data(out), n);
    finish_kernel("cuda_tensor_softplus_backward");
    return out;
}

Tensor cuda_tensor_sub(const Tensor& a, const Tensor& b) {
    return binary(a, b, Sub, "cuda_tensor_sub");
}

Tensor cuda_tensor_sub_backward_b(const Tensor& g) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, "cuda_tensor_sub_backward_b");
    negate_kernel<<<blocks(n), 256>>>(tensor_data(g), tensor_data(out), n);
    finish_kernel("cuda_tensor_sub_backward_b");
    return out;
}

Tensor cuda_tensor_div(const Tensor& a, const Tensor& b) {
    return binary(a, b, Div, "cuda_tensor_div");
}

Tensor cuda_tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, "cuda_tensor_div_backward_a");
    div_backward_a_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(b), tensor_data(out), n);
    finish_kernel("cuda_tensor_div_backward_a");
    return out;
}

Tensor cuda_tensor_div_backward_b(const Tensor& g, const Tensor& a,
                                  const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    set_device(g, "cuda_tensor_div_backward_b");
    div_backward_b_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(a), tensor_data(b), tensor_data(out), n);
    finish_kernel("cuda_tensor_div_backward_b");
    return out;
}

Tensor cuda_tensor_softmax(const Tensor& a, int axis, Tensor& saved_softmax) {
    const int rank = static_cast<int>(a.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("softmax: axis out of range");
    }
    Tensor out = Tensor::empty(a.shape(), a.device());
    saved_softmax = Tensor::empty(a.shape(), a.device());
    if (a.elements() == 0) return out;
    const std::vector<int64_t> strides = [&a] {
        std::vector<int64_t> result(a.shape().rank(), 1);
        int64_t stride = 1;
        for (std::size_t i = a.shape().rank(); i-- > 0;) {
            result[i] = stride;
            stride *= a.shape().sizes[i];
        }
        return result;
    }();
    set_device(a, "cuda_tensor_softmax");
    softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        tensor_data(a), tensor_data(out), tensor_data(saved_softmax),
        a.elements(), a.shape()[axis], strides[axis]);
    finish_kernel("cuda_tensor_softmax");
    return out;
}

Tensor cuda_tensor_softmax_backward(const Tensor& g,
                                    const Tensor& saved_softmax, int axis) {
    const int rank = static_cast<int>(g.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("softmax_backward: axis out of range");
    }
    Tensor out = Tensor::empty(g.shape(), g.device());
    if (g.elements() == 0) return out;
    int64_t axis_stride = 1;
    for (std::size_t i = g.shape().rank(); i-- > static_cast<std::size_t>(axis + 1);) {
        axis_stride *= g.shape().sizes[i];
    }
    set_device(g, "cuda_tensor_softmax_backward");
    softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        tensor_data(g), tensor_data(saved_softmax), tensor_data(out),
        g.elements(), g.shape()[axis], axis_stride);
    finish_kernel("cuda_tensor_softmax_backward");
    return out;
}

Tensor cuda_tensor_log_softmax(const Tensor& a, int axis,
                               Tensor& saved_log_softmax) {
    const int rank = static_cast<int>(a.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("log_softmax: axis out of range");
    }
    Tensor out = Tensor::empty(a.shape(), a.device());
    saved_log_softmax = Tensor::empty(a.shape(), a.device());
    if (a.elements() == 0) return out;
    int64_t axis_stride = 1;
    for (std::size_t i = a.shape().rank(); i-- > static_cast<std::size_t>(axis + 1);) {
        axis_stride *= a.shape().sizes[i];
    }
    set_device(a, "cuda_tensor_log_softmax");
    log_softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        tensor_data(a), tensor_data(out), tensor_data(saved_log_softmax),
        a.elements(), a.shape()[axis], axis_stride);
    finish_kernel("cuda_tensor_log_softmax");
    return out;
}

Tensor cuda_tensor_log_softmax_backward(const Tensor& g,
                                        const Tensor& saved_log_softmax,
                                        int axis) {
    const int rank = static_cast<int>(g.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("log_softmax_backward: axis out of range");
    }
    Tensor out = Tensor::empty(g.shape(), g.device());
    if (g.elements() == 0) return out;
    int64_t axis_stride = 1;
    for (std::size_t i = g.shape().rank(); i-- > static_cast<std::size_t>(axis + 1);) {
        axis_stride *= g.shape().sizes[i];
    }
    set_device(g, "cuda_tensor_log_softmax_backward");
    log_softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        tensor_data(g), tensor_data(saved_log_softmax), tensor_data(out),
        g.elements(), g.shape()[axis], axis_stride);
    finish_kernel("cuda_tensor_log_softmax_backward");
    return out;
}

Tensor cuda_tensor_zeros(const Shape& shape, Device device) {
    Tensor out = Tensor::empty(shape, device);
    if (out.elements() == 0) return out;
    set_device(out, "cuda_tensor_zeros");
    check(cudaMemset(tensor_data(out), 0, out.elements() * sizeof(float)),
          "cuda_tensor_zeros");
    return out;
}

}  // namespace detail
}  // namespace ag
