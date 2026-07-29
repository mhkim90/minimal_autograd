#include "detail/tensor_cuda_ops.h"

#include "detail/tensor_storage.h"

#include <math_constants.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
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

enum OopBinaryOp { OopAdd, OopMul, OopSub, OopDiv };
enum OopUnaryOp {
    OopReLU,
    OopSigmoid,
    OopTanh,
    OopExp,
    OopLog,
    OopSqrt,
};

__device__ float oop_sigmoid_value(float x) {
    if (x >= 0.f) return 1.f / (1.f + expf(-x));
    const float e = expf(x);
    return e / (1.f + e);
}

__device__ float oop_unary_value(float x, int op) {
    switch (op) {
        case OopReLU: return x > 0.f ? x : 0.f;
        case OopSigmoid: return oop_sigmoid_value(x);
        case OopTanh: return tanhf(x);
        case OopExp: return expf(x);
        case OopLog: return logf(x);
        case OopSqrt: return sqrtf(x);
    }
    return 0.f;
}

__global__ void oop_binary_forward_kernel(const float* a, const float* b,
                                          float* out, std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (op) {
        case OopAdd: out[i] = a[i] + b[i]; break;
        case OopMul: out[i] = a[i] * b[i]; break;
        case OopSub: out[i] = a[i] - b[i]; break;
        case OopDiv: out[i] = a[i] / b[i]; break;
    }
}

__global__ void oop_scale_kernel(const float* a, float* out,
                                 float scalar, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * scalar;
}

__global__ void oop_fill_kernel(float* out, float value, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void oop_unary_forward_kernel(const float* a, float* out,
                                         std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = oop_unary_value(a[i], op);
}

__global__ void oop_unary_backward_kernel(const float* g, const float* saved,
                                          float* out, std::size_t n, int op) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = saved[i];
    float derivative = 0.f;
    switch (op) {
        case OopReLU: derivative = s > 0.f ? 1.f : 0.f; break;
        case OopSigmoid: derivative = s * (1.f - s); break;
        case OopTanh: derivative = 1.f - s * s; break;
        case OopExp: derivative = s; break;
        case OopLog: derivative = 1.f / s; break;
        case OopSqrt: derivative = 1.f / (2.f * s); break;
    }
    out[i] = g[i] * derivative;
}

__global__ void oop_silu_forward_kernel(const float* a, float* out,
                                        float* sigmoid_out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = oop_sigmoid_value(a[i]);
    sigmoid_out[i] = s;
    out[i] = a[i] * s;
}

__global__ void oop_silu_backward_kernel(const float* g, const float* x,
                                         const float* sigmoid, float* out,
                                         std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = sigmoid[i];
    out[i] = g[i] * (s + x[i] * s * (1.f - s));
}

__global__ void oop_softplus_forward_kernel(const float* a, float* out,
                                            std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    out[i] = fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
}

__global__ void oop_softplus_backward_kernel(const float* g, const float* x,
                                             float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g[i] * oop_sigmoid_value(x[i]);
}

__global__ void oop_div_backward_a_kernel(const float* g, const float* b,
                                          float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g[i] / b[i];
}

__global__ void oop_div_backward_b_kernel(const float* g, const float* a,
                                          const float* b, float* out,
                                          std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -(g[i] * a[i]) / (b[i] * b[i]);
}

__global__ void oop_negate_kernel(const float* g, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -g[i];
}

__global__ void oop_broadcast_scalar_kernel(const float* scalar, float* out,
                                            std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scalar[0];
}

__global__ void oop_sum_kernel(const float* input, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, input[i]);
}

}  // namespace

namespace detail {
namespace {

inline void oop_set_device(const Tensor& tensor, const char* what) {
    check(cudaSetDevice(tensor.device().index()), what);
}

inline const float* oop_data(const Tensor& tensor) {
    return CudaTensorAccess::cuda_data_const(tensor);
}

inline float* oop_data(Tensor& tensor) {
    return CudaTensorAccess::cuda_data_mutable(tensor);
}

// Metadata kernels use the default stream. Destruction calls cudaFree,
// which waits for preceding default-stream use before releasing the buffer.
// Non-default streams are outside this backend's current contract.
class CudaMetadata {
public:
    CudaMetadata(const std::vector<int64_t>& values, int device,
                 const char* what)
        : device_(device) {
        if (values.empty()) return;
        check(cudaSetDevice(device_), what);
        check(cudaMalloc(reinterpret_cast<void**>(&data_),
                         values.size() * sizeof(int64_t)), what);
        try {
            check(cudaMemcpy(data_, values.data(),
                             values.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice), what);
        } catch (...) {
            cudaFree(data_);
            data_ = nullptr;
            throw;
        }
    }

    CudaMetadata(const CudaMetadata&) = delete;
    CudaMetadata& operator=(const CudaMetadata&) = delete;

    ~CudaMetadata() {
        if (data_ != nullptr) {
            cudaSetDevice(device_);
            cudaFree(data_);
        }
    }

    const int64_t* data() const { return data_; }

private:
    int64_t* data_ = nullptr;
    int device_ = 0;
};

std::vector<int64_t> row_major_strides(const Shape& shape) {
    std::vector<int64_t> strides(shape.rank(), 1);
    int64_t stride = 1;
    for (std::size_t i = shape.rank(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape.sizes[i];
    }
    return strides;
}

std::vector<int64_t> shape_sizes(const Shape& shape) {
    return std::vector<int64_t>(shape.sizes.begin(), shape.sizes.end());
}

Shape broadcast_shape(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("broadcast_add: device mismatch");
    }
    const int rank_a = static_cast<int>(a.shape().rank());
    const int rank_b = static_cast<int>(b.shape().rank());
    const int rank = std::max(rank_a, rank_b);
    Dims sizes(rank, 1);
    for (int d = 0; d < rank; ++d) {
        const int ai = d - (rank - rank_a);
        const int bi = d - (rank - rank_b);
        const int64_t ad = ai < 0 ? 1 : a.shape()[ai];
        const int64_t bd = bi < 0 ? 1 : b.shape()[bi];
        if (ad != bd && ad != 1 && bd != 1) {
            std::ostringstream os;
            os << "broadcast_add: dimension mismatch ("
               << a.shape() << " vs " << b.shape() << ")";
            throw std::invalid_argument(os.str());
        }
        sizes[d] = ad == 1 ? bd : ad;
    }
    return Shape(sizes);
}

Shape sum_axes_shape(const Shape& input,
                     const std::vector<int>& axes,
                     bool keep_dims) {
    std::vector<bool> reduced(input.rank(), false);
    for (int axis : axes) reduced[axis] = true;
    Dims sizes;
    sizes.reserve(keep_dims ? input.rank() : input.rank() - axes.size());
    for (std::size_t i = 0; i < input.rank(); ++i) {
        if (reduced[i]) {
            if (keep_dims) sizes.push_back(1);
        } else {
            sizes.push_back(input.sizes[i]);
        }
    }
    return Shape(sizes);
}

__global__ void broadcast_forward_kernel(const float* a, const float* b,
                                         float* out, std::size_t n,
                                         const int64_t* a_dims,
                                         const int64_t* a_strides, int rank_a,
                                         const int64_t* b_dims,
                                         const int64_t* b_strides, int rank_b,
                                         const int64_t* out_dims,
                                         const int64_t* out_strides,
                                         int rank_out) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    for (int d = 0; d < rank_out; ++d) {
        const int ai = d - (rank_out - rank_a);
        const int bi = d - (rank_out - rank_b);
        const int64_t coord = (flat / out_strides[d]) % out_dims[d];
        if (ai >= 0 && a_dims[ai] != 1) a_offset += coord * a_strides[ai];
        if (bi >= 0 && b_dims[bi] != 1) b_offset += coord * b_strides[bi];
    }
    out[flat] = a[a_offset] + b[b_offset];
}

__global__ void broadcast_backward_kernel(const float* g, float* out,
                                          std::size_t n,
                                          const int64_t* g_dims,
                                          const int64_t* g_strides,
                                          int rank_g,
                                          const int64_t* input_dims,
                                          const int64_t* input_strides,
                                          int rank_input) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    int64_t input_offset = 0;
    for (int d = 0; d < rank_g; ++d) {
        const int64_t dim = d + rank_input - rank_g;
        const int64_t coord = (flat / g_strides[d]) % g_dims[d];
        if (dim >= 0 && input_dims[dim] != 1) {
            input_offset += coord * input_strides[dim];
        }
    }
    atomicAdd(out + input_offset, g[flat]);
}

__global__ void sum_axes_forward_kernel(const float* input, float* out,
                                        std::size_t n,
                                        const int64_t* in_dims,
                                        const int64_t* in_strides,
                                        const int64_t* out_strides,
                                        const int64_t* reduced,
                                        int rank, bool keep_dims) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    int64_t out_offset = 0;
    int out_axis = 0;
    for (int d = 0; d < rank; ++d) {
        const int64_t coord = (flat / in_strides[d]) % in_dims[d];
        if (reduced[d] != 0) {
            if (keep_dims) ++out_axis;
        } else {
            out_offset += coord * out_strides[out_axis++];
        }
    }
    atomicAdd(out + out_offset, input[flat]);
}

__global__ void sum_axes_backward_kernel(const float* g, float* out,
                                         std::size_t n,
                                         const int64_t* in_dims,
                                         const int64_t* in_strides,
                                         const int64_t* g_strides,
                                         const int64_t* reduced,
                                         int rank, bool keep_dims) {
    const std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    int64_t g_offset = 0;
    int g_axis = 0;
    for (int d = 0; d < rank; ++d) {
        const int64_t coord = (flat / in_strides[d]) % in_dims[d];
        if (reduced[d] != 0) {
            if (keep_dims) ++g_axis;
        } else {
            g_offset += coord * g_strides[g_axis++];
        }
    }
    out[flat] = g[g_offset];
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

__global__ void log_softmax_backward_kernel(const float* g, const float* saved,
                                            float* out, std::size_t n,
                                            int64_t axis_dim,
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

Tensor oop_binary(const Tensor& a, const Tensor& b, int op, const char* name) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    oop_set_device(a, name);
    oop_binary_forward_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(b), oop_data(out), n, op);
    finish_kernel(name);
    return out;
}

Tensor oop_unary(const Tensor& a, int op, const char* name) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    oop_set_device(a, name);
    oop_unary_forward_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(out), n, op);
    finish_kernel(name);
    return out;
}

Tensor oop_unary_backward(const Tensor& g,
                          const Tensor& saved,
                          int op,
                          const char* name) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, name);
    oop_unary_backward_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(saved), oop_data(out), n, op);
    finish_kernel(name);
    return out;
}

}  // namespace

Tensor cuda_tensor_add(const Tensor& a, const Tensor& b) {
    return oop_binary(a, b, OopAdd, "cuda_tensor_add");
}

Tensor cuda_tensor_mul(const Tensor& a, const Tensor& b) {
    return oop_binary(a, b, OopMul, "cuda_tensor_mul");
}

Tensor cuda_tensor_scale(const Tensor& a, float scalar) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    oop_set_device(a, "cuda_tensor_scale");
    oop_scale_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(out), scalar, n);
    finish_kernel("cuda_tensor_scale");
    return out;
}

Tensor cuda_tensor_ones(const Shape& shape, Device device) {
    Tensor out = Tensor::empty(shape, device);
    const std::size_t n = out.elements();
    if (n == 0) return out;
    oop_set_device(out, "cuda_tensor_ones");
    oop_fill_kernel<<<blocks(n), 256>>>(oop_data(out), 1.f, n);
    finish_kernel("cuda_tensor_ones");
    return out;
}

Tensor cuda_tensor_sum(const Tensor& a) {
    Tensor out = Tensor::empty(Shape{}, a.device());
    oop_set_device(a, "cuda_tensor_sum");
    check(cudaMemset(oop_data(out), 0, sizeof(float)),
          "cudaMemset(cuda_tensor_sum)");
    if (a.elements() == 0) {
        return out;
    }
    oop_sum_kernel<<<blocks(a.elements()), 256>>>(
        oop_data(a), oop_data(out), a.elements());
    finish_kernel("cuda_tensor_sum");
    return out;
}

Tensor cuda_tensor_broadcast_scalar(const Tensor& scalar,
                                    const Shape& target) {
    Tensor out = Tensor::empty(target, scalar.device());
    const std::size_t n = out.elements();
    if (n == 0) return out;
    oop_set_device(scalar, "cuda_tensor_broadcast_scalar");
    oop_broadcast_scalar_kernel<<<blocks(n), 256>>>(
        oop_data(scalar), oop_data(out), n);
    finish_kernel("cuda_tensor_broadcast_scalar");
    return out;
}

Tensor cuda_tensor_relu(const Tensor& a) {
    return oop_unary(a, OopReLU, "cuda_tensor_relu");
}

Tensor cuda_tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopReLU,
                              "cuda_tensor_relu_backward");
}

Tensor cuda_tensor_sigmoid(const Tensor& a) {
    return oop_unary(a, OopSigmoid, "cuda_tensor_sigmoid");
}

Tensor cuda_tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopSigmoid,
                              "cuda_tensor_sigmoid_backward");
}

Tensor cuda_tensor_tanh(const Tensor& a) {
    return oop_unary(a, OopTanh, "cuda_tensor_tanh");
}

Tensor cuda_tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopTanh,
                              "cuda_tensor_tanh_backward");
}

Tensor cuda_tensor_exp(const Tensor& a) {
    return oop_unary(a, OopExp, "cuda_tensor_exp");
}

Tensor cuda_tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopExp,
                              "cuda_tensor_exp_backward");
}

Tensor cuda_tensor_log(const Tensor& a) {
    return oop_unary(a, OopLog, "cuda_tensor_log");
}

Tensor cuda_tensor_log_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopLog,
                              "cuda_tensor_log_backward");
}

Tensor cuda_tensor_sqrt(const Tensor& a) {
    return oop_unary(a, OopSqrt, "cuda_tensor_sqrt");
}

Tensor cuda_tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    return oop_unary_backward(g, saved, OopSqrt,
                              "cuda_tensor_sqrt_backward");
}

Tensor cuda_tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    sigmoid_out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    oop_set_device(a, "cuda_tensor_silu_forward");
    oop_silu_forward_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(out), oop_data(sigmoid_out), n);
    finish_kernel("cuda_tensor_silu_forward");
    return out;
}

Tensor cuda_tensor_silu_backward(const Tensor& g,
                                 const Tensor& x,
                                 const Tensor& sig) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, "cuda_tensor_silu_backward");
    oop_silu_backward_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(x), oop_data(sig), oop_data(out), n);
    finish_kernel("cuda_tensor_silu_backward");
    return out;
}

Tensor cuda_tensor_softplus(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    oop_set_device(a, "cuda_tensor_softplus");
    oop_softplus_forward_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(out), n);
    finish_kernel("cuda_tensor_softplus");
    return out;
}

Tensor cuda_tensor_softplus_backward(const Tensor& g, const Tensor& x) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, "cuda_tensor_softplus_backward");
    oop_softplus_backward_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(x), oop_data(out), n);
    finish_kernel("cuda_tensor_softplus_backward");
    return out;
}

Tensor cuda_tensor_sub(const Tensor& a, const Tensor& b) {
    return oop_binary(a, b, OopSub, "cuda_tensor_sub");
}

Tensor cuda_tensor_sub_backward_b(const Tensor& g) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, "cuda_tensor_sub_backward_b");
    oop_negate_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(out), n);
    finish_kernel("cuda_tensor_sub_backward_b");
    return out;
}

Tensor cuda_tensor_div(const Tensor& a, const Tensor& b) {
    return oop_binary(a, b, OopDiv, "cuda_tensor_div");
}

Tensor cuda_tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, "cuda_tensor_div_backward_a");
    oop_div_backward_a_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(b), oop_data(out), n);
    finish_kernel("cuda_tensor_div_backward_a");
    return out;
}

Tensor cuda_tensor_div_backward_b(const Tensor& g,
                                  const Tensor& a,
                                  const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    oop_set_device(g, "cuda_tensor_div_backward_b");
    oop_div_backward_b_kernel<<<blocks(n), 256>>>(
        oop_data(g), oop_data(a), oop_data(b), oop_data(out), n);
    finish_kernel("cuda_tensor_div_backward_b");
    return out;
}

Tensor cuda_tensor_broadcast_add(const Tensor& a, const Tensor& b) {
    const Shape out_shape = broadcast_shape(a, b);
    Tensor out = Tensor::empty(out_shape, a.device());
    const std::size_t n = out.elements();
    if (n == 0) return out;

    oop_set_device(a, "cuda_tensor_broadcast_add");
    const std::vector<int64_t> a_dims = shape_sizes(a.shape());
    const std::vector<int64_t> b_dims = shape_sizes(b.shape());
    const std::vector<int64_t> out_dims = shape_sizes(out_shape);
    const std::vector<int64_t> a_strides = row_major_strides(a.shape());
    const std::vector<int64_t> b_strides = row_major_strides(b.shape());
    const std::vector<int64_t> out_strides = row_major_strides(out_shape);
    CudaMetadata a_dims_d(a_dims, a.device().index(),
                          "cuda_tensor_broadcast_add metadata");
    CudaMetadata b_dims_d(b_dims, a.device().index(),
                          "cuda_tensor_broadcast_add metadata");
    CudaMetadata out_dims_d(out_dims, a.device().index(),
                            "cuda_tensor_broadcast_add metadata");
    CudaMetadata a_strides_d(a_strides, a.device().index(),
                             "cuda_tensor_broadcast_add metadata");
    CudaMetadata b_strides_d(b_strides, a.device().index(),
                             "cuda_tensor_broadcast_add metadata");
    CudaMetadata out_strides_d(out_strides, a.device().index(),
                               "cuda_tensor_broadcast_add metadata");
    broadcast_forward_kernel<<<blocks(n), 256>>>(
        oop_data(a), oop_data(b), oop_data(out), n,
        a_dims_d.data(), a_strides_d.data(), static_cast<int>(a_dims.size()),
        b_dims_d.data(), b_strides_d.data(), static_cast<int>(b_dims.size()),
        out_dims_d.data(), out_strides_d.data(),
        static_cast<int>(out_dims.size()));
    finish_kernel("cuda_tensor_broadcast_add");
    return out;
}

Tensor cuda_tensor_broadcast_add_backward(const Tensor& g,
                                          const Shape& input_shape) {
    const int rank_g = static_cast<int>(g.shape().rank());
    const int rank_input = static_cast<int>(input_shape.rank());
    if (rank_input > rank_g) {
        throw std::invalid_argument("broadcast_add_backward: rank mismatch");
    }
    Tensor out = Tensor::empty(input_shape, g.device());
    if (out.elements() == 0) return out;

    oop_set_device(g, "cuda_tensor_broadcast_add_backward");
    check(cudaMemset(oop_data(out), 0, out.elements() * sizeof(float)),
          "cudaMemset(cuda_tensor_broadcast_add_backward)");
    if (g.elements() == 0) return out;
    const std::vector<int64_t> g_dims = shape_sizes(g.shape());
    const std::vector<int64_t> g_strides = row_major_strides(g.shape());
    const std::vector<int64_t> input_dims = shape_sizes(input_shape);
    const std::vector<int64_t> input_strides = row_major_strides(input_shape);
    CudaMetadata g_dims_d(g_dims, g.device().index(),
                          "cuda_tensor_broadcast_add_backward metadata");
    CudaMetadata g_strides_d(g_strides, g.device().index(),
                             "cuda_tensor_broadcast_add_backward metadata");
    CudaMetadata input_dims_d(input_dims, g.device().index(),
                              "cuda_tensor_broadcast_add_backward metadata");
    CudaMetadata input_strides_d(input_strides, g.device().index(),
                                 "cuda_tensor_broadcast_add_backward metadata");
    broadcast_backward_kernel<<<blocks(g.elements()), 256>>>(
        oop_data(g), oop_data(out), g.elements(),
        g_dims_d.data(), g_strides_d.data(), rank_g,
        input_dims_d.data(), input_strides_d.data(), rank_input);
    finish_kernel("cuda_tensor_broadcast_add_backward");
    return out;
}

Tensor cuda_tensor_sum_axes(const Tensor& a,
                            const std::vector<int>& axes,
                            bool keep_dims) {
    const Shape out_shape = sum_axes_shape(a.shape(), axes, keep_dims);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (out.elements() == 0) return out;

    oop_set_device(a, "cuda_tensor_sum_axes");
    check(cudaMemset(oop_data(out), 0, out.elements() * sizeof(float)),
          "cudaMemset(cuda_tensor_sum_axes)");
    if (a.elements() == 0) return out;
    const std::vector<int64_t> in_dims = shape_sizes(a.shape());
    const std::vector<int64_t> in_strides = row_major_strides(a.shape());
    const std::vector<int64_t> out_strides = row_major_strides(out_shape);
    std::vector<int64_t> reduced(a.shape().rank(), 0);
    for (int axis : axes) reduced[axis] = 1;
    CudaMetadata in_dims_d(in_dims, a.device().index(),
                          "cuda_tensor_sum_axes metadata");
    CudaMetadata in_strides_d(in_strides, a.device().index(),
                              "cuda_tensor_sum_axes metadata");
    CudaMetadata out_strides_d(out_strides, a.device().index(),
                               "cuda_tensor_sum_axes metadata");
    CudaMetadata reduced_d(reduced, a.device().index(),
                           "cuda_tensor_sum_axes metadata");
    sum_axes_forward_kernel<<<blocks(a.elements()), 256>>>(
        oop_data(a), oop_data(out), a.elements(),
        in_dims_d.data(), in_strides_d.data(), out_strides_d.data(),
        reduced_d.data(), static_cast<int>(in_dims.size()), keep_dims);
    finish_kernel("cuda_tensor_sum_axes");
    return out;
}

Tensor cuda_tensor_sum_axes_backward(const Tensor& g,
                                     const Shape& input_shape,
                                     const std::vector<int>& axes,
                                     bool keep_dims) {
    Tensor out = Tensor::empty(input_shape, g.device());
    if (out.elements() == 0) return out;

    oop_set_device(g, "cuda_tensor_sum_axes_backward");
    const std::vector<int64_t> in_dims = shape_sizes(input_shape);
    const std::vector<int64_t> in_strides = row_major_strides(input_shape);
    const std::vector<int64_t> g_strides = row_major_strides(g.shape());
    std::vector<int64_t> reduced(input_shape.rank(), 0);
    for (int axis : axes) reduced[axis] = 1;
    CudaMetadata in_dims_d(in_dims, g.device().index(),
                          "cuda_tensor_sum_axes_backward metadata");
    CudaMetadata in_strides_d(in_strides, g.device().index(),
                              "cuda_tensor_sum_axes_backward metadata");
    CudaMetadata g_strides_d(g_strides, g.device().index(),
                             "cuda_tensor_sum_axes_backward metadata");
    CudaMetadata reduced_d(reduced, g.device().index(),
                           "cuda_tensor_sum_axes_backward metadata");
    sum_axes_backward_kernel<<<blocks(out.elements()), 256>>>(
        oop_data(g), oop_data(out), out.elements(),
        in_dims_d.data(), in_strides_d.data(), g_strides_d.data(),
        reduced_d.data(), static_cast<int>(in_dims.size()), keep_dims);
    finish_kernel("cuda_tensor_sum_axes_backward");
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
    const std::vector<int64_t> strides = row_major_strides(a.shape());
    oop_set_device(a, "cuda_tensor_softmax");
    softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        oop_data(a), oop_data(out), oop_data(saved_softmax), a.elements(),
        a.shape()[axis], strides[axis]);
    finish_kernel("cuda_tensor_softmax");
    return out;
}

Tensor cuda_tensor_softmax_backward(const Tensor& g,
                                    const Tensor& saved_softmax,
                                    int axis) {
    const int rank = static_cast<int>(g.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("softmax_backward: axis out of range");
    }
    Tensor out = Tensor::empty(g.shape(), g.device());
    if (g.elements() == 0) return out;
    const std::vector<int64_t> strides = row_major_strides(g.shape());
    oop_set_device(g, "cuda_tensor_softmax_backward");
    softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        oop_data(g), oop_data(saved_softmax), oop_data(out), g.elements(),
        g.shape()[axis], strides[axis]);
    finish_kernel("cuda_tensor_softmax_backward");
    return out;
}

Tensor cuda_tensor_log_softmax(const Tensor& a,
                               int axis,
                               Tensor& saved_log_softmax) {
    const int rank = static_cast<int>(a.shape().rank());
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("log_softmax: axis out of range");
    }
    Tensor out = Tensor::empty(a.shape(), a.device());
    saved_log_softmax = Tensor::empty(a.shape(), a.device());
    if (a.elements() == 0) return out;
    const std::vector<int64_t> strides = row_major_strides(a.shape());
    oop_set_device(a, "cuda_tensor_log_softmax");
    log_softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        oop_data(a), oop_data(out), oop_data(saved_log_softmax), a.elements(),
        a.shape()[axis], strides[axis]);
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
    const std::vector<int64_t> strides = row_major_strides(g.shape());
    oop_set_device(g, "cuda_tensor_log_softmax_backward");
    log_softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        oop_data(g), oop_data(saved_log_softmax), oop_data(out), g.elements(),
        g.shape()[axis], strides[axis]);
    finish_kernel("cuda_tensor_log_softmax_backward");
    return out;
}

}  // namespace detail
}  // namespace ag
