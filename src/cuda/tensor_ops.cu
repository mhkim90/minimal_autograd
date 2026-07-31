#include "detail/tensor_cuda_ops.h"

#include "detail/constants.h"
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

__global__ void broadcast_scalar_kernel(const float* scalar, float* out,
                                            std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scalar[0];
}

__global__ void sum_kernel(const float* input, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, input[i]);
}

__global__ void conv2d_im2col_kernel(const float* input, float* col,
                                     int total, int N, int C, int H, int W,
                                     int kH, int kW, int stride, int pad,
                                     int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int P = oH * oW;
    const int K = C * kH * kW;
    const int p = i % P;
    const int k = (i / P) % K;
    const int n = i / (P * K);
    const int oh = p / oW;
    const int ow = p - oh * oW;
    const int c = k / (kH * kW);
    const int rem = k - c * kH * kW;
    const int kh = rem / kW;
    const int kw = rem - kh * kW;
    const int ih = oh * stride + kh - pad;
    const int iw = ow * stride + kw - pad;
    float value = 0.f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        value = input[((n * C + c) * H + ih) * W + iw];
    }
    col[i] = value;
}

__global__ void conv2d_forward_kernel(const float* input, const float* weight,
                                      const float* bias, float* out,
                                      int total, int N, int C, int H, int W,
                                      int OC, int kH, int kW, int stride,
                                      int pad, int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ow = i % oW;
    const int oh = (i / oW) % oH;
    const int oc = (i / (oH * oW)) % OC;
    const int n = i / (OC * oH * oW);
    float value = bias[oc];
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            const int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                const int in_idx = ((n * C + c) * H + ih) * W + iw;
                const int w_idx = ((oc * C + c) * kH + kh) * kW + kw;
                value += input[in_idx] * weight[w_idx];
            }
        }
    }
    out[i] = value;
}

__global__ void conv2d_backward_input_kernel(float* out, const float* g,
                                             const float* weight, int total,
                                             int N, int C, int H, int W,
                                             int OC, int kH, int kW,
                                             int stride, int pad,
                                             int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int w = i % W;
    const int h = (i / W) % H;
    const int c = (i / (H * W)) % C;
    const int n = i / (C * H * W);
    float value = 0.f;
    for (int oc = 0; oc < OC; ++oc) {
        for (int kh = 0; kh < kH; ++kh) {
            const int oh_num = h + pad - kh;
            if (oh_num < 0 || oh_num % stride != 0) continue;
            const int oh = oh_num / stride;
            if (oh < 0 || oh >= oH) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int ow_num = w + pad - kw;
                if (ow_num < 0 || ow_num % stride != 0) continue;
                const int ow = ow_num / stride;
                if (ow < 0 || ow >= oW) continue;
                const int w_idx = ((oc * C + c) * kH + kh) * kW + kw;
                const int g_idx = ((n * OC + oc) * oH + oh) * oW + ow;
                value += weight[w_idx] * g[g_idx];
            }
        }
    }
    out[i] = value;
}

__global__ void conv2d_backward_weight_kernel(float* out, const float* g,
                                              const float* col, int total,
                                              int N, int C, int K, int P,
                                              int OC, int kH, int kW,
                                              int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int kw = i % kW;
    const int kh = (i / kW) % kH;
    const int c = (i / (kH * kW)) % C;
    const int oc = i / (C * kH * kW);
    float value = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < oH; ++oh) {
            for (int ow = 0; ow < oW; ++ow) {
                const int p = oh * oW + ow;
                const int col_idx = (n * K + c * kH * kW + kh * kW + kw) * P + p;
                const int g_idx = ((n * OC + oc) * oH + oh) * oW + ow;
                value += col[col_idx] * g[g_idx];
            }
        }
    }
    out[i] = value;
}

__global__ void conv2d_backward_bias_kernel(float* out, const float* g,
                                            int total, int N, int OC,
                                            int oH, int oW) {
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= total) return;
    float value = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < oH; ++oh) {
            for (int ow = 0; ow < oW; ++ow) {
                value += g[((n * OC + oc) * oH + oh) * oW + ow];
            }
        }
    }
    out[oc] = value;
}

__global__ void maxpool2d_forward_kernel(const float* input, float* out,
                                         float* mask, int total,
                                         int N, int C, int H, int W,
                                         int kH, int kW, int stride,
                                         int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int ow = i % oW;
    const int oh = (i / oW) % oH;
    const int c = (i / (oH * oW)) % C;
    const int n = i / (C * oH * oW);
    int best_k = 0;
    float best = input[((n * C + c) * H + oh * stride) * W + ow * stride];
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            if (kh == 0 && kw == 0) continue;
            const int ih = oh * stride + kh;
            const int iw = ow * stride + kw;
            const float value = input[((n * C + c) * H + ih) * W + iw];
            if (value > best) {
                best = value;
                best_k = kh * kW + kw;
            }
        }
    }
    out[i] = best;
    mask[i] = static_cast<float>(best_k);
}

__global__ void maxpool2d_backward_kernel(float* out, const float* g,
                                          const float* mask, int total,
                                          int N, int C, int H, int W,
                                          int kH, int kW, int stride,
                                          int oH, int oW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int w = i % W;
    const int h = (i / W) % H;
    const int c = (i / (H * W)) % C;
    const int n = i / (C * H * W);
    float value = 0.f;
    for (int oh = 0; oh < oH; ++oh) {
        const int kh = h - oh * stride;
        if (kh < 0 || kh >= kH) continue;
        for (int ow = 0; ow < oW; ++ow) {
            const int kw = w - ow * stride;
            if (kw < 0 || kw >= kW) continue;
            const int out_idx = ((n * C + c) * oH + oh) * oW + ow;
            if (static_cast<int>(mask[out_idx]) == kh * kW + kw) {
                value += g[out_idx];
            }
        }
    }
    out[i] = value;
}

__global__ void dft2_last2_kernel(const float* real_in, const float* imag_in,
                                  float* real_out, float* imag_out,
                                  int total, int H, int W, bool inverse,
                                  float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int plane = H * W;
    const int batch = i / plane;
    const int output = i - batch * plane;
    const int kr = output / W;
    const int kc = output - kr * W;
    float sum_r = 0.f;
    float sum_i = 0.f;
    const float sign = inverse ? 1.f : -1.f;
    const float* rp = real_in + batch * plane;
    const float* ip = imag_in + batch * plane;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const float angle = 2.f * detail::kPi *
                (static_cast<float>(kr * r) / static_cast<float>(H) +
                 static_cast<float>(kc * c) / static_cast<float>(W));
            const float wr = cosf(angle);
            const float wi = sign * sinf(angle);
            const float xr = rp[r * W + c];
            const float xi = ip[r * W + c];
            sum_r += xr * wr - xi * wi;
            sum_i += xr * wi + xi * wr;
        }
    }
    real_out[i] = scale * sum_r;
    imag_out[i] = scale * sum_i;
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

Tensor unary_backward(const Tensor& g,
                          const Tensor& saved,
                          int op,
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
    scale_kernel<<<blocks(n), 256>>>(
        tensor_data(a), tensor_data(out), scalar, n);
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

Tensor cuda_tensor_sum(const Tensor& a) {
    Tensor out = Tensor::empty(Shape{}, a.device());
    set_device(a, "cuda_tensor_sum");
    check(cudaMemset(tensor_data(out), 0, sizeof(float)),
          "cudaMemset(cuda_tensor_sum)");
    if (a.elements() == 0) {
        return out;
    }
    sum_kernel<<<blocks(a.elements()), 256>>>(
        tensor_data(a), tensor_data(out), a.elements());
    finish_kernel("cuda_tensor_sum");
    return out;
}

Tensor cuda_tensor_broadcast_scalar(const Tensor& scalar,
                                    const Shape& target) {
    Tensor out = Tensor::empty(target, scalar.device());
    const std::size_t n = out.elements();
    if (n == 0) return out;
    set_device(scalar, "cuda_tensor_broadcast_scalar");
    broadcast_scalar_kernel<<<blocks(n), 256>>>(
        tensor_data(scalar), tensor_data(out), n);
    finish_kernel("cuda_tensor_broadcast_scalar");
    return out;
}

Tensor cuda_tensor_relu(const Tensor& a) {
    return unary(a, ReLU, "cuda_tensor_relu");
}

Tensor cuda_tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, ReLU,
                              "cuda_tensor_relu_backward");
}

Tensor cuda_tensor_sigmoid(const Tensor& a) {
    return unary(a, Sigmoid, "cuda_tensor_sigmoid");
}

Tensor cuda_tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Sigmoid,
                              "cuda_tensor_sigmoid_backward");
}

Tensor cuda_tensor_tanh(const Tensor& a) {
    return unary(a, Tanh, "cuda_tensor_tanh");
}

Tensor cuda_tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Tanh,
                              "cuda_tensor_tanh_backward");
}

Tensor cuda_tensor_exp(const Tensor& a) {
    return unary(a, Exp, "cuda_tensor_exp");
}

Tensor cuda_tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Exp,
                              "cuda_tensor_exp_backward");
}

Tensor cuda_tensor_log(const Tensor& a) {
    return unary(a, Log, "cuda_tensor_log");
}

Tensor cuda_tensor_log_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Log,
                              "cuda_tensor_log_backward");
}

Tensor cuda_tensor_sqrt(const Tensor& a) {
    return unary(a, Sqrt, "cuda_tensor_sqrt");
}

Tensor cuda_tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    return unary_backward(g, saved, Sqrt,
                              "cuda_tensor_sqrt_backward");
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

Tensor cuda_tensor_silu_backward(const Tensor& g,
                                 const Tensor& x,
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
    negate_kernel<<<blocks(n), 256>>>(
        tensor_data(g), tensor_data(out), n);
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

Tensor cuda_tensor_div_backward_b(const Tensor& g,
                                  const Tensor& a,
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

Tensor cuda_tensor_broadcast_add(const Tensor& a, const Tensor& b) {
    const Shape out_shape = broadcast_shape(a, b);
    Tensor out = Tensor::empty(out_shape, a.device());
    const std::size_t n = out.elements();
    if (n == 0) return out;

    set_device(a, "cuda_tensor_broadcast_add");
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
        tensor_data(a), tensor_data(b), tensor_data(out), n,
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

    set_device(g, "cuda_tensor_broadcast_add_backward");
    check(cudaMemset(tensor_data(out), 0, out.elements() * sizeof(float)),
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
        tensor_data(g), tensor_data(out), g.elements(),
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

    set_device(a, "cuda_tensor_sum_axes");
    check(cudaMemset(tensor_data(out), 0, out.elements() * sizeof(float)),
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
        tensor_data(a), tensor_data(out), a.elements(),
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

    set_device(g, "cuda_tensor_sum_axes_backward");
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
        tensor_data(g), tensor_data(out), out.elements(),
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
    set_device(a, "cuda_tensor_softmax");
    softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        tensor_data(a), tensor_data(out), tensor_data(saved_softmax), a.elements(),
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
    set_device(g, "cuda_tensor_softmax_backward");
    softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        tensor_data(g), tensor_data(saved_softmax), tensor_data(out), g.elements(),
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
    set_device(a, "cuda_tensor_log_softmax");
    log_softmax_forward_kernel<<<blocks(a.elements()), 256>>>(
        tensor_data(a), tensor_data(out), tensor_data(saved_log_softmax), a.elements(),
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
    set_device(g, "cuda_tensor_log_softmax_backward");
    log_softmax_backward_kernel<<<blocks(g.elements()), 256>>>(
        tensor_data(g), tensor_data(saved_log_softmax), tensor_data(out), g.elements(),
        g.shape()[axis], strides[axis]);
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

namespace {

int spatial_extent(int input, int kernel, int stride, int pad,
                   const char* what) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || pad < 0) {
        throw std::invalid_argument(std::string(what) +
                                    ": invalid kernel geometry");
    }
    const int span = input + 2 * pad - kernel;
    if (span < 0 || span % stride != 0) {
        throw std::invalid_argument(std::string(what) +
                                    ": output extent is not integral");
    }
    return span / stride + 1;
}

}  // namespace

Tensor cuda_tensor_conv2d_nchw_forward(const Tensor& input,
                                       const Tensor& weight,
                                       const Tensor& bias,
                                       int stride,
                                       int pad,
                                       Tensor& saved_col) {
    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int OC = static_cast<int>(weight.shape()[0]);
    const int kH = static_cast<int>(weight.shape()[2]);
    const int kW = static_cast<int>(weight.shape()[3]);
    const int oH = spatial_extent(H, kH, stride, pad, "cuda_conv2d");
    const int oW = spatial_extent(W, kW, stride, pad, "cuda_conv2d");
    const int K = C * kH * kW;
    const int P = oH * oW;
    saved_col = Tensor::empty(Shape{N, K, P}, input.device());
    Tensor out = Tensor::empty(Shape{N, OC, oH, oW}, input.device());
    if (out.elements() == 0) return out;
    set_device(input, "cuda_tensor_conv2d_nchw_forward");
    if (saved_col.elements() > 0) {
        conv2d_im2col_kernel<<<blocks(saved_col.elements()), 256>>>(
            tensor_data(input), tensor_data(saved_col),
            static_cast<int>(saved_col.elements()), N, C, H, W,
            kH, kW, stride, pad, oH, oW);
        finish_kernel("cuda_tensor_conv2d_nchw_im2col");
    }
    conv2d_forward_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(input), tensor_data(weight), tensor_data(bias),
        tensor_data(out), static_cast<int>(out.elements()), N, C, H, W,
        OC, kH, kW, stride, pad, oH, oW);
    finish_kernel("cuda_tensor_conv2d_nchw_forward");
    return out;
}

Tensor cuda_tensor_conv2d_nchw_backward_input(const Tensor& g,
                                              const Tensor& weight,
                                              int N, int C, int H, int W,
                                              int kH, int kW,
                                              int stride, int pad) {
    const int oH = spatial_extent(H, kH, stride, pad,
                                  "cuda_conv2d_backward_input");
    const int oW = spatial_extent(W, kW, stride, pad,
                                  "cuda_conv2d_backward_input");
    const int OC = static_cast<int>(weight.shape()[0]);
    Tensor out = Tensor::empty(Shape{N, C, H, W}, g.device());
    if (out.elements() == 0) return out;
    set_device(g, "cuda_tensor_conv2d_nchw_backward_input");
    conv2d_backward_input_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(out), tensor_data(g), tensor_data(weight),
        static_cast<int>(out.elements()), N, C, H, W, OC,
        kH, kW, stride, pad, oH, oW);
    finish_kernel("cuda_tensor_conv2d_nchw_backward_input");
    return out;
}

Tensor cuda_tensor_conv2d_nchw_backward_weight(const Tensor& g,
                                               const Tensor& col,
                                               const Shape& weight_shape) {
    const int N = static_cast<int>(col.shape()[0]);
    const int OC = static_cast<int>(weight_shape[0]);
    const int C = static_cast<int>(weight_shape[1]);
    const int kH = static_cast<int>(weight_shape[2]);
    const int kW = static_cast<int>(weight_shape[3]);
    const int K = C * kH * kW;
    const int oH = static_cast<int>(g.shape()[2]);
    const int oW = static_cast<int>(g.shape()[3]);
    const int P = oH * oW;
    Tensor out = Tensor::empty(weight_shape, g.device());
    if (out.elements() == 0) return out;
    set_device(g, "cuda_tensor_conv2d_nchw_backward_weight");
    conv2d_backward_weight_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(out), tensor_data(g), tensor_data(col),
        static_cast<int>(out.elements()), N, C, K, P, OC, kH, kW, oH, oW);
    finish_kernel("cuda_tensor_conv2d_nchw_backward_weight");
    return out;
}

Tensor cuda_tensor_conv2d_nchw_backward_bias(const Tensor& g) {
    const int N = static_cast<int>(g.shape()[0]);
    const int OC = static_cast<int>(g.shape()[1]);
    const int oH = static_cast<int>(g.shape()[2]);
    const int oW = static_cast<int>(g.shape()[3]);
    Tensor out = Tensor::empty(Shape{OC}, g.device());
    if (out.elements() == 0) return out;
    set_device(g, "cuda_tensor_conv2d_nchw_backward_bias");
    conv2d_backward_bias_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(out), tensor_data(g), OC, N, OC, oH, oW);
    finish_kernel("cuda_tensor_conv2d_nchw_backward_bias");
    return out;
}

Tensor cuda_tensor_maxpool2d_nchw_forward(const Tensor& input,
                                          int kH, int kW, int stride,
                                          Tensor& saved_mask) {
    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int oH = spatial_extent(H, kH, stride, 0, "cuda_max_pool2d");
    const int oW = spatial_extent(W, kW, stride, 0, "cuda_max_pool2d");
    Tensor out = Tensor::empty(Shape{N, C, oH, oW}, input.device());
    saved_mask = Tensor::empty(out.shape(), input.device());
    if (out.elements() == 0) return out;
    set_device(input, "cuda_tensor_maxpool2d_nchw_forward");
    maxpool2d_forward_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(input), tensor_data(out), tensor_data(saved_mask),
        static_cast<int>(out.elements()), N, C, H, W,
        kH, kW, stride, oH, oW);
    finish_kernel("cuda_tensor_maxpool2d_nchw_forward");
    return out;
}

Tensor cuda_tensor_maxpool2d_nchw_backward(const Tensor& g,
                                           const Tensor& mask,
                                           int N, int C, int H, int W,
                                           int kH, int kW, int stride) {
    const int oH = spatial_extent(H, kH, stride, 0,
                                  "cuda_max_pool2d_backward");
    const int oW = spatial_extent(W, kW, stride, 0,
                                  "cuda_max_pool2d_backward");
    Tensor out = Tensor::empty(Shape{N, C, H, W}, g.device());
    if (out.elements() == 0) return out;
    set_device(g, "cuda_tensor_maxpool2d_nchw_backward");
    maxpool2d_backward_kernel<<<blocks(out.elements()), 256>>>(
        tensor_data(out), tensor_data(g), tensor_data(mask),
        static_cast<int>(out.elements()), N, C, H, W,
        kH, kW, stride, oH, oW);
    finish_kernel("cuda_tensor_maxpool2d_nchw_backward");
    return out;
}

CudaTensorDFT2Result cuda_tensor_dft2_last2(const Tensor& real,
                                            const Tensor& imag,
                                            bool inverse,
                                            bool scale_output) {
    const int rank = static_cast<int>(real.shape().rank());
    const int H = static_cast<int>(real.shape()[rank - 2]);
    const int W = static_cast<int>(real.shape()[rank - 1]);
    CudaTensorDFT2Result out{
        Tensor::empty(real.shape(), real.device()),
        Tensor::empty(imag.shape(), imag.device())};
    if (out.real.elements() == 0) return out;
    set_device(real, "cuda_tensor_dft2_last2");
    const float scale = scale_output
        ? 1.f / static_cast<float>(H * W) : 1.f;
    dft2_last2_kernel<<<blocks(out.real.elements()), 256>>>(
        tensor_data(real), tensor_data(imag), tensor_data(out.real),
        tensor_data(out.imag), static_cast<int>(out.real.elements()),
        H, W, inverse, scale);
    finish_kernel("cuda_tensor_dft2_last2");
    return out;
}

// ── matmul (rank-N, batched over identical leading dims) ──────────────
//
// Each kernel runs as a flat 1-D launch that covers all batches
// simultaneously: the host launches blocks(total_elements) blocks
// (single grid-X dim), where total = batch * per_batch_count. The
// kernel derives batch = idx / per_batch_count and the local
// row/column from the flat remainder. Because storage is dense
// contiguous row-major, every batch slice is exactly per_batch_count
// contiguous floats so the kernel can compute each operand's base
// directly as `operand + batch * per_batch_count`; no offset table
// or device metadata is required. The flat launch avoids the CUDA
// grid-Z dimension cap (65535) so large batched matmuls do not
// silently truncate.

__global__ void matmul_forward_kernel(
    const float* a, const float* b, float* out,
    int64_t M, int64_t N, int64_t K, int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t per_batch = M * N;
    const int64_t batch = idx / per_batch;
    const int64_t rem = idx - batch * per_batch;
    const int64_t row = rem / N;
    const int64_t col = rem - row * N;
    const float* a_ptr = a + batch * (M * K);
    const float* b_ptr = b + batch * (K * N);
    float* out_ptr = out + batch * per_batch;
    float acc = 0.f;
    for (int64_t p = 0; p < K; ++p) {
        acc += a_ptr[row * K + p] * b_ptr[p * N + col];
    }
    out_ptr[row * N + col] = acc;
}

__global__ void matmul_backward_a_kernel(
    const float* g, const float* b, float* out,
    int64_t M, int64_t N, int64_t K, int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t per_batch = M * K;
    const int64_t batch = idx / per_batch;
    const int64_t rem = idx - batch * per_batch;
    const int64_t row = rem / K;
    const int64_t col = rem - row * K;
    const float* g_ptr = g + batch * (M * N);
    const float* b_ptr = b + batch * (K * N);
    float* out_ptr = out + batch * per_batch;
    float acc = 0.f;
    for (int64_t n = 0; n < N; ++n) {
        acc += g_ptr[row * N + n] * b_ptr[col * N + n];
    }
    out_ptr[row * K + col] = acc;
}

__global__ void matmul_backward_b_kernel(
    const float* a, const float* g, float* out,
    int64_t M, int64_t N, int64_t K, int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t per_batch = K * N;
    const int64_t batch = idx / per_batch;
    const int64_t rem = idx - batch * per_batch;
    const int64_t row = rem / N;
    const int64_t col = rem - row * N;
    const float* a_ptr = a + batch * (M * K);
    const float* g_ptr = g + batch * (M * N);
    float* out_ptr = out + batch * per_batch;
    float acc = 0.f;
    for (int64_t m = 0; m < M; ++m) {
        acc += a_ptr[m * K + row] * g_ptr[m * N + col];
    }
    out_ptr[row * N + col] = acc;
}

// Element-wise per-index in-place updates for optim::SGD and
// optim::Adam. They take raw device pointers (no Tensor API) so the
// OOP optimizer can mutate parameters without an extra host
// round-trip in the hot path.
__global__ void sgd_step_kernel(float* p, const float* grad,
                                float lr, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] -= lr * grad[i];
}

__global__ void adam_step_kernel(float* p, float* m, float* v,
                                 const float* grad,
                                 float lr, float beta1, float beta2,
                                 float eps,
                                 float bias_correction1, float bias_correction2,
                                 std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grad[i];
    const float mi = beta1 * m[i] + (1.f - beta1) * g;
    const float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    const float m_hat = mi / bias_correction1;
    const float v_hat = vi / bias_correction2;
    p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void validate_matmul_nd(const Tensor& a, const Tensor& b, const char* what) {
    if (a.device() != b.device()) {
        throw std::invalid_argument(std::string(what) + ": device mismatch");
    }
    const Shape& sa = a.shape();
    const Shape& sb = b.shape();
    const int rank_a = static_cast<int>(sa.rank());
    const int rank_b = static_cast<int>(sb.rank());
    if (rank_a < 2 || rank_b < 2) {
        std::ostringstream os;
        os << what << ": requires rank >= 2 (got " << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    if (rank_a != rank_b) {
        std::ostringstream os;
        os << what << ": rank mismatch (" << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t K = sa[rank_a - 1];
    const int64_t K2 = sb[rank_b - 2];
    if (K != K2) {
        std::ostringstream os;
        os << what << ": inner dimensions mismatch ("
           << sa << " vs " << sb << ")";
        throw std::invalid_argument(os.str());
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sb[d]) {
            std::ostringstream os;
            os << what << ": batch dim " << d << " mismatch ("
               << sa[d] << " vs " << sb[d] << ")";
            throw std::invalid_argument(os.str());
        }
    }
}

Tensor cuda_tensor_matmul_nd(const Tensor& a, const Tensor& b) {
    validate_matmul_nd(a, b, "matmul");
    const Shape& sa = a.shape();
    const int rank = static_cast<int>(sa.rank());
    const int64_t M = sa[rank - 2];
    const int64_t K = sa[rank - 1];
    const int64_t N = b.shape()[rank - 1];

    Dims out_sizes(sa.sizes.begin(), sa.sizes.end());
    out_sizes[rank - 2] = M;
    out_sizes[rank - 1] = N;
    Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (out.elements() == 0) return out;
    if (K == 0) return Tensor::zeros(out_shape, a.device());

    int64_t batch = 1;
    for (int d = 0; d < rank - 2; ++d) batch *= sa[d];

    set_device(a, "cuda_tensor_matmul_nd");
    const int64_t total = batch * M * N;
    matmul_forward_kernel<<<blocks(total), 256>>>(
        tensor_data(a), tensor_data(b), tensor_data(out),
        M, N, K, total);
    finish_kernel("cuda_tensor_matmul_nd");
    return out;
}

Tensor cuda_tensor_matmul_backward_a_nd(const Tensor& g, const Tensor& b) {
    const Shape& sg = g.shape();
    const Shape& sb = b.shape();
    const int rank_g = static_cast<int>(sg.rank());
    if (rank_g < 2) {
        throw std::invalid_argument(
            "matmul_backward_a: requires rank >= 2");
    }
    if (rank_g != static_cast<int>(sb.rank())) {
        throw std::invalid_argument("matmul_backward_a: rank mismatch");
    }
    const int64_t M = sg[rank_g - 2];
    const int64_t N = sg[rank_g - 1];
    const int64_t K = sb[rank_g - 2];
    if (N != sb[rank_g - 1]) {
        throw std::invalid_argument(
            "matmul_backward_a: g/b inner mismatch");
    }
    for (int d = 0; d < rank_g - 2; ++d) {
        if (sg[d] != sb[d]) {
            throw std::invalid_argument("matmul_backward_a: batch mismatch");
        }
    }
    Dims out_sizes(sg.sizes.begin(), sg.sizes.end());
    out_sizes[rank_g - 1] = K;
    Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, g.device());
    if (out.elements() == 0) return out;
    if (N == 0) return Tensor::zeros(out_shape, g.device());

    int64_t batch = 1;
    for (int d = 0; d < rank_g - 2; ++d) batch *= sg[d];

    set_device(g, "cuda_tensor_matmul_backward_a_nd");
    const int64_t total = batch * M * K;
    matmul_backward_a_kernel<<<blocks(total), 256>>>(
        tensor_data(g), tensor_data(b), tensor_data(out),
        M, N, K, total);
    finish_kernel("cuda_tensor_matmul_backward_a_nd");
    return out;
}

Tensor cuda_tensor_matmul_backward_b_nd(const Tensor& a, const Tensor& g) {
    const Shape& sa = a.shape();
    const Shape& sg = g.shape();
    const int rank_a = static_cast<int>(sa.rank());
    if (rank_a < 2) {
        throw std::invalid_argument(
            "matmul_backward_b: requires rank >= 2");
    }
    if (rank_a != static_cast<int>(sg.rank())) {
        throw std::invalid_argument("matmul_backward_b: rank mismatch");
    }
    const int64_t M = sa[rank_a - 2];
    const int64_t K = sa[rank_a - 1];
    const int64_t N = sg[rank_a - 1];
    if (sg[rank_a - 2] != M) {
        throw std::invalid_argument(
            "matmul_backward_b: a/g inner mismatch");
    }
    for (int d = 0; d < rank_a - 2; ++d) {
        if (sa[d] != sg[d]) {
            throw std::invalid_argument("matmul_backward_b: batch mismatch");
        }
    }
    Dims out_sizes(sa.sizes.begin(), sa.sizes.end());
    out_sizes[rank_a - 2] = K;
    out_sizes[rank_a - 1] = N;
    Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, g.device());
    if (out.elements() == 0) return out;
    if (M == 0) return Tensor::zeros(out_shape, g.device());

    int64_t batch = 1;
    for (int d = 0; d < rank_a - 2; ++d) batch *= sa[d];

    set_device(g, "cuda_tensor_matmul_backward_b_nd");
    const int64_t total = batch * K * N;
    matmul_backward_b_kernel<<<blocks(total), 256>>>(
        tensor_data(a), tensor_data(g), tensor_data(out),
        M, N, K, total);
    finish_kernel("cuda_tensor_matmul_backward_b_nd");
    return out;
}

void cuda_sgd_step(float* p, const float* grad, float lr,
                   std::size_t n, int device) {
    if (n == 0) return;
    check(cudaSetDevice(device), "cuda_sgd_step");
    sgd_step_kernel<<<blocks(n), 256>>>(p, grad, lr, n);
    finish_kernel("cuda_sgd_step");
}

void cuda_adam_step(float* p, float* m, float* v,
                    const float* grad,
                    float lr, float beta1, float beta2, float eps,
                    float bias_correction1, float bias_correction2,
                    std::size_t n, int device) {
    if (n == 0) return;
    check(cudaSetDevice(device), "cuda_adam_step");
    adam_step_kernel<<<blocks(n), 256>>>(
        p, m, v, grad, lr, beta1, beta2, eps,
        bias_correction1, bias_correction2, n);
    finish_kernel("cuda_adam_step");
}

}  // namespace detail
}  // namespace ag
