#include "detail/tensor_cuda_ops.h"

#include "detail/tensor_storage.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

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
                const int col_idx =
                    (n * K + c * kH * kW + kh * kW + kw) * P + p;
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
                                       const Tensor& bias, int stride,
                                       int pad, Tensor& saved_col) {
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
                                              int kH, int kW, int stride,
                                              int pad) {
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

}  // namespace detail
}  // namespace ag
