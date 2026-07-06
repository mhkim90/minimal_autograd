#include "autograd/cuda_core.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ag {
namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

int blocks(std::size_t n) {
    return static_cast<int>((n + 255) / 256);
}

__global__ void add_kernel(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void scale_kernel(const float* a, float s, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s * a[i];
}

__global__ void matmul_kernel(const float* a, const float* b, float* out,
                              int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int p = 0; p < k; ++p) {
        acc += a[row + p * m] * b[p + col * k];
    }
    out[row + col * m] = acc;
}

__global__ void matmul_grad_a_kernel(float* da, const float* g, const float* b,
                                     int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || p >= k) return;
    float acc = 0.f;
    for (int col = 0; col < n; ++col) {
        acc += g[row + col * m] * b[p + col * k];
    }
    da[row + p * m] += acc;
}

__global__ void matmul_grad_b_kernel(float* db, const float* a, const float* g,
                                     int m, int n, int k) {
    int p = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= k || col >= n) return;
    float acc = 0.f;
    for (int row = 0; row < m; ++row) {
        acc += a[row + p * m] * g[row + col * m];
    }
    db[p + col * k] += acc;
}

__global__ void broadcast_add_kernel(const float* a, const float* b, float* out,
                                     int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i >= total) return;
    int col = i / rows;
    out[i] = a[i] + b[col];
}

__global__ void relu_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.f ? a[i] : 0.f;
}

__device__ float sigmoid_value(float x) {
    if (x >= 0.f) {
        return 1.f / (1.f + expf(-x));
    }
    float e = expf(x);
    return e / (1.f + e);
}

__global__ void sigmoid_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sigmoid_value(a[i]);
}

__global__ void tanh_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}

__global__ void exp_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

__global__ void log_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}

__global__ void sqrt_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(a[i]);
}

__global__ void silu_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * sigmoid_value(a[i]);
}

__global__ void softplus_kernel(const float* a, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void div_kernel(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

__global__ void axpy_kernel(float* dst, const float* x, float alpha, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * x[i];
}

__global__ void adam_step_kernel(float* p, const float* grad, float* m, float* v,
                                 float lr, float beta1, float beta2, float eps,
                                 float bias_correction1, float bias_correction2,
                                 std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

__global__ void mul_grad_kernel(float* da, float* db, const float* g,
                                const float* a, const float* b, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        da[i] += g[i] * b[i];
        db[i] += g[i] * a[i];
    }
}

__global__ void broadcast_add_grad_kernel(float* da, float* db, const float* g,
                                          int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i >= total) return;
    int col = i / rows;
    da[i] += g[i];
    atomicAdd(&db[col], g[i]);
}

__global__ void relu_grad_kernel(float* dx, const float* g, const float* x, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += x[i] > 0.f ? g[i] : 0.f;
}

__global__ void sigmoid_grad_kernel(float* dx, const float* g,
                                    const float* y, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] * y[i] * (1.f - y[i]);
}

__global__ void tanh_grad_kernel(float* dx, const float* g,
                                 const float* y, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] * (1.f - y[i] * y[i]);
}

__global__ void exp_grad_kernel(float* dx, const float* g,
                                const float* y, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] * y[i];
}

__global__ void log_grad_kernel(float* dx, const float* g,
                                const float* x, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] / x[i];
}

__global__ void sqrt_grad_kernel(float* dx, const float* g,
                                 const float* y, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] / (2.f * y[i]);
}

__global__ void silu_grad_kernel(float* dx, const float* g,
                                 const float* x, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid_value(x[i]);
        dx[i] += g[i] * (s + x[i] * s * (1.f - s));
    }
}

__global__ void softplus_grad_kernel(float* dx, const float* g,
                                     const float* x, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[i] * sigmoid_value(x[i]);
}

__global__ void sub_grad_kernel(float* da, float* db, const float* g, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        da[i] += g[i];
        db[i] -= g[i];
    }
}

__global__ void div_grad_kernel(float* da, float* db, const float* g,
                                const float* a, const float* b, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        da[i] += g[i] / b[i];
        db[i] -= g[i] * a[i] / (b[i] * b[i]);
    }
}

__global__ void sum_kernel(const float* x, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}

__global__ void sum_backward_kernel(float* dx, const float* g, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[0];
}

__global__ void col_slice_kernel(const float* x, float* out,
                                 int rows, int start, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * len;
    if (i >= total) return;
    int row = i % rows;
    int out_col = i / rows;
    out[i] = x[row + (start + out_col) * rows];
}

__global__ void col_slice_backward_kernel(float* dx, const float* g,
                                          int rows, int start, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * len;
    if (i >= total) return;
    int row = i % rows;
    int out_col = i / rows;
    dx[row + (start + out_col) * rows] += g[i];
}

__global__ void row_slice_kernel(const float* x, float* out,
                                 int rows, int cols, int start, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = len * cols;
    if (i >= total) return;
    int out_row = i % len;
    int col = i / len;
    out[i] = x[(start + out_row) + col * rows];
}

__global__ void row_slice_backward_kernel(float* dx, const float* g,
                                          int rows, int cols, int start, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = len * cols;
    if (i >= total) return;
    int out_row = i % len;
    int col = i / len;
    dx[(start + out_row) + col * rows] += g[i];
}

__global__ void softmax_kernel(const float* x, float* out, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float mx = x[r];
    for (int c = 1; c < cols; ++c) {
        float v = x[r + c * rows];
        if (v > mx) mx = v;
    }
    float denom = 0.f;
    for (int c = 0; c < cols; ++c) {
        denom += expf(x[r + c * rows] - mx);
    }
    for (int c = 0; c < cols; ++c) {
        out[r + c * rows] = expf(x[r + c * rows] - mx) / denom;
    }
}

__global__ void log_softmax_kernel(const float* x, float* out, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float mx = x[r];
    for (int c = 1; c < cols; ++c) {
        float v = x[r + c * rows];
        if (v > mx) mx = v;
    }
    float denom = 0.f;
    for (int c = 0; c < cols; ++c) {
        denom += expf(x[r + c * rows] - mx);
    }
    float logsum = mx + logf(denom);
    for (int c = 0; c < cols; ++c) {
        out[r + c * rows] = x[r + c * rows] - logsum;
    }
}

__global__ void softmax_backward_kernel(float* dx, const float* g,
                                        const float* s, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float gs = 0.f;
    for (int c = 0; c < cols; ++c) {
        int idx = r + c * rows;
        gs += g[idx] * s[idx];
    }
    for (int c = 0; c < cols; ++c) {
        int idx = r + c * rows;
        dx[idx] += s[idx] * (g[idx] - gs);
    }
}

__global__ void log_softmax_backward_kernel(float* dx, const float* g,
                                            const float* lsm, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float gsum = 0.f;
    for (int c = 0; c < cols; ++c) {
        gsum += g[r + c * rows];
    }
    for (int c = 0; c < cols; ++c) {
        int idx = r + c * rows;
        float sm = expf(lsm[idx]);
        dx[idx] += g[idx] - sm * gsum;
    }
}

__global__ void conv2d_forward_kernel(const float* input, const float* weight,
                                      const float* bias, float* out,
                                      int total, int N, int C, int H, int W,
                                      int out_ch, int kH, int kW,
                                      int stride, int pad, int oH, int oW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int spatial = oH * oW;
    int n = i % N;
    int out_flat = i / N;
    int oc = out_flat / spatial;
    int pos = out_flat - oc * spatial;
    int oh = pos / oW;
    int ow = pos - oh * oW;

    float acc = bias[oc];
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                int in_flat = c * H * W + ih * W + iw;
                int w_flat = (c * kH + kh) * kW + kw;
                acc += input[n + in_flat * N] * weight[oc + w_flat * out_ch];
            }
        }
    }
    out[i] = acc;
}

__global__ void conv2d_grad_input_kernel(float* dx, const float* grad_out,
                                         const float* weight,
                                         int total, int N, int C, int H, int W,
                                         int out_ch, int kH, int kW,
                                         int stride, int pad, int oH, int oW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int n = i % N;
    int flat = i / N;
    int c = flat / (H * W);
    int rem = flat - c * H * W;
    int h = rem / W;
    int w = rem - h * W;
    int spatial = oH * oW;

    float acc = 0.f;
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int kh = 0; kh < kH; ++kh) {
            int oh_num = h + pad - kh;
            if (oh_num < 0 || oh_num % stride != 0) continue;
            int oh = oh_num / stride;
            if (oh < 0 || oh >= oH) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int ow_num = w + pad - kw;
                if (ow_num < 0 || ow_num % stride != 0) continue;
                int ow = ow_num / stride;
                if (ow < 0 || ow >= oW) continue;
                int w_flat = (c * kH + kh) * kW + kw;
                int pos = oh * oW + ow;
                acc += weight[oc + w_flat * out_ch] *
                       grad_out[n + (oc * spatial + pos) * N];
            }
        }
    }
    dx[i] += acc;
}

__global__ void conv2d_grad_weight_kernel(float* dw, const float* grad_out,
                                          const float* input,
                                          int total, int N, int C, int H, int W,
                                          int out_ch, int kH, int kW,
                                          int stride, int pad, int oH, int oW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int oc = i % out_ch;
    int w_flat = i / out_ch;
    int c = w_flat / (kH * kW);
    int rem = w_flat - c * kH * kW;
    int kh = rem / kW;
    int kw = rem - kh * kW;
    int spatial = oH * oW;

    float acc = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < oH; ++oh) {
            int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < oW; ++ow) {
                int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                int in_flat = c * H * W + ih * W + iw;
                int pos = oh * oW + ow;
                acc += grad_out[n + (oc * spatial + pos) * N] *
                       input[n + in_flat * N];
            }
        }
    }
    dw[i] += acc;
}

__global__ void conv2d_grad_bias_kernel(float* db, const float* grad_out,
                                        int N, int out_ch, int spatial) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_ch) return;
    float acc = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int pos = 0; pos < spatial; ++pos) {
            acc += grad_out[n + (oc * spatial + pos) * N];
        }
    }
    db[oc] += acc;
}

__global__ void maxpool2d_forward_kernel(const float* input, float* out,
                                         float* mask, int total,
                                         int N, int C, int H, int W,
                                         int kH, int kW, int stride,
                                         int oH, int oW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int spatial = oH * oW;
    int ksz = kH * kW;
    int n = i % N;
    int out_flat = i / N;
    int c = out_flat / spatial;
    int pos = out_flat - c * spatial;
    int oh = pos / oW;
    int ow = pos - oh * oW;
    int col_idx = n * spatial + pos;

    int best_k = 0;
    int ih0 = oh * stride;
    int iw0 = ow * stride;
    float best_v = input[n + (c * H * W + ih0 * W + iw0) * N];
    for (int k = 1; k < ksz; ++k) {
        int kh = k / kW;
        int kw = k - kh * kW;
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        float v = input[n + (c * H * W + ih * W + iw) * N];
        if (v > best_v) {
            best_v = v;
            best_k = k;
        }
    }
    out[i] = best_v;
    mask[c * ksz + best_k + col_idx * (C * ksz)] = 1.f;
}

__global__ void maxpool2d_backward_kernel(float* dx, const float* grad_out,
                                          const float* mask, int total,
                                          int N, int C, int H, int W,
                                          int kH, int kW, int stride,
                                          int oH, int oW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int n = i % N;
    int flat = i / N;
    int c = flat / (H * W);
    int rem = flat - c * H * W;
    int h = rem / W;
    int w = rem - h * W;
    int spatial = oH * oW;
    int ksz = kH * kW;

    float acc = 0.f;
    for (int kh = 0; kh < kH; ++kh) {
        int oh_num = h - kh;
        if (oh_num < 0 || oh_num % stride != 0) continue;
        int oh = oh_num / stride;
        if (oh < 0 || oh >= oH) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int ow_num = w - kw;
            if (ow_num < 0 || ow_num % stride != 0) continue;
            int ow = ow_num / stride;
            if (ow < 0 || ow >= oW) continue;
            int pos = oh * oW + ow;
            int col_idx = n * spatial + pos;
            int mask_row = c * ksz + kh * kW + kw;
            acc += mask[mask_row + col_idx * (C * ksz)] *
                   grad_out[n + (c * spatial + pos) * N];
        }
    }
    dx[i] += acc;
}

int cuda_output_extent(int input, int kernel, int stride, int pad, const char* what) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || pad < 0) {
        throw std::runtime_error(std::string(what) + ": invalid kernel geometry");
    }
    const int span = input + 2 * pad - kernel;
    if (span < 0 || span % stride != 0) {
        throw std::runtime_error(std::string(what) + ": kernel/stride geometry mismatch");
    }
    return span / stride + 1;
}

void require_cuda(VarPtr v) {
    if (!v->is_cuda()) throw std::runtime_error("cuda op requires CUDA Var input");
}

void require_same_device(VarPtr a, VarPtr b) {
    if (a->cuda_device() != b->cuda_device()) {
        throw std::runtime_error("cuda op requires inputs on the same device");
    }
}

void require_same_shape(VarPtr a, VarPtr b) {
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw std::runtime_error("cuda op shape mismatch");
    }
    require_same_device(a, b);
}

void require_cuda_slice_range(const char* op, int size, int start, int len) {
    if (start < 0 || len <= 0 || start + len > size) {
        throw std::runtime_error(std::string(op) + ": slice out of range");
    }
}

void require_matmul_shape(VarPtr a, VarPtr b) {
    if (a->data.cols() != b->data.rows()) {
        throw std::runtime_error("cuda_matmul_op: shape mismatch");
    }
    require_same_device(a, b);
}

void require_broadcast_add_shape(VarPtr a, VarPtr b) {
    if (b->data.rows() != 1 || b->data.cols() != a->data.cols()) {
        throw std::runtime_error("cuda_broadcast_add_op: shape mismatch");
    }
    require_same_device(a, b);
}

void require_conv2d_shape(VarPtr input, VarPtr weight, VarPtr bias,
                          int N, int C, int H, int W, int kH, int kW) {
    if (input->data.rows() != N || input->data.cols() != C * H * W ||
        weight->data.cols() != C * kH * kW ||
        bias->data.rows() != 1 || bias->data.cols() != weight->data.rows()) {
        throw std::runtime_error("cuda_conv2d_op: shape mismatch");
    }
    require_same_device(input, weight);
    require_same_device(input, bias);
}

void require_maxpool2d_shape(VarPtr input, int N, int C, int H, int W) {
    if (input->data.rows() != N || input->data.cols() != C * H * W) {
        throw std::runtime_error("cuda_maxpool2d_op: shape mismatch");
    }
}

VarPtr make_cuda_like(VarPtr a) {
    auto out = Var::make(Mat::Zero(a->data.rows(), a->data.cols()))->cuda(a->cuda_device());
    out->set_shape(a->shape());
    return out;
}

void finish_kernel(const char* what) {
    check(cudaGetLastError(), what);
}

} // namespace

CudaRuntimeInfo cuda_runtime_info(int preferred_device) {
    CudaRuntimeInfo info;

    cudaError_t err = cudaDriverGetVersion(&info.driver_version);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDriverGetVersion: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaRuntimeGetVersion(&info.runtime_version);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaRuntimeGetVersion: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaGetDeviceCount(&info.device_count);
    if (err == cudaErrorNoDevice) {
        info.device_count = 0;
        info.status = "no CUDA devices visible";
        return info;
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount: ") +
                                 cudaGetErrorString(err));
    }
    if (info.device_count <= 0) {
        info.status = "no CUDA devices visible";
        return info;
    }
    if (preferred_device < 0 || preferred_device >= info.device_count) {
        throw std::runtime_error("cuda_runtime_info: preferred device out of range");
    }

    check(cudaSetDevice(preferred_device), "cudaSetDevice");
    check(cudaGetDevice(&info.current_device), "cudaGetDevice");

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, info.current_device), "cudaGetDeviceProperties");
    info.device.device = info.current_device;
    info.device.name = prop.name;
    info.device.compute_major = prop.major;
    info.device.compute_minor = prop.minor;
    info.device.total_global_mem = prop.totalGlobalMem;
    info.device.multiprocessor_count = prop.multiProcessorCount;
    info.status = "CUDA device available";
    return info;
}

std::string cuda_runtime_summary(const CudaRuntimeInfo& info) {
    std::ostringstream os;
    os << "CUDA runtime: compiled=yes"
       << ", driver=" << info.driver_version
       << ", runtime=" << info.runtime_version
       << ", device_count=" << info.device_count;
    if (info.has_device()) {
        os << ", current_device=" << info.current_device
           << ", name=\"" << info.device.name << "\""
           << ", compute=" << info.device.compute_major << "."
           << info.device.compute_minor
           << ", global_mem_mib=" << (info.device.total_global_mem / (1024 * 1024))
           << ", multiprocessors=" << info.device.multiprocessor_count;
    }
    if (!info.status.empty()) {
        os << ", status=\"" << info.status << "\"";
    }
    return os.str();
}

void cuda_alloc_floats(float** p, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)), "cudaMalloc");
}

void cuda_free_floats(float* p) {
    check(cudaFree(p), "cudaFree");
}

void cuda_copy_h2d(float* dst, const float* src, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

void cuda_copy_d2h(float* dst, const float* src, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void cuda_zero(float* dst, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaMemset(dst, 0, n * sizeof(float)), "cudaMemset");
}

void cuda_fill(float* dst, float value, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
    if (value == 0.f) {
        cuda_zero(dst, n, device);
        return;
    }
    Mat tmp = Mat::Constant(1, static_cast<int>(n), value);
    cuda_copy_h2d(dst, tmp.data(), n, device);
}

VarPtr cuda_add_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b); require_same_shape(a, b);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    add_kernel<<<blocks(n), 256>>>(a->cuda_data(), b->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_add_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        axpy_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(), 1.f, n);
        axpy_kernel<<<blocks(n), 256>>>(b->cuda_grad(), self->cuda_grad(), 1.f, n);
        finish_kernel("cuda_add_backward");
    };
    return out;
}

VarPtr cuda_mul_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b); require_same_shape(a, b);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    mul_kernel<<<blocks(n), 256>>>(a->cuda_data(), b->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_mul_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        mul_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), b->cuda_grad(),
                                            self->cuda_grad(),
                                            a->cuda_data(), b->cuda_data(), n);
        finish_kernel("cuda_mul_backward");
    };
    return out;
}

VarPtr cuda_matmul_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b);
    require_matmul_shape(a, b);
    const int m = a->data.rows();
    const int k = a->data.cols();
    const int n = b->data.cols();
    auto out = Var::make(Mat::Zero(m, n))->cuda(a->cuda_device());
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a->cuda_data(), b->cuda_data(), out->cuda_data(),
                                   m, n, k);
    finish_kernel("cuda_matmul_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), m, n, k, block]() {
        auto self = wp.lock();
        dim3 grid_a((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        dim3 grid_b((n + block.x - 1) / block.x, (k + block.y - 1) / block.y);
        matmul_grad_a_kernel<<<grid_a, block>>>(a->cuda_grad(), self->cuda_grad(),
                                                b->cuda_data(), m, n, k);
        matmul_grad_b_kernel<<<grid_b, block>>>(b->cuda_grad(), a->cuda_data(),
                                                self->cuda_grad(), m, n, k);
        finish_kernel("cuda_matmul_backward");
    };
    return out;
}

VarPtr cuda_broadcast_add_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b);
    require_broadcast_add_shape(a, b);
    const int rows = a->data.rows();
    const int cols = a->data.cols();
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    broadcast_add_kernel<<<blocks(n), 256>>>(a->cuda_data(), b->cuda_data(),
                                             out->cuda_data(), rows, cols);
    finish_kernel("cuda_broadcast_add_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), rows, cols, n]() {
        auto self = wp.lock();
        broadcast_add_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), b->cuda_grad(),
                                                      self->cuda_grad(), rows, cols);
        finish_kernel("cuda_broadcast_add_backward");
    };
    return out;
}

VarPtr cuda_scale_op(VarPtr a, float s) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    scale_kernel<<<blocks(n), 256>>>(a->cuda_data(), s, out->cuda_data(), n);
    finish_kernel("cuda_scale_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), s, n]() {
        auto self = wp.lock();
        axpy_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(), s, n);
        finish_kernel("cuda_scale_backward");
    };
    return out;
}

VarPtr cuda_relu_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    relu_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_relu_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        relu_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                             a->cuda_data(), n);
        finish_kernel("cuda_relu_backward");
    };
    return out;
}

VarPtr cuda_sigmoid_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    sigmoid_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_sigmoid_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        sigmoid_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                                self->cuda_data(), n);
        finish_kernel("cuda_sigmoid_backward");
    };
    return out;
}

VarPtr cuda_tanh_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    tanh_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_tanh_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        tanh_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                             self->cuda_data(), n);
        finish_kernel("cuda_tanh_backward");
    };
    return out;
}

VarPtr cuda_exp_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    exp_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_exp_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        exp_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                            self->cuda_data(), n);
        finish_kernel("cuda_exp_backward");
    };
    return out;
}

VarPtr cuda_log_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    log_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_log_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        log_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                            a->cuda_data(), n);
        finish_kernel("cuda_log_backward");
    };
    return out;
}

VarPtr cuda_sqrt_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    sqrt_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_sqrt_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        sqrt_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                             self->cuda_data(), n);
        finish_kernel("cuda_sqrt_backward");
    };
    return out;
}

VarPtr cuda_silu_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    silu_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_silu_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        silu_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                             a->cuda_data(), n);
        finish_kernel("cuda_silu_backward");
    };
    return out;
}

VarPtr cuda_softplus_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    softplus_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_softplus_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        softplus_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                                 a->cuda_data(), n);
        finish_kernel("cuda_softplus_backward");
    };
    return out;
}

VarPtr cuda_sub_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b); require_same_shape(a, b);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    sub_kernel<<<blocks(n), 256>>>(a->cuda_data(), b->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_sub_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        sub_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), b->cuda_grad(),
                                            self->cuda_grad(), n);
        finish_kernel("cuda_sub_backward");
    };
    return out;
}

VarPtr cuda_div_op(VarPtr a, VarPtr b) {
    require_cuda(a); require_cuda(b); require_same_shape(a, b);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    div_kernel<<<blocks(n), 256>>>(a->cuda_data(), b->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_div_op");
    out->parents = {a, b};
    out->back_fn = [a, b, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        div_grad_kernel<<<blocks(n), 256>>>(a->cuda_grad(), b->cuda_grad(),
                                            self->cuda_grad(),
                                            a->cuda_data(), b->cuda_data(), n);
        finish_kernel("cuda_div_backward");
    };
    return out;
}

VarPtr cuda_sum_op(VarPtr a) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = Var::make(Mat::Zero(1, 1))->cuda(a->cuda_device());
    cuda_zero(out->cuda_data(), 1, a->cuda_device());
    sum_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(), n);
    finish_kernel("cuda_sum_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        sum_backward_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(), n);
        finish_kernel("cuda_sum_backward");
    };
    return out;
}

VarPtr cuda_col_slice_op(VarPtr a, int start, int len) {
    require_cuda(a);
    const int rows = a->data.rows();
    const int cols = a->data.cols();
    require_cuda_slice_range("cuda_col_slice_op", cols, start, len);
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(len);
    auto out = Var::make(Mat::Zero(rows, len))->cuda(a->cuda_device());
    col_slice_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(),
                                         rows, start, len);
    finish_kernel("cuda_col_slice_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), rows, start, len, n]() {
        auto self = wp.lock();
        col_slice_backward_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                                      rows, start, len);
        finish_kernel("cuda_col_slice_backward");
    };
    return out;
}

VarPtr cuda_row_slice_op(VarPtr a, int start, int len) {
    require_cuda(a);
    const int rows = a->data.rows();
    const int cols = a->data.cols();
    require_cuda_slice_range("cuda_row_slice_op", rows, start, len);
    const std::size_t n = static_cast<std::size_t>(len) * static_cast<std::size_t>(cols);
    auto out = Var::make(Mat::Zero(len, cols))->cuda(a->cuda_device());
    row_slice_kernel<<<blocks(n), 256>>>(a->cuda_data(), out->cuda_data(),
                                         rows, cols, start, len);
    finish_kernel("cuda_row_slice_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), rows, cols, start, len, n]() {
        auto self = wp.lock();
        row_slice_backward_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(),
                                                      rows, cols, start, len);
        finish_kernel("cuda_row_slice_backward");
    };
    return out;
}

void cuda_sgd_step(Var& p, float lr) {
    if (!p.is_cuda()) return;
    const std::size_t n = static_cast<std::size_t>(p.data.size());
    axpy_kernel<<<blocks(n), 256>>>(p.cuda_data(), p.cuda_grad(), -lr, n);
    finish_kernel("cuda_sgd_step");
}

void cuda_adam_step(Var& p, float* m, float* v,
                    float lr, float beta1, float beta2, float eps,
                    float bias_correction1, float bias_correction2) {
    if (!p.is_cuda()) return;
    check(cudaSetDevice(p.cuda_device()), "cudaSetDevice");
    const std::size_t n = static_cast<std::size_t>(p.data.size());
    adam_step_kernel<<<blocks(n), 256>>>(p.cuda_data(), p.cuda_grad(), m, v,
                                         lr, beta1, beta2, eps,
                                         bias_correction1, bias_correction2, n);
    finish_kernel("cuda_adam_step");
}

VarPtr cuda_softmax_op(VarPtr a) {
    require_cuda(a);
    const int rows = a->data.rows();
    const int cols = a->data.cols();
    auto out = make_cuda_like(a);
    softmax_kernel<<<rows, 1>>>(a->cuda_data(), out->cuda_data(), rows, cols);
    finish_kernel("cuda_softmax_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), rows, cols]() {
        auto self = wp.lock();
        softmax_backward_kernel<<<rows, 1>>>(a->cuda_grad(), self->cuda_grad(),
                                             self->cuda_data(), rows, cols);
        finish_kernel("cuda_softmax_backward");
    };
    return out;
}

VarPtr cuda_log_softmax_op(VarPtr a) {
    require_cuda(a);
    const int rows = a->data.rows();
    const int cols = a->data.cols();
    auto out = make_cuda_like(a);
    log_softmax_kernel<<<rows, 1>>>(a->cuda_data(), out->cuda_data(), rows, cols);
    finish_kernel("cuda_log_softmax_op");
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), rows, cols]() {
        auto self = wp.lock();
        log_softmax_backward_kernel<<<rows, 1>>>(a->cuda_grad(), self->cuda_grad(),
                                                 self->cuda_data(), rows, cols);
        finish_kernel("cuda_log_softmax_backward");
    };
    return out;
}

VarPtr cuda_conv2d_op(VarPtr input, VarPtr weight, VarPtr bias,
                      int N, int C, int H, int W,
                      int kH, int kW, int stride, int pad) {
    require_cuda(input); require_cuda(weight); require_cuda(bias);
    require_conv2d_shape(input, weight, bias, N, C, H, W, kH, kW);
    const int out_ch = weight->data.rows();
    const int oH = cuda_output_extent(H, kH, stride, pad, "cuda_conv2d_op");
    const int oW = cuda_output_extent(W, kW, stride, pad, "cuda_conv2d_op");
    const int total = N * out_ch * oH * oW;
    auto out = Var::make(Mat::Zero(N, out_ch * oH * oW))->cuda(input->cuda_device());
    out->set_shape({N, out_ch, oH, oW});
    conv2d_forward_kernel<<<blocks(total), 256>>>(
        input->cuda_data(), weight->cuda_data(), bias->cuda_data(), out->cuda_data(),
        total, N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW);
    finish_kernel("cuda_conv2d_op");
    out->parents = {input, weight, bias};
    out->back_fn = [input, weight, bias, wp = std::weak_ptr<Var>(out),
                    N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW]() {
        auto self = wp.lock();
        const int input_total = N * C * H * W;
        const int weight_total = out_ch * C * kH * kW;
        const int spatial = oH * oW;
        conv2d_grad_input_kernel<<<blocks(input_total), 256>>>(
            input->cuda_grad(), self->cuda_grad(), weight->cuda_data(),
            input_total, N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW);
        finish_kernel("cuda_conv2d_grad_input");
        conv2d_grad_weight_kernel<<<blocks(weight_total), 256>>>(
            weight->cuda_grad(), self->cuda_grad(), input->cuda_data(),
            weight_total, N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW);
        finish_kernel("cuda_conv2d_grad_weight");
        conv2d_grad_bias_kernel<<<blocks(out_ch), 256>>>(
            bias->cuda_grad(), self->cuda_grad(), N, out_ch, spatial);
        finish_kernel("cuda_conv2d_grad_bias");
    };
    return out;
}

VarPtr cuda_maxpool2d_op(VarPtr input,
                         int N, int C, int H, int W,
                         int kH, int kW, int stride) {
    require_cuda(input);
    require_maxpool2d_shape(input, N, C, H, W);
    const int oH = cuda_output_extent(H, kH, stride, 0, "cuda_maxpool2d_op");
    const int oW = cuda_output_extent(W, kW, stride, 0, "cuda_maxpool2d_op");
    const int total = N * C * oH * oW;
    const int mask_rows = C * kH * kW;
    const int mask_cols = N * oH * oW;
    auto out = Var::make(Mat::Zero(N, C * oH * oW))->cuda(input->cuda_device());
    auto mask = Var::make(Mat::Zero(mask_rows, mask_cols))->cuda(input->cuda_device());
    out->set_shape({N, C, oH, oW});
    cuda_zero(mask->cuda_data(), static_cast<std::size_t>(mask->data.size()),
              input->cuda_device());
    maxpool2d_forward_kernel<<<blocks(total), 256>>>(
        input->cuda_data(), out->cuda_data(), mask->cuda_data(), total,
        N, C, H, W, kH, kW, stride, oH, oW);
    finish_kernel("cuda_maxpool2d_op");
    out->parents = {input};
    out->back_fn = [input, mask, wp = std::weak_ptr<Var>(out),
                    N, C, H, W, kH, kW, stride, oH, oW]() {
        auto self = wp.lock();
        const int input_total = N * C * H * W;
        maxpool2d_backward_kernel<<<blocks(input_total), 256>>>(
            input->cuda_grad(), self->cuda_grad(), mask->cuda_data(), input_total,
            N, C, H, W, kH, kW, stride, oH, oW);
        finish_kernel("cuda_maxpool2d_backward");
    };
    return out;
}

} // namespace ag
