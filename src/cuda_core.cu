#include "autograd/cuda_core.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
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

__global__ void axpy_kernel(float* dst, const float* x, float alpha, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * x[i];
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

__global__ void sum_kernel(const float* x, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}

__global__ void sum_backward_kernel(float* dx, const float* g, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += g[0];
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
    int oc = i % out_ch;
    int col_idx = i / out_ch;
    int n = col_idx / spatial;
    int pos = col_idx - n * spatial;
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
                int col_idx = n * spatial + oh * oW + ow;
                acc += weight[oc + w_flat * out_ch] *
                       grad_out[oc + col_idx * out_ch];
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
                int col_idx = n * spatial + oh * oW + ow;
                acc += grad_out[oc + col_idx * out_ch] *
                       input[n + in_flat * N];
            }
        }
    }
    dw[i] += acc;
}

__global__ void conv2d_grad_bias_kernel(float* db, const float* grad_out,
                                        int out_ch, int cols) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_ch) return;
    float acc = 0.f;
    for (int col = 0; col < cols; ++col) {
        acc += grad_out[oc + col * out_ch];
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
    int c = i % C;
    int col_idx = i / C;
    int n = col_idx / spatial;
    int pos = col_idx - n * spatial;
    int oh = pos / oW;
    int ow = pos - oh * oW;

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
            int col_idx = n * spatial + oh * oW + ow;
            int mask_row = c * ksz + kh * kW + kw;
            acc += mask[mask_row + col_idx * (C * ksz)] *
                   grad_out[c + col_idx * C];
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

void cuda_sgd_step(Var& p, float lr) {
    if (!p.is_cuda()) return;
    const std::size_t n = static_cast<std::size_t>(p.data.size());
    axpy_kernel<<<blocks(n), 256>>>(p.cuda_data(), p.cuda_grad(), -lr, n);
    finish_kernel("cuda_sgd_step");
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
        const int grad_cols = N * oH * oW;
        conv2d_grad_input_kernel<<<blocks(input_total), 256>>>(
            input->cuda_grad(), self->cuda_grad(), weight->cuda_data(),
            input_total, N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW);
        finish_kernel("cuda_conv2d_grad_input");
        conv2d_grad_weight_kernel<<<blocks(weight_total), 256>>>(
            weight->cuda_grad(), self->cuda_grad(), input->cuda_data(),
            weight_total, N, C, H, W, out_ch, kH, kW, stride, pad, oH, oW);
        finish_kernel("cuda_conv2d_grad_weight");
        conv2d_grad_bias_kernel<<<blocks(out_ch), 256>>>(
            bias->cuda_grad(), self->cuda_grad(), out_ch, grad_cols);
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
