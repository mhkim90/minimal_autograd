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

} // namespace ag
