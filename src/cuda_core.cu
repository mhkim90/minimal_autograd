#include "autograd/cuda_core.h"

#include <cuda_runtime.h>

#include <cassert>
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

void require_cuda(VarPtr v) {
    if (!v->is_cuda()) throw std::runtime_error("cuda op requires CUDA Var input");
}

void require_same_shape(VarPtr a, VarPtr b) {
    assert(a->data.rows() == b->data.rows());
    assert(a->data.cols() == b->data.cols());
    assert(a->cuda_device() == b->cuda_device());
}

VarPtr make_cuda_like(VarPtr a) {
    auto out = Var::make(Mat::Zero(a->data.rows(), a->data.cols()))->cuda(a->cuda_device());
    out->set_shape(a->shape());
    return out;
}

void finish_kernel(const char* what) {
    check(cudaGetLastError(), what);
    check(cudaDeviceSynchronize(), what);
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
    out->sync_data_from_cuda();
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
    out->sync_data_from_cuda();
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

VarPtr cuda_scale_op(VarPtr a, float s) {
    require_cuda(a);
    const std::size_t n = static_cast<std::size_t>(a->data.size());
    auto out = make_cuda_like(a);
    scale_kernel<<<blocks(n), 256>>>(a->cuda_data(), s, out->cuda_data(), n);
    finish_kernel("cuda_scale_op");
    out->sync_data_from_cuda();
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
    out->sync_data_from_cuda();
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
    out->sync_data_from_cuda();
    out->parents = {a};
    out->back_fn = [a, wp = std::weak_ptr<Var>(out), n]() {
        auto self = wp.lock();
        sum_backward_kernel<<<blocks(n), 256>>>(a->cuda_grad(), self->cuda_grad(), n);
        finish_kernel("cuda_sum_backward");
    };
    return out;
}

} // namespace ag
