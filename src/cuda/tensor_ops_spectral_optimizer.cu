#include "detail/tensor_cuda_ops.h"

#include "detail/constants.h"
#include "detail/tensor_storage.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

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

__global__ void sgd_step_kernel(float* p, const float* grad,
                                float lr, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] -= lr * grad[i];
}

__global__ void adam_step_kernel(float* p, float* m, float* v,
                                 const float* grad, float lr, float beta1,
                                 float beta2, float eps,
                                 float bias_correction1,
                                 float bias_correction2, std::size_t n) {
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

}  // namespace

CudaTensorDFT2Result cuda_tensor_dft2_last2(const Tensor& real,
                                            const Tensor& imag, bool inverse,
                                            bool scale_output) {
    const int rank = static_cast<int>(real.shape().rank());
    const int H = static_cast<int>(real.shape()[rank - 2]);
    const int W = static_cast<int>(real.shape()[rank - 1]);
    CudaTensorDFT2Result out{
        Tensor::empty(real.shape(), real.device()),
        Tensor::empty(imag.shape(), imag.device())};
    if (out.real.elements() == 0) return out;
    set_device(real, "cuda_tensor_dft2_last2");
    const float scale = scale_output ? 1.f / static_cast<float>(H * W) : 1.f;
    dft2_last2_kernel<<<blocks(out.real.elements()), 256>>>(
        tensor_data(real), tensor_data(imag), tensor_data(out.real),
        tensor_data(out.imag), static_cast<int>(out.real.elements()),
        H, W, inverse, scale);
    finish_kernel("cuda_tensor_dft2_last2");
    return out;
}

void cuda_sgd_step(float* p, const float* grad, float lr,
                   std::size_t n, int device) {
    if (n == 0) return;
    check(cudaSetDevice(device), "cuda_sgd_step");
    sgd_step_kernel<<<blocks(n), 256>>>(p, grad, lr, n);
    finish_kernel("cuda_sgd_step");
}

void cuda_adam_step(float* p, float* m, float* v, const float* grad,
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
