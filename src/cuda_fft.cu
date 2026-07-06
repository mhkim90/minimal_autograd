#include "autograd/fft.h"
#include "autograd/cuda_core.h"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ag {
namespace detail {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kBlock = 256;
constexpr int kMaxCudaFftDim = 256;

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

int blocks(std::size_t n) {
    return static_cast<int>((n + kBlock - 1) / kBlock);
}

bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void validate_cuda_fft_shape(int rows, int cols) {
    if (!is_power_of_two(rows) || !is_power_of_two(cols)) {
        throw std::runtime_error("cuda fft2: dimensions must be powers of two");
    }
    if (rows > kMaxCudaFftDim || cols > kMaxCudaFftDim) {
        throw std::runtime_error("cuda fft2: dimensions exceed supported max 256");
    }
}

__global__ void row_dft_kernel(const float* in_real,
                               const float* in_imag,
                               float* tmp_real,
                               float* tmp_imag,
                               int rows,
                               int cols,
                               bool inverse) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int row = idx % rows;
    const int out_col = idx / rows;
    float sum_r = 0.f;
    float sum_i = 0.f;
    for (int c = 0; c < cols; ++c) {
        const float angle = 2.f * kPi * static_cast<float>(out_col * c) /
                            static_cast<float>(cols);
        const float wr = cosf(angle);
        const float wi = inverse ? sinf(angle) : -sinf(angle);
        const int in_idx = row + c * rows;
        const float xr = in_real[in_idx];
        const float xi = in_imag[in_idx];
        sum_r += xr * wr - xi * wi;
        sum_i += xr * wi + xi * wr;
    }
    tmp_real[idx] = sum_r;
    tmp_imag[idx] = sum_i;
}

__global__ void col_dft_kernel(const float* tmp_real,
                               const float* tmp_imag,
                               float* out_real,
                               float* out_imag,
                               int rows,
                               int cols,
                               bool inverse,
                               float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int out_row = idx % rows;
    const int col = idx / rows;
    float sum_r = 0.f;
    float sum_i = 0.f;
    for (int r = 0; r < rows; ++r) {
        const float angle = 2.f * kPi * static_cast<float>(out_row * r) /
                            static_cast<float>(rows);
        const float wr = cosf(angle);
        const float wi = inverse ? sinf(angle) : -sinf(angle);
        const int in_idx = r + col * rows;
        const float xr = tmp_real[in_idx];
        const float xi = tmp_imag[in_idx];
        sum_r += xr * wr - xi * wi;
        sum_i += xr * wi + xi * wr;
    }
    out_real[idx] = scale * sum_r;
    out_imag[idx] = scale * sum_i;
}

__global__ void add_pair_kernel(float* dst_real,
                                float* dst_imag,
                                const float* src_real,
                                const float* src_imag,
                                std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst_real[idx] += src_real[idx];
    dst_imag[idx] += src_imag[idx];
}

void run_cuda_dft(const float* in_real,
                  const float* in_imag,
                  float* out_real,
                  float* out_imag,
                  int rows,
                  int cols,
                  bool inverse,
                  float scale,
                  int device) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    float* tmp_real = nullptr;
    float* tmp_imag = nullptr;

    try {
        cuda_alloc_floats(&tmp_real, n, device);
        cuda_alloc_floats(&tmp_imag, n, device);
        check(cudaSetDevice(device), "cudaSetDevice");
        row_dft_kernel<<<blocks(n), kBlock>>>(in_real,
                                              in_imag,
                                              tmp_real,
                                              tmp_imag,
                                              rows,
                                              cols,
                                              inverse);
        check(cudaGetLastError(), "cuda fft2 row pass");
        col_dft_kernel<<<blocks(n), kBlock>>>(tmp_real,
                                              tmp_imag,
                                              out_real,
                                              out_imag,
                                              rows,
                                              cols,
                                              inverse,
                                              scale);
        check(cudaGetLastError(), "cuda fft2 col pass");
        check(cudaDeviceSynchronize(), "cuda fft2 synchronize");
    } catch (...) {
        cuda_free_floats(tmp_real);
        cuda_free_floats(tmp_imag);
        throw;
    }

    cuda_free_floats(tmp_real);
    cuda_free_floats(tmp_imag);
}

void accumulate_cuda_fft_adjoint(const float* grad_real,
                                 const float* grad_imag,
                                 float* input_grad_real,
                                 float* input_grad_imag,
                                 int rows,
                                 int cols,
                                 bool forward_was_inverse,
                                 int device) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    float* adj_real = nullptr;
    float* adj_imag = nullptr;

    try {
        cuda_alloc_floats(&adj_real, n, device);
        cuda_alloc_floats(&adj_imag, n, device);
        const bool adjoint_inverse = !forward_was_inverse;
        const float scale = forward_was_inverse ? 1.f / static_cast<float>(rows * cols) : 1.f;
        run_cuda_dft(grad_real,
                     grad_imag,
                     adj_real,
                     adj_imag,
                     rows,
                     cols,
                     adjoint_inverse,
                     scale,
                     device);
        add_pair_kernel<<<blocks(n), kBlock>>>(input_grad_real,
                                               input_grad_imag,
                                               adj_real,
                                               adj_imag,
                                               n);
        check(cudaGetLastError(), "cuda fft2 backward accumulate");
        check(cudaDeviceSynchronize(), "cuda fft2 backward synchronize");
    } catch (...) {
        cuda_free_floats(adj_real);
        cuda_free_floats(adj_imag);
        throw;
    }

    cuda_free_floats(adj_real);
    cuda_free_floats(adj_imag);
}

void attach_cuda_fft_backward(ComplexVar& out, const ComplexVar& in, bool inverse) {
    out.real->parents = {in.real, in.imag};
    out.imag->parents = {in.real, in.imag};
    const int rows = in.real->data.rows();
    const int cols = in.real->data.cols();
    const int device = in.real->cuda_device();
    auto input_real = in.real;
    auto input_imag = in.imag;

    out.real->back_fn = [input_real, input_imag, rows, cols, inverse, device,
                         wp = std::weak_ptr<Var>(out.real)]() {
        auto self = wp.lock();
        if (!self) return;
        float* zero = nullptr;
        try {
            const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
            cuda_alloc_floats(&zero, n, device);
            cuda_zero(zero, n, device);
            accumulate_cuda_fft_adjoint(self->cuda_grad(),
                                        zero,
                                        input_real->cuda_grad(),
                                        input_imag->cuda_grad(),
                                        rows,
                                        cols,
                                        inverse,
                                        device);
        } catch (...) {
            cuda_free_floats(zero);
            throw;
        }
        cuda_free_floats(zero);
    };
    out.imag->back_fn = [input_real, input_imag, rows, cols, inverse, device,
                         wp = std::weak_ptr<Var>(out.imag)]() {
        auto self = wp.lock();
        if (!self) return;
        float* zero = nullptr;
        try {
            const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
            cuda_alloc_floats(&zero, n, device);
            cuda_zero(zero, n, device);
            accumulate_cuda_fft_adjoint(zero,
                                        self->cuda_grad(),
                                        input_real->cuda_grad(),
                                        input_imag->cuda_grad(),
                                        rows,
                                        cols,
                                        inverse,
                                        device);
        } catch (...) {
            cuda_free_floats(zero);
            throw;
        }
        cuda_free_floats(zero);
    };
}

} // namespace

ComplexVar cuda_fft2_forward(const ComplexVar& z, bool inverse) {
    validate_complex_pair(z, "cuda_fft2_forward");
    if (!z.real->is_cuda() || !z.imag->is_cuda()) {
        throw std::runtime_error("cuda fft2: expected CUDA complex input");
    }

    const int rows = z.real->data.rows();
    const int cols = z.real->data.cols();
    validate_cuda_fft_shape(rows, cols);
    const int device = z.real->cuda_device();

    auto out_real = Var::make(Mat::Zero(rows, cols))->cuda(device);
    auto out_imag = Var::make(Mat::Zero(rows, cols))->cuda(device);
    out_real->set_shape(z.real->shape());
    out_imag->set_shape(z.imag->shape());

    const float scale = inverse ? 1.f / static_cast<float>(rows * cols) : 1.f;
    run_cuda_dft(z.real->cuda_data(),
                 z.imag->cuda_data(),
                 out_real->cuda_data(),
                 out_imag->cuda_data(),
                 rows,
                 cols,
                 inverse,
                 scale,
                 device);

    ComplexVar out = make_complex(std::move(out_real), std::move(out_imag));
    attach_cuda_fft_backward(out, z, inverse);
    return out;
}

} // namespace detail
} // namespace ag
