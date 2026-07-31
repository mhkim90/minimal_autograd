#include "autograd/tensor.h"
#include "autograd/core/ops.h"
#include "autograd/extension/cuda.h"
#include "autograd/extension/custom_op.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "CUDA custom consumer must not include Eigen"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

__global__ void affine_kernel(const float* input,
                              float* output,
                              std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = 2.f * input[index] + 1.f;
}

__global__ void scale_kernel(const float* input,
                             float* output,
                             std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = 2.f * input[index];
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-5f;
}

bool check_values(const ag::Tensor& tensor,
                  const std::vector<float>& expected) {
    std::vector<float> actual(expected.size());
    tensor.copy_to_host(actual.data(), actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!near(actual[i], expected[i])) return false;
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice || device_count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 0;
    }
    if (!cuda_ok(count_status, "cudaGetDeviceCount") ||
        !cuda_ok(cudaSetDevice(0), "cudaSetDevice")) {
        return 1;
    }

    const ag::Shape shape{2, 3};
    const std::vector<float> input_values = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const std::vector<float> output_values = {
        3.f, 5.f, 7.f, 9.f, 11.f, 13.f};
    const std::vector<float> gradient_values(6, 2.f);

    ag::Tensor input_tensor = ag::Tensor::from_host(
        input_values.data(), shape, ag::Device::cuda());
    ag::Variable input(input_tensor, true);

    ag::Tensor output_tensor = ag::Tensor::empty(shape, ag::Device::cuda());
    const ag::ConstCudaTensorView input_view = ag::cuda_view(input.value());
    const ag::CudaTensorView output_view = ag::cuda_view_mut(output_tensor);
    const int blocks = static_cast<int>((input_view.numel + 127) / 128);
    affine_kernel<<<blocks, 128>>>(input_view.data, output_view.data,
                                    input_view.numel);
    if (!cuda_ok(cudaDeviceSynchronize(), "affine kernel synchronization") ||
        !check_values(output_tensor, output_values)) {
        std::fprintf(stderr, "FAIL: CUDA custom operation forward\n");
        return 1;
    }

    ag::Variable output = ag::make_custom_variable(
        std::move(output_tensor), {input}, [](const ag::Tensor& output_gradient) {
            ag::Tensor input_gradient = ag::Tensor::empty(
                output_gradient.shape(), output_gradient.device());
            const ag::ConstCudaTensorView gradient_view =
                ag::cuda_view(output_gradient);
            const ag::CudaTensorView input_gradient_view =
                ag::cuda_view_mut(input_gradient);
            const int gradient_blocks = static_cast<int>(
                (gradient_view.numel + 127) / 128);
            scale_kernel<<<gradient_blocks, 128>>>(
                gradient_view.data, input_gradient_view.data,
                gradient_view.numel);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                throw std::runtime_error(
                    "CUDA custom operation backward synchronization failed");
            }
            return std::vector<ag::Tensor>{std::move(input_gradient)};
        });

    ag::sum(output).backward();
    if (!input.has_grad() || !check_values(input.grad(), gradient_values)) {
        std::fprintf(stderr, "FAIL: CUDA custom operation backward\n");
        return 1;
    }

    std::printf("OK: CUDA custom op forward/backward with borrowed views\n");
    return 0;
}
