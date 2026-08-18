// CPU-only stub for the OOP Tensor's CUDA runtime helpers.
//
// This translation unit is selected only for CPU-only builds and provides
// bodies for the helpers declared in src/detail/tensor_cuda_runtime.h. CUDA
// builds instead select src/cuda/tensor_storage_runtime.cu through CMake.
//
// All throwing helpers raise std::runtime_error with a clear,
// build-aware message so any accidental attempt to construct or
// transfer a CUDA Tensor in a CPU-only build surfaces immediately
// at the point of the Storage operation rather than as a cryptic
// missing-symbol linker error. cuda_runtime_free is noexcept and
// silently does nothing on a CPU-only build (no CUDA buffers can
// exist on a CPU-only build because validate_device rejects every
// CUDA factory before any cuda_runtime_alloc is reached).

#include "detail/tensor_cuda_runtime.h"

#include <sstream>
#include <stdexcept>

namespace ag {
namespace detail {

void cuda_runtime_alloc(float** out, std::size_t n, int device) {
    (void)out; (void)n; (void)device;
    throw std::runtime_error(
        "Tensor storage: CUDA is not supported in this build "
        "(requested cuda_runtime_alloc)");
}

void cuda_runtime_free(float* p, int device) noexcept {
    (void)p; (void)device;
    // No-op. CPU-only builds never reach a CUDA Storage because
    // validate_device rejects CUDA factories, but the noexcept
    // contract is honored so the Storage destructor never throws.
}

void cuda_runtime_validate_device(int device) {
    std::ostringstream os;
    os << "Tensor storage: CUDA is not supported in this build "
          "(requested device cuda:" << device
       << " through cuda_runtime_validate_device)";
    throw std::runtime_error(os.str());
}

void cuda_runtime_copy_h2d(float* dst, const float* src,
                           std::size_t n, int device) {
    (void)dst; (void)src; (void)n; (void)device;
    throw std::runtime_error(
        "Tensor storage: CUDA is not supported in this build "
        "(requested cuda_runtime_copy_h2d)");
}

void cuda_runtime_copy_d2h(float* dst, const float* src,
                           std::size_t n, int device) {
    (void)dst; (void)src; (void)n; (void)device;
    throw std::runtime_error(
        "Tensor storage: CUDA is not supported in this build "
        "(requested cuda_runtime_copy_d2h)");
}

void cuda_runtime_copy_d2d(float* dst, const float* src,
                           std::size_t n, int device) {
    (void)dst; (void)src; (void)n; (void)device;
    throw std::runtime_error(
        "Tensor storage: CUDA is not supported in this build "
        "(requested cuda_runtime_copy_d2d)");
}

void cuda_runtime_copy_peer(float* dst, int dst_device,
                            const float* src, int src_device,
                            std::size_t n) {
    (void)dst; (void)dst_device; (void)src; (void)src_device; (void)n;
    throw std::runtime_error(
        "Tensor storage: CUDA is not supported in this build "
        "(requested cuda_runtime_copy_peer)");
}

}  // namespace detail
}  // namespace ag
