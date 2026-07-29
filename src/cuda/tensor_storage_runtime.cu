// src/cuda/tensor_storage_runtime.cu — CUDA-backend implementations of
// the OOP Tensor's runtime helpers.
//
// This translation unit is compiled only when AUTOGRAD_USE_CUDA is
// enabled. It provides the helpers declared in
// src/detail/tensor_cuda_runtime.h, in terms of cudaMalloc / cudaFree /
// cudaMemcpy / cudaMemcpyPeer directly (without going through
// src/cuda_core.cu, since this gate owns its own dispatch surface).
//
// The helpers are synchronous explicit copies with no host mirror:
//   * alloc / free use cudaMalloc / cudaFree on the requested device;
//     free sets the owning device active first and is noexcept;
//   * validate_device probes the requested device index through
//     cudaGetDeviceCount / cudaSetDevice and throws on failure;
//   * h2d / d2h are synchronous Host<->Device copies;
//   * d2d is a synchronous same-device Device->Device copy that
//     switches the active device to `device` first;
//   * peer is a synchronous Device->Device copy across two distinct
//     CUDA devices through cudaMemcpyPeer.
//
// There is no cached allocator, no stream pool, and no asynchronous
// path. Asynchronous execution and caching allocators are explicitly
// out of scope per ARCHITECTURE_REFACTOR_PLAN.md §8.

#include "detail/tensor_cuda_runtime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace ag {
namespace detail {

namespace {

// Throwing error helper used by every function except cuda_runtime_free.
inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

}  // namespace

void cuda_runtime_alloc(float** out, std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice(cuda_runtime_alloc)");
    check(cudaMalloc(reinterpret_cast<void**>(out),
                     n * sizeof(float)),
          "cudaMalloc(cuda_runtime_alloc)");
}

void cuda_runtime_free(float* p, int device) noexcept {
    if (p == nullptr) return;
    // Best-effort: switch the owning device active, then call
    // cudaFree. cudaFree returns the pointer to the driver; we
    // deliberately do not throw from this noexcept helper. If the
    // runtime reports an error we still drop the pointer through
    // cudaFree (cudaFree accepts any driver-allocated pointer and
    // the helper guarded nullptr above).
    cudaError_t set_err = cudaSetDevice(device);
    if (set_err != cudaSuccess) return;
    cudaError_t free_err = cudaFree(p);
    (void)free_err;
}

void cuda_runtime_validate_device(int device) {
    if (device < 0) {
        std::ostringstream os;
        os << "cuda_runtime_validate_device: negative device index ("
           << device << ")";
        throw std::runtime_error(os.str());
    }
    int count = 0;
    check(cudaGetDeviceCount(&count),
          "cudaGetDeviceCount(cuda_runtime_validate_device)");
    if (device >= count) {
        std::ostringstream os;
        os << "cuda_runtime_validate_device: device index " << device
           << " is out of range (visible device count: " << count << ")";
        throw std::runtime_error(os.str());
    }
    check(cudaSetDevice(device),
          "cudaSetDevice(cuda_runtime_validate_device)");
}

void cuda_runtime_copy_h2d(float* dst, const float* src,
                           std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice(cuda_runtime_copy_h2d)");
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D(cuda_runtime_copy_h2d)");
}

void cuda_runtime_copy_d2h(float* dst, const float* src,
                           std::size_t n, int device) {
    check(cudaSetDevice(device), "cudaSetDevice(cuda_runtime_copy_d2h)");
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H(cuda_runtime_copy_d2h)");
}

void cuda_runtime_copy_d2d(float* dst, const float* src,
                           std::size_t n, int device) {
    // Same-device Device->Device copy. The runtime requires the
    // active device to match the buffer's owning device.
    check(cudaSetDevice(device), "cudaSetDevice(cuda_runtime_copy_d2d)");
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice),
          "cudaMemcpy D2D(cuda_runtime_copy_d2d)");
}

void cuda_runtime_copy_peer(float* dst, int dst_device,
                            const float* src, int src_device,
                            std::size_t n) {
    if (dst_device == src_device) {
        cuda_runtime_copy_d2d(dst, src, n, dst_device);
        return;
    }
    check(cudaSetDevice(dst_device),
          "cudaSetDevice(dst_device) (cuda_runtime_copy_peer)");
    check(cudaMemcpyPeer(dst, dst_device, src, src_device,
                         n * sizeof(float)),
          "cudaMemcpyPeer(cuda_runtime_copy_peer)");
}

}  // namespace detail
}  // namespace ag
