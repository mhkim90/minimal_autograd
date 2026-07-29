#pragma once
// Private CUDA-runtime-touching helpers for the OOP Tensor type.
//
// This header is the internal interface between src/core/tensor.cpp
// (which owns `ag::Tensor`'s authoritative Storage) and the optional
// CUDA runtime implementation in src/cuda/tensor_storage_runtime.cu.
// It is NOT installed. It declares:
//   * cuda_runtime_alloc:       allocate a fresh device buffer;
//   * cuda_runtime_free:        release a device buffer; noexcept;
//   * cuda_runtime_validate_device: confirm a device index refers
//                                 to an installed CUDA device;
//   * cuda_runtime_copy_h2d:    synchronous Host->Device copy;
//   * cuda_runtime_copy_d2h:    synchronous Device->Host copy;
//   * cuda_runtime_copy_d2d:    synchronous same-device Device->Device
//                                 copy;
//   * cuda_runtime_copy_peer:   synchronous Device->Device copy
//                                 across two distinct CUDA devices.
//
// In a CUDA-enabled build the definitions live in
// src/cuda/tensor_storage_runtime.cu, which calls the CUDA runtime
// directly.
// In a CPU-only build src/core/cuda_runtime_stubs.cpp provides bodies
// that always throw std::runtime_error with a clear message; this keeps
// the public `autograd/tensor.h` clean of CUDA includes while still
// allowing src/core/tensor.cpp to call into a uniform helper surface.
//
// All helpers except cuda_runtime_free throw std::runtime_error on
// failure. cuda_runtime_free is noexcept; on a CUDA build it sets the
// owning device and calls cudaFree, swallowing the diagnostic if the
// runtime reports an error so the destructor remains noexcept.

#include <cstddef>

namespace ag {
namespace detail {

void cuda_runtime_alloc(float** out, std::size_t n, int device);
void cuda_runtime_free(float* p, int device) noexcept;

// Confirms the device index refers to an installed CUDA device.
// Throws std::runtime_error otherwise. Called from every factory and
// transfer path so even zero-element Tensor construction surfaces an
// invalid device index clearly rather than silently building a CUDA
// Storage that the runtime cannot service.
void cuda_runtime_validate_device(int device);

void cuda_runtime_copy_h2d(float* dst, const float* src,
                           std::size_t n, int device);
void cuda_runtime_copy_d2h(float* dst, const float* src,
                           std::size_t n, int device);

// Same-device Device->Device copy: both buffers live on `device`.
// On CUDA the implementation sets the active device to `device`
// before issuing cudaMemcpyDeviceToDevice.
void cuda_runtime_copy_d2d(float* dst, const float* src,
                           std::size_t n, int device);

// Cross-device Device->Device copy: the source lives on `src_device`
// and the destination on `dst_device`. The CUDA implementation uses
// synchronous cudaMemcpyPeer.
void cuda_runtime_copy_peer(float* dst, int dst_device,
                            const float* src, int src_device,
                            std::size_t n);

}  // namespace detail
}  // namespace ag
