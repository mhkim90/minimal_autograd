#pragma once
// Private storage declaration for the OOP Tensor type.
//
// This header is the bridge between the canonical `ag::Tensor` public
// surface and the small private helpers that need to see Tensor
// storage directly (currently: the borrowed CUDA view extension, and
// the math/optim path's device-aware boundaries). It lives under
// src/detail and is NOT installed.
//
// Storage holds exactly one authoritative allocation per Tensor:
//   * std::vector<float> on CPU, OR
//   * a single float* device pointer on CUDA.
// There is no host/device mirror, no copy-on-write, and no exposed
// mutator that would change which device holds the data. The
// authoritative numel is `count`; for CPU tensors `count == data.size()`
// by construction.
//
// The destructor is RAII and best-effort: for a CUDA allocation it
// calls ag::detail::cuda_runtime_free, which is `noexcept` and
// sets the owning device active before issuing cudaFree. On a
// CUDA-enabled build the helper releases the device buffer; any
// runtime error reported by cudaFree is swallowed because we cannot
// throw from a destructor. CPU-only builds use the no-op stub in
// src/core/cuda_runtime_stubs.cpp (no CUDA buffers can exist
// because validate_device rejects every CUDA factory). See
// src/cuda/tensor_storage_runtime.cu for the exact best-effort
// semantics.
//
// This file does NOT include any CUDA runtime header. The runtime-
// touching helpers (cuda_runtime_alloc, cuda_runtime_free,
// cuda_runtime_validate_device, cuda_runtime_copy_*, ...) live in
// src/cuda/tensor_storage_runtime.cu and have their declarations
// exposed through the separate private header
// src/detail/tensor_cuda_runtime.h.
//
#include "autograd/device.h"
#include "autograd/shape.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace ag {

class Tensor;

namespace detail {

struct Storage {
    Device device;
    std::vector<float> data;
    float* cuda_data = nullptr;
    std::size_t count = 0;

    Storage() : device(Device::cpu()), data(), cuda_data(nullptr), count(0) {}

    explicit Storage(Device d)
        : device(d), data(), cuda_data(nullptr), count(0) {}

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = delete;
    Storage& operator=(Storage&&) = delete;

    ~Storage();
};

class TensorImpl {
public:
    Shape shape;
    std::shared_ptr<Storage> storage;

    TensorImpl(Shape s, std::shared_ptr<Storage> st)
        : shape(std::move(s)), storage(std::move(st)) {}
};

// Private bridge between Tensor and the opt-in CUDA view extension.
// This header is not installed, so normal consumers cannot bypass
// cuda_view()/cuda_view_mut() to access storage.
struct CudaTensorAccess {
    static const float* cuda_data_const(const Tensor& tensor);
    static float* cuda_data_mutable(Tensor& tensor);
};

}  // namespace detail
}  // namespace ag
