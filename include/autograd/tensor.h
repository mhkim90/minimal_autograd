#pragma once
// tensor.h — OOP Tensor value type.
//
// Standalone public header:
//   * No Eigen include. The public Tensor type holds no Eigen matrix.
//   * No CUDA header / runtime symbol. CUDA storage is reached only
//     through factories and `to(...)` calls; the borrowed CUDA view
//     surface lives behind an explicit opt-in extension header
//     (include/autograd/extension/cuda.h).
//   * No autograd-graph state. Tensor is numerical storage only;
//     Variable owns a Tensor plus the autograd node.
//
// Storage and per-Tensor logical metadata are hidden in the .cpp.
// Storage is dense, contiguous, and last-axis contiguous (canonical
// row-major): for shape (D0, ..., D{n-1}), stride[n-1] = 1 and
// stride[i] = stride[i+1] * shape[i+1]. The public handle holds a
// single std::shared_ptr<detail::TensorImpl>. Each TensorImpl owns
// a Shape + shared_ptr<Storage>, where Storage holds exactly one
// authoritative allocation (CPU std::vector<float> OR a single float*
// device pointer). There is no host/device mirror, no public raw
// allocation field, and no exposed allocator/free method.
// Ordinary Tensor copies share the TensorImpl; reshape() returns a new
// TensorImpl that references the same Storage; clone() deep-copies the
// Storage on the same device.
//
// Initial guarantees (per ARCHITECTURE_REFACTOR_PLAN.md §5.3):
//   * float32 only (host-pointer APIs take const float* / float*).
//   * dense, contiguous storage only; no strided view.
//   * Default-constructed Tensor is Shape{0} (rank-1, single zero dim),
//     CPU device, zero elements. Safe to query: shape(), device(),
//     elements(), and empty() are all well-defined and return
//     coherent values (shape().numel() == elements() == 0).
//   * Rank-0 factories follow the existing Shape contract:
//     Shape{}.numel() == 1, so they produce a scalar with exactly one
//     element.
//
// Validation contract (errors are std::invalid_argument /
// std::runtime_error / std::overflow_error as documented):
//   * Factories with a zero-element shape always produce an empty
//     Tensor (elements() == 0); host null is accepted for zero-element
//     shapes.
//   * Factories with a non-zero-element shape require shape.numel() to
//     be a valid int64 and to fit in size_t (no truncation); otherwise
//     std::overflow_error / std::invalid_argument is thrown.
//   * from_host(nullptr, shape) is allowed when shape.numel() == 0
//     and rejected when shape.numel() > 0.
//   * copy_to_host / copy_from_host require count == elements()
//     exactly; null pointer is rejected unless count == 0.
//   * reshape(new_shape) requires new_shape.numel() == elements() and
//     throws std::invalid_argument otherwise; it never reallocates and
//     never mutates the source Tensor's shape or the shape of any
//     other Tensor that shares the source's TensorImpl.
//   * to(target) is a shallow share when source and target devices
//     match; otherwise the function allocates a fresh Storage on the
//     target device and performs an explicit synchronous copy (H2D,
//     D2H, or D2D / peer-to-peer for cross-device transfers). No host
//     mirror is kept; the source-side Storage is untouched and aliases
//     continue to point at the original allocation.
//   * clone() is an independent deep copy on the same device as the
//     source. Aliases of the source do NOT observe any change through
//     clone(); they continue to share their original Storage.
//   * Repeated to(...) calls keep the source-side Storage identity
//     intact; each call produces an independent destination Storage.
//   * The borrowed CUDA view in include/autograd/extension/cuda.h is
//     the only public path to a raw device pointer; it is documented
//     in that header as non-owning and may not outlive the source.
#include "autograd/shape.h"
#include "autograd/device.h"

#include <cstddef>
#include <memory>

namespace ag {

class Tensor;

namespace detail {
class TensorImpl;
struct CudaTensorAccess;
}  // namespace detail

class Tensor {
public:
    // Default-constructed: Shape{0} (rank-1, dim 0), CPU, zero elements.
    // Safe to query: shape(), device(), elements(), empty() all return
    // coherent values.
    Tensor();

    static Tensor empty(const Shape& shape, Device device = Device::cpu());
    static Tensor zeros(const Shape& shape, Device device = Device::cpu());
    static Tensor ones(const Shape& shape, Device device = Device::cpu());
    static Tensor from_host(const float* data,
                            const Shape& shape,
                            Device target = Device::cpu());

    const Shape& shape() const noexcept;
    Device device() const noexcept;
    std::size_t elements() const noexcept;
    bool empty() const noexcept;

    // Same-device transfer is a shallow share. Cross-device transfer
    // allocates a fresh Storage on the target device and synchronously
    // copies (H2D / D2H / D2D or peer-to-peer for cross-device
    // transfers). The source-side Storage is untouched; aliases
    // continue to point at the original allocation.
    Tensor to(Device target) const;

    // Independent deep copy of storage and metadata. Aliases of the
    // source are unaffected; their borrowed views, if any, remain
    // valid because the source's Storage identity is preserved.
    Tensor clone() const;

    // Logical reshape. Shares the underlying float32 Storage; creates
    // a new TensorImpl with the requested Shape. The source Tensor's
    // shape and the shape of any ordinary alias of it are unchanged.
    Tensor reshape(const Shape& new_shape) const;

    // count must equal elements() exactly; null is rejected unless
    // count == 0. Throws std::runtime_error on mismatch or null.
    void copy_to_host(float* destination, std::size_t count) const;
    void copy_from_host(const float* source, std::size_t count);

private:
    std::shared_ptr<detail::TensorImpl> impl_;

    // Private bridge used by the opt-in CUDA extension.
    friend struct detail::CudaTensorAccess;
};

}  // namespace ag
