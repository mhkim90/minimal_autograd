#pragma once
// tensor.h — thin CPU Tensor value type.
//
// Standalone public header:
//   * No Eigen include. The public Tensor type holds no Eigen matrix.
//   * No CUDA header / runtime symbol. CUDA storage is not supported in
//     this gate; Device::cuda() descriptors remain valid but Tensor
//     creation or to(cuda) fails clearly at runtime (see src/core/tensor.cpp).
//   * No autograd-graph state. Tensor is numerical storage only; Variable
//     (Phase 3) will own a Tensor plus the autograd node.
//
// Storage and per-Tensor logical metadata are hidden in the .cpp. The
// public handle holds a single std::shared_ptr<detail::TensorImpl>.
// Each TensorImpl owns a Shape + Device + shared_ptr<Storage>, where
// Storage is the library-owned float32 buffer. Ordinary Tensor copies
// share the TensorImpl; reshape() returns a new TensorImpl that
// references the same Storage (data is shared, logical metadata is
// independent); clone() deep-copies the Storage.
//
// Initial guarantees (per ARCHITECTURE_REFACTOR_PLAN.md §5.3):
//   * float32 only (host-pointer APIs take const float* / float*).
//   * dense, contiguous storage only; no strided view.
//   * Default-constructed Tensor is Shape{0} (rank-1, single zero dim),
//     CPU device, zero elements. Safe to query: shape(), device(),
//     elements(), and empty() are all well-defined and return
//     coherent values (shape().numel() == elements() == 0).
//   * Rank-0 factories (Tensor::empty(Shape{}), zeros(Shape{}),
//     ones(Shape{}), from_host(data, Shape{})) follow the existing
//     Shape contract: Shape{}.numel() == 1, so they produce a scalar
//     with exactly one element.
//
// Validation contract (errors are std::invalid_argument /
// std::runtime_error / std::overflow_error as documented):
//   * Factories with a zero-element shape always produce an empty
//     Tensor (elements() == 0); Tensor::zeros fills with zero,
//     Tensor::ones fills with one, and host null is accepted for
//     zero-element shapes.
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
//   * to(cuda) and any Tensor factory on Device::cuda() throw
//     std::runtime_error with a clear message in CPU-only builds.

#include "autograd/shape.h"
#include "autograd/device.h"

#include <cstddef>
#include <memory>

namespace ag {

namespace detail {
class TensorImpl;
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

    // Same-device transfer is a shallow share. CUDA target throws.
    Tensor to(Device target) const;

    // Independent deep copy of storage and metadata.
    Tensor clone() const;

    // Logical reshape. Shares the underlying float32 Storage; creates a
    // new TensorImpl with the requested Shape. The source Tensor's
    // shape and the shape of any ordinary alias of it are unchanged.
    Tensor reshape(const Shape& new_shape) const;

    // count must equal elements() exactly; null is rejected unless
    // count == 0. Throws std::runtime_error on mismatch or null.
    void copy_to_host(float* destination, std::size_t count) const;
    void copy_from_host(const float* source, std::size_t count);

private:
    std::shared_ptr<detail::TensorImpl> impl_;
};

}  // namespace ag
