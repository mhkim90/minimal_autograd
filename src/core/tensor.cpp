// src/core/tensor.cpp — thin CPU Tensor implementation.
//
// Hidden in this translation unit:
//   * detail::Storage — the library-owned float32 buffer plus its
//     device. One Storage per allocation; shared between Tensors that
//     alias each other (ordinary copies, reshape views).
//   * detail::TensorImpl — the per-Tensor logical metadata (Shape +
//     shared_ptr<Storage>). Ordinary copies share the TensorImpl;
//     reshape() creates a new TensorImpl that shares the Storage
//     but holds an independent Shape; clone() creates a new Storage
//     and a new TensorImpl.
//
// CUDA is intentionally not supported in this gate. Tensor creation
// or to(cuda) throws std::runtime_error with a message that names the
// requested device and the build configuration.

#include "autograd/tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {
namespace detail {

// Storage — the authoritative float32 buffer for a Tensor.
//
// Owns one std::vector<float> plus its device. The vector's size is
// the actual number of float32 elements; it is not derived from the
// logical Shape — different Tensors sharing the same Storage may
// report different Shapes (e.g. after reshape).
class Storage {
public:
    Device device;
    std::vector<float> data;

    explicit Storage(Device d) : device(d) {}
};

// TensorImpl — the per-Tensor logical view onto a Storage.
//
// Two Tensors that share a TensorImpl have the same Shape and Device
// and the same Storage. Two Tensors that share only the Storage (e.g.
// after reshape) have independent Shape and Device values that are
// still equal at the moment of the view, but the view is free to
// change its Shape without affecting the source.
class TensorImpl {
public:
    Shape shape;
    std::shared_ptr<Storage> storage;

    TensorImpl(Shape s, std::shared_ptr<Storage> st)
        : shape(std::move(s)), storage(std::move(st)) {}
};

}  // namespace detail

// ── Helpers ──────────────────────────────────────────────────────────────

namespace {

// Convert an int64_t numel to a size_t allocation length with a
// portable overflow / range check. Used by every factory that
// allocates Storage::data.
inline std::size_t safe_numel_to_size(const char* what, std::int64_t n) {
    if (n < 0) {
        std::ostringstream os;
        os << what << ": negative numel (" << n << ")";
        throw std::invalid_argument(os.str());
    }
    constexpr std::uint64_t kSizeMax =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (static_cast<std::uint64_t>(n) > kSizeMax) {
        std::ostringstream os;
        os << what << ": numel " << n << " exceeds size_t range";
        throw std::overflow_error(os.str());
    }
    return static_cast<std::size_t>(n);
}

inline void reject_cuda(const char* what, Device d) {
    if (d.is_cuda()) {
        std::ostringstream os;
        os << what << ": CUDA Tensor storage is not supported in this "
              "build (requested device " << d.to_string() << ")";
        throw std::runtime_error(os.str());
    }
}

}  // namespace

// ── Construction / factories ─────────────────────────────────────────────

Tensor::Tensor() {
    auto storage = std::make_shared<detail::Storage>(Device::cpu());
    // Default Tensor is Shape{0}: a rank-1 logical view with a single
    // zero-extent dim. This keeps the rank-0 scalar convention
    // (Shape{}.numel() == 1) intact for explicit rank-0 factories.
    impl_ = std::make_shared<detail::TensorImpl>(Shape{0}, storage);
}

Tensor Tensor::empty(const Shape& shape, Device device) {
    reject_cuda("Tensor::empty", device);
    const auto n = safe_numel_to_size("Tensor::empty", shape.numel());
    auto storage = std::make_shared<detail::Storage>(device);
    storage->data.resize(n);
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

Tensor Tensor::zeros(const Shape& shape, Device device) {
    reject_cuda("Tensor::zeros", device);
    const auto n = safe_numel_to_size("Tensor::zeros", shape.numel());
    auto storage = std::make_shared<detail::Storage>(device);
    storage->data.resize(n);  // std::vector zero-initializes numeric elements
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

Tensor Tensor::ones(const Shape& shape, Device device) {
    reject_cuda("Tensor::ones", device);
    const auto n = safe_numel_to_size("Tensor::ones", shape.numel());
    auto storage = std::make_shared<detail::Storage>(device);
    storage->data.assign(n, 1.f);
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

Tensor Tensor::from_host(const float* data,
                         const Shape& shape,
                         Device target) {
    reject_cuda("Tensor::from_host", target);
    const auto n = safe_numel_to_size("Tensor::from_host", shape.numel());
    auto storage = std::make_shared<detail::Storage>(target);
    if (n == 0) {
        // Zero-element shape: null data is permitted and no copy is done.
        Tensor t;
        t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
        return t;
    }
    if (data == nullptr) {
        std::ostringstream os;
        os << "Tensor::from_host: null source for non-zero tensor "
              "(shape numel = " << n << ")";
        throw std::runtime_error(os.str());
    }
    storage->data.assign(data, data + n);
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

// ── Accessors ────────────────────────────────────────────────────────────

const Shape& Tensor::shape() const noexcept { return impl_->shape; }
Device Tensor::device() const noexcept { return impl_->storage->device; }
std::size_t Tensor::elements() const noexcept {
    return impl_->storage->data.size();
}
bool Tensor::empty() const noexcept { return impl_->storage->data.empty(); }

// ── Transfer / copy / reshape ────────────────────────────────────────────

Tensor Tensor::to(Device target) const {
    if (target.is_cuda()) {
        std::ostringstream os;
        os << "Tensor::to: CUDA target is not supported in this build "
              "(requested " << target.to_string() << ")";
        throw std::runtime_error(os.str());
    }
    if (impl_->storage->device == target) {
        // Same-device transfer: shallow share.
        Tensor t;
        t.impl_ = impl_;
        return t;
    }
    std::ostringstream os;
    os << "Tensor::to: cross-device transfer not supported in this "
          "build (from " << impl_->storage->device.to_string() << " to "
       << target.to_string() << ")";
    throw std::runtime_error(os.str());
}

Tensor Tensor::clone() const {
    auto storage = std::make_shared<detail::Storage>(impl_->storage->device);
    storage->data = impl_->storage->data;  // deep copy of float32 buffer
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(impl_->shape, storage);
    return t;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    const auto n = safe_numel_to_size("Tensor::reshape", new_shape.numel());
    const auto current = impl_->storage->data.size();
    if (n != current) {
        std::ostringstream os;
        os << "Tensor::reshape: numel mismatch (requested " << n
           << ", have " << current << ")";
        throw std::invalid_argument(os.str());
    }
    // Reshape shares the underlying Storage; it creates a new
    // TensorImpl holding the new Shape so the source Tensor's shape
    // and any other Tensor that aliases it via shared_ptr<TensorImpl>
    // are NOT mutated.
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(new_shape,
                                                   impl_->storage);
    return t;
}

// ── Host copies ─────────────────────────────────────────────────────────

void Tensor::copy_to_host(float* destination, std::size_t count) const {
    const auto& data = impl_->storage->data;
    if (count != data.size()) {
        std::ostringstream os;
        os << "Tensor::copy_to_host: count mismatch (requested " << count
           << ", elements = " << data.size() << ")";
        throw std::runtime_error(os.str());
    }
    if (data.empty()) return;  // zero-element: no-op, null OK
    if (destination == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_to_host: null destination for non-zero tensor");
    }
    for (std::size_t i = 0; i < data.size(); ++i) destination[i] = data[i];
}

void Tensor::copy_from_host(const float* source, std::size_t count) {
    auto& data = impl_->storage->data;
    if (count != data.size()) {
        std::ostringstream os;
        os << "Tensor::copy_from_host: count mismatch (requested " << count
           << ", elements = " << data.size() << ")";
        throw std::runtime_error(os.str());
    }
    if (data.empty()) return;  // zero-element: no-op, null OK
    if (source == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_from_host: null source for non-zero tensor");
    }
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = source[i];
}

}  // namespace ag
