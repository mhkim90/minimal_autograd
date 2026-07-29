// src/core/tensor.cpp — OOP Tensor implementation.
//
// Storage and per-Tensor logical metadata are private. The public
// handle holds a single std::shared_ptr<detail::TensorImpl>. Each
// TensorImpl owns a Shape + shared_ptr<Storage>, where Storage is the
// library-owned float32 buffer.
//
// Storage holds exactly one authoritative allocation per Tensor:
//   * std::vector<float> on CPU, OR
//   * float* device pointer on CUDA.
// There is no host/device mirror. The choice is made at construction
// (the factory's `Device` argument) and the resulting Tensor is bound
// to that device until an explicit `to(...)` returns a new Tensor on a
// new Storage. Ordinary copies share the Storage (and therefore the
// allocation). `clone()` allocates a fresh Storage of the same device
// and copies the contents. `reshape()` shares the Storage and changes
// only the per-Tensor Shape.
//
// Every CUDA factory / transfer path validates the requested device
// index through cuda_runtime_validate_device so an out-of-range index
// fails fast with a clear runtime_error before any allocation or
// cudaMemcpy is attempted. The validation is performed for zero-element
// tensors as well so a Tensor::empty(Shape{0}, cuda(99)) and a
// zero-element Tensor::to(cuda(99)) are both rejected rather than
// silently accepted.
//
// Lifetime / synchronization contract is documented on
// include/autograd/tensor.h. The borrowed CUDA view surface (raw
// device pointer + Shape + device index) is supplied through the
// explicit opt-in extension header include/autograd/extension/cuda.h
// and is implemented in src/core/cuda_view.cpp. Tensor exposes no
// public mutable accessor for its Storage; the only consumer of the
// private impl_ field outside this translation unit is
// detail::CudaTensorAccess, forward-declared in
// include/autograd/tensor.h, defined in the private
// src/detail/tensor_storage.h header, and implemented below.

#include "autograd/tensor.h"

#include "detail/tensor_cuda_runtime.h"
#include "detail/tensor_storage.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {
namespace detail {

// ── Storage ─────────────────────────────────────────────────────────────

Storage::~Storage() {
    if (device.is_cuda() && cuda_data != nullptr) {
        // Destruction is best-effort because CUDA errors cannot be
        // reported safely from a destructor.
        cuda_runtime_free(cuda_data, device.index());
    }
    // CPU: data vector frees itself.
}

}  // namespace detail

// ── Narrow bridge to the borrowed CUDA view extension ─────────────────

namespace detail {

const float* CudaTensorAccess::cuda_data_const(const Tensor& t) {
    if (!t.device().is_cuda() || t.empty()) {
        return nullptr;
    }
    return t.impl_->storage->cuda_data;
}

float* CudaTensorAccess::cuda_data_mutable(Tensor& t) {
    if (!t.device().is_cuda() || t.empty()) {
        return nullptr;
    }
    return t.impl_->storage->cuda_data;
}

}  // namespace detail

// ── Helpers ─────────────────────────────────────────────────────────────

namespace {

// Convert an int64_t numel to a size_t allocation length with a
// portable overflow / range check. Used by every factory that
// allocates Storage data.
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

// Allocate a fresh Storage for the requested device. CPU uses a
// zero-initialized vector; CUDA uses cuda_runtime_alloc. Factories
// that need a non-zero fill (Tensor::ones, Tensor::from_host with
// host data) call this and then write into the storage.
//
// On a CUDA device, the device index is validated up front through
// cuda_runtime_validate_device so an out-of-range index throws
// before any allocation is attempted. Zero-element CUDA tensors are
// constructed without allocating a device buffer (count=0,
// cuda_data=nullptr) but still pay the validation cost so the
// public contract is uniform.
std::shared_ptr<detail::Storage> make_storage(const Shape& shape,
                                              Device device,
                                              const char* op) {
    if (device.is_cuda()) {
        // Validate the device index even for zero-element shapes so
        // Tensor::empty(Shape{0}, cuda(99)) surfaces a clear error.
        detail::cuda_runtime_validate_device(device.index());
    }
    auto storage = std::make_shared<detail::Storage>(device);
    const auto n = safe_numel_to_size(op, shape.numel());
    if (device.is_cuda()) {
        if (n == 0) {
            storage->count = 0;
            storage->cuda_data = nullptr;
            return storage;
        }
        detail::cuda_runtime_alloc(&storage->cuda_data, n, device.index());
        storage->count = n;
        return storage;
    }
    storage->data.resize(n);  // zero-initializes numeric elements
    storage->count = n;
    return storage;
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
    auto storage = make_storage(shape, device, "Tensor::empty");
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

Tensor Tensor::zeros(const Shape& shape, Device device) {
    // CPU: zero-initializes the std::vector by resize.
    // CUDA: a fresh allocation is undefined; we explicitly zero the
    // contents via copy_from_host of a host-zero buffer of the same
    // shape. The host-zero buffer is well-defined for any numel.
    if (device.is_cuda() && shape.numel() > 0) {
        Tensor t = Tensor::empty(shape, device);
        std::vector<float> zeros(static_cast<std::size_t>(shape.numel()),
                                  0.f);
        t.copy_from_host(zeros.data(), zeros.size());
        return t;
    }
    auto storage = make_storage(shape, device, "Tensor::zeros");
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    return t;
}

Tensor Tensor::ones(const Shape& shape, Device device) {
    Tensor t = Tensor::empty(shape, device);
    if (shape.numel() > 0) {
        std::vector<float> ones(static_cast<std::size_t>(shape.numel()),
                                1.f);
        t.copy_from_host(ones.data(), ones.size());
    }
    return t;
}

Tensor Tensor::from_host(const float* data,
                         const Shape& shape,
                         Device target) {
    const auto n = safe_numel_to_size("Tensor::from_host", shape.numel());
    auto storage = make_storage(shape, target, "Tensor::from_host");
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(shape, storage);
    if (n == 0) {
        // Zero-element shape: null data is permitted and no copy is done.
        return t;
    }
    if (data == nullptr) {
        std::ostringstream os;
        os << "Tensor::from_host: null source for non-zero tensor "
              "(shape numel = " << n << ")";
        throw std::runtime_error(os.str());
    }
    t.copy_from_host(data, n);
    return t;
}

// ── Accessors ────────────────────────────────────────────────────────────

const Shape& Tensor::shape() const noexcept { return impl_->shape; }
Device Tensor::device() const noexcept { return impl_->storage->device; }
std::size_t Tensor::elements() const noexcept {
    return impl_->storage->count;
}
bool Tensor::empty() const noexcept { return impl_->storage->count == 0; }

// ── Transfer / copy / reshape ────────────────────────────────────────────

Tensor Tensor::to(Device target) const {
    const Device src_device = impl_->storage->device;
    if (src_device == target) {
        // Same-device transfer: shallow share of the existing Storage.
        Tensor t;
        t.impl_ = impl_;
        return t;
    }
    // Validate the destination device up front so an out-of-range
    // index throws before any allocation or copy is attempted, even
    // for zero-element cross-device transfers.
    if (target.is_cuda()) {
        detail::cuda_runtime_validate_device(target.index());
    }
    if (src_device.is_cuda()) {
        // Validate the source device too so a stale invalid Storage
        // never reaches the copy helpers.
        detail::cuda_runtime_validate_device(src_device.index());
    }
    const auto n = impl_->storage->count;
    auto dst_storage = std::make_shared<detail::Storage>(target);
    if (n == 0) {
        // Zero-element cross-device transfer: share nothing on the
        // destination; the Storage's cuda_data stays nullptr.
        dst_storage->count = 0;
        dst_storage->cuda_data = nullptr;
        Tensor t;
        t.impl_ = std::make_shared<detail::TensorImpl>(impl_->shape,
                                                       dst_storage);
        return t;
    }
    if (target.is_cuda()) {
        detail::cuda_runtime_alloc(&dst_storage->cuda_data, n,
                                   target.index());
        dst_storage->count = n;
        if (!src_device.is_cuda()) {
            detail::cuda_runtime_copy_h2d(dst_storage->cuda_data,
                                          impl_->storage->data.data(),
                                          n, target.index());
        } else {
            // Cross-device D2D: source and destination live on
            // distinct CUDA devices. Use peer-to-peer copy through
            // the runtime helper so the active-device switching
            // happens in one place.
            detail::cuda_runtime_copy_peer(dst_storage->cuda_data,
                                           target.index(),
                                           impl_->storage->cuda_data,
                                           src_device.index(),
                                           n);
        }
    } else {
        // Target is CPU. CPU storage inherits the source vector.
        dst_storage->data.resize(n);
        dst_storage->count = n;
        if (!src_device.is_cuda()) {
            // CPU -> CPU (different devices shouldn't differ, but be
            // complete: same-device already returned above).
            std::memcpy(dst_storage->data.data(),
                        impl_->storage->data.data(),
                        n * sizeof(float));
        } else {
            detail::cuda_runtime_copy_d2h(dst_storage->data.data(),
                                          impl_->storage->cuda_data,
                                          n, src_device.index());
        }
    }
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(impl_->shape,
                                                   dst_storage);
    return t;
}

Tensor Tensor::clone() const {
    const Device dev = impl_->storage->device;
    if (dev.is_cuda()) {
        detail::cuda_runtime_validate_device(dev.index());
    }
    const auto n = impl_->storage->count;
    auto storage = std::make_shared<detail::Storage>(dev);
    if (dev.is_cuda()) {
        if (n > 0) {
            detail::cuda_runtime_alloc(&storage->cuda_data, n, dev.index());
            detail::cuda_runtime_copy_d2d(storage->cuda_data,
                                          impl_->storage->cuda_data,
                                          n, dev.index());
        }
        storage->count = n;
    } else {
        storage->data = impl_->storage->data;  // deep copy of float32 buffer
        storage->count = storage->data.size();
    }
    Tensor t;
    t.impl_ = std::make_shared<detail::TensorImpl>(impl_->shape, storage);
    return t;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    const auto n = safe_numel_to_size("Tensor::reshape", new_shape.numel());
    if (n != impl_->storage->count) {
        std::ostringstream os;
        os << "Tensor::reshape: numel mismatch (requested " << n
           << ", have " << impl_->storage->count << ")";
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
    const auto& st = *impl_->storage;
    if (count != st.count) {
        std::ostringstream os;
        os << "Tensor::copy_to_host: count mismatch (requested " << count
           << ", elements = " << st.count << ")";
        throw std::runtime_error(os.str());
    }
    if (count == 0) return;  // zero-element: no-op, null OK
    if (destination == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_to_host: null destination for non-zero tensor");
    }
    if (st.device.is_cuda()) {
        // Validate the owning device before the D2H copy. A stale
        // Storage from an out-of-range index is caught here rather
        // than as a cryptic cudaMemcpy failure.
        detail::cuda_runtime_validate_device(st.device.index());
        detail::cuda_runtime_copy_d2h(destination, st.cuda_data, count,
                                      st.device.index());
    } else {
        std::memcpy(destination, st.data.data(), count * sizeof(float));
    }
}

void Tensor::copy_from_host(const float* source, std::size_t count) {
    auto& st = *impl_->storage;
    if (count != st.count) {
        std::ostringstream os;
        os << "Tensor::copy_from_host: count mismatch (requested " << count
           << ", elements = " << st.count << ")";
        throw std::runtime_error(os.str());
    }
    if (count == 0) return;  // zero-element: no-op, null OK
    if (source == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_from_host: null source for non-zero tensor");
    }
    if (st.device.is_cuda()) {
        detail::cuda_runtime_validate_device(st.device.index());
        detail::cuda_runtime_copy_h2d(st.cuda_data, source, count,
                                      st.device.index());
    } else {
        std::memcpy(st.data.data(), source, count * sizeof(float));
    }
}

}  // namespace ag
