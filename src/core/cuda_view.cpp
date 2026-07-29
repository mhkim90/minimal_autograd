// src/core/cuda_view.cpp — implementations of the borrowed CUDA view
// surface.
//
// The two public free functions declared in
// include/autograd/extension/cuda.h (cuda_view / cuda_view_mut) are
// defined here. They reach the source Tensor's authoritative Storage
// through `ag::detail::CudaTensorAccess`, the narrow accessor struct
// forward-declared in include/autograd/tensor.h and defined in the
// private src/detail/tensor_storage.h header. CudaTensorAccess is the
// only consumer of
// Tensor's private `impl_` field outside Tensor's own member
// functions; this translation unit does not see Tensor's internals
// directly.
//
// The two views differ in const correctness: cuda_view(const Tensor&)
// returns ConstCudaTensorView with `const float*`; cuda_view_mut(Tensor&)
// returns CudaTensorView with `float*`. Both routes reach the same
// underlying pointer through the appropriate accessor.
//
// On a CPU-only build, cuda_runtime_validate_device rejects any
// Tensor that points at a CUDA Storage (none can exist), so every
// source Tensor here reports a CPU device and the runtime_error path
// is the only thing this file actually produces.

#include "autograd/extension/cuda.h"
#include "autograd/shape.h"
#include "autograd/tensor.h"

#include "detail/tensor_storage.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace ag {

namespace {

// Build either ConstCudaTensorView or CudaTensorView from the
// source Tensor and the borrowed raw pointer returned by the
// appropriate CudaTensorAccess overload. The pointer type is
// determined by the View's `data` field so the const-correct
// assignment matches without an explicit cast.
template <typename View>
View make_view(const Tensor& t,
               decltype(std::declval<View>().data) raw) {
    View v;
    v.data = raw;
    v.shape = t.shape();
    v.device_index = t.device().index();
    v.numel = t.elements();
    return v;
}

inline void require_cuda_source(const Tensor& t, const char* op) {
    if (!t.device().is_cuda()) {
        std::ostringstream os;
        os << op << ": source Tensor is not on a CUDA device "
              "(got " << t.device().to_string() << ")";
        throw std::runtime_error(os.str());
    }
}

}  // namespace

ConstCudaTensorView cuda_view(const Tensor& t) {
    require_cuda_source(t, "cuda_view");
    return make_view<ConstCudaTensorView>(
        t, detail::CudaTensorAccess::cuda_data_const(t));
}

CudaTensorView cuda_view_mut(Tensor& t) {
    require_cuda_source(t, "cuda_view_mut");
    // The non-const source selects the writable accessor path while
    // preserving the same storage identity.
    return make_view<CudaTensorView>(
        t, detail::CudaTensorAccess::cuda_data_mutable(t));
}

}  // namespace ag
