#pragma once
// extension/cuda.h — opt-in expert header for the borrowed CUDA view.
//
// This header is the ARCHITECTURE_REFACTOR_PLAN.md §6.3 expert boundary
// for CUDA interoperation. The normal public
// `autograd/tensor.h` is intentionally CUDA-runtime-free; downstream
// code (custom CUDA kernels, integration tests) that wants to inspect
// or write into a `ag::Tensor`'s raw device buffer includes this
// header explicitly to opt back into the borrowed-view surface.
//
// Header hygiene:
//   * This file includes NO CUDA runtime header. It compiles cleanly on
//     a CPU-only install. The implementation in src/core/cuda_view.cpp
//     links against src/cuda/tensor_storage_runtime.cu on a CUDA build
//     and against the throwing stubs in src/core/cuda_runtime_stubs.cpp
//     on a CPU-only build.
//   * The view structs are POD-style aggregates of plain C++17 types
//     (raw pointer, ag::Shape by value, plain int, std::size_t). They
//     do not depend on any CUDA type or macro.
//
// Borrowed-view contract (FUNDAMENTAL):
//   * View owns nothing. The library retains ownership of every byte
//     the view points to.
//   * Keep at least one Tensor alias owning the source Storage alive
//     while using a view. Once the last owner is destroyed or
//     reassigned, the pointer dangles.
//   * Tensor::to, Tensor::clone, and Tensor::reshape do NOT mutate
//     the source Tensor's Storage. The source's borrowed view
//     remains valid through any number of those calls. The
//     copy_from_host writes the existing allocation and copy_to_host
//     reads it; neither reallocates.
//   * The caller MUST synchronize any external CUDA work that touches
//     the source Tensor's storage before the last owning Tensor
//     alias is destroyed or reassigned, and before any library
//     operation that reads or writes that storage. The library does
//     not insert any synchronization.
//   * Multiple views into the same source share the same buffer;
//     there is no aliasing protection in the view itself.
//
// Public surface:
//   * struct ConstCudaTensorView { const float* data; Shape shape;
//                                  int device_index; std::size_t numel; };
//     returned by cuda_view(const Tensor&); data is read-only.
//   * struct CudaTensorView { float* data; Shape shape; int device_index;
//                             std::size_t numel; };
//     returned by cuda_view_mut(Tensor&); data is writable.
//   * ConstCudaTensorView cuda_view(const Tensor& t);
//     CudaTensorView cuda_view_mut(Tensor& t);
//     Both throw std::runtime_error when the source is not on a
//     CUDA device.

#include "autograd/shape.h"
#include "autograd/tensor.h"

#include <cstddef>

namespace ag {

struct ConstCudaTensorView {
    const float* data = nullptr;
    Shape shape;
    int device_index = 0;
    std::size_t numel = 0;
};

struct CudaTensorView {
    float* data = nullptr;
    Shape shape;
    int device_index = 0;
    std::size_t numel = 0;
};

// Checked borrowed view of a CUDA Tensor's storage. Throws if the
// source's device is not CUDA. The returned view does not own the
// pointer; see the file header for the lifetime and synchronization
// contract. On a CPU-only build the function is implemented by
// src/core/cuda_view.cpp's CPU stub and always throws.
ConstCudaTensorView cuda_view(const Tensor& t);

// Same as cuda_view(const Tensor&) but the returned data pointer is
// intended to be written through. The library treats the source Tensor
// as live and unchanged until the caller leaves the borrowed buffer's
// exclusive access path.
CudaTensorView cuda_view_mut(Tensor& t);

}  // namespace ag
