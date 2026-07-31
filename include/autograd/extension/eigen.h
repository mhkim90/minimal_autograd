#pragma once
// extension/eigen.h — opt-in expert header for Eigen interoperation.
//
// This header is the ARCHITECTURE_REFACTOR_PLAN.md §6.2 expert boundary
// for CPU interoperation. The normal public
// `autograd/tensor.h` is intentionally Eigen-free; legacy code (and
// downstream projects such as CppResist) that still want the
// `ag::Mat` / `ag::Mats` / `shape(Mat)` / `numel(Mat)` aliases
// include this header explicitly to opt back into the Eigen-coupled
// view of the library.
//
// Use of this header is an explicit choice. It pulls in <Eigen/Dense>
// and the alias surface in this file. It is the only place where the
// public Tensor API touches Eigen. Do not include it from headers that
// must compile without Eigen.
//
// The aliases here are exactly the contract that used to live in
// `autograd/tensor.h` before the hidden-storage Tensor refactor. They are
// preserved verbatim so existing in-tree code (Var, Function, ops,
// modules) continues to compile and link without changing its call sites.
// They will be removed once the legacy facade is removed in a future
// cleanup pass.
//
// Installed along with the rest of the public API by the top-level
// CMakeLists.txt install() rule.

#include "autograd/shape.h"
#include "autograd/tensor.h"

#include <Eigen/Dense>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ag {

using Mat = Eigen::MatrixXf;
using Mats = std::vector<Mat>;

// Return the shape of m as {rows, cols}. Preserved for the legacy
// 2D-only contract.
inline std::vector<int64_t> shape(const Mat& m) {
    return {static_cast<int64_t>(m.rows()),
            static_cast<int64_t>(m.cols())};
}

// Number of elements in m. Preserved for the legacy 2D-only contract.
inline int64_t numel(const Mat& m) {
    return static_cast<int64_t>(m.rows()) *
           static_cast<int64_t>(m.cols());
}

// Copy the Eigen matrix into a freshly allocated CPU Tensor with
// Shape{rows, cols}. Logical (row, col) indexing is preserved: the
// returned Tensor's flat index r * cols + c mirrors the Eigen matrix's
// logical (row, col) layout. Eigen stores column-major; the
// canonical Tensor stores row-major, so this is a real copy with an
// explicit reordering — it does not alias the Eigen storage and does
// not reinterpret raw bytes. It is not a raw view and performs no device
// transfer.
inline Tensor tensor_from_eigen(const Mat& m) {
    const int64_t rows = static_cast<int64_t>(m.rows());
    const int64_t cols = static_cast<int64_t>(m.cols());
    Shape s = make_shape(rows, cols);
    std::vector<float> host(s.numel());
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            host[static_cast<std::size_t>(r * cols + c)] = m(
                static_cast<Eigen::Index>(r),
                static_cast<Eigen::Index>(c));
        }
    }
    return Tensor::from_host(host.empty() ? nullptr : host.data(), s);
}

// Materialize a rank-2 CPU Tensor into a freshly allocated Eigen matrix.
// Logical (row, col) values are preserved across the row-major Tensor
// / column-major Eigen boundary by an explicit reordering; the returned
// matrix does not alias the Tensor's storage and is not a raw view. It throws
// std::invalid_argument when the Tensor is not rank-2 or not on a CPU device;
// it never transfers a non-CPU Tensor implicitly.
inline Mat tensor_to_eigen(const Tensor& t) {
    if (t.shape().rank() != 2) {
        std::ostringstream os;
        os << "tensor_to_eigen: expected rank-2 Tensor, got rank "
           << t.shape().rank();
        throw std::invalid_argument(os.str());
    }
    if (!t.device().is_cpu()) {
        std::ostringstream os;
        os << "tensor_to_eigen: expected CPU Tensor, got device "
           << t.device().to_string();
        throw std::invalid_argument(os.str());
    }
    const int64_t rows = t.shape()[0];
    const int64_t cols = t.shape()[1];
    std::vector<float> host(t.elements());
    t.copy_to_host(host.empty() ? nullptr : host.data(), host.size());
    Mat m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            m(static_cast<Eigen::Index>(r),
              static_cast<Eigen::Index>(c)) =
                host[static_cast<std::size_t>(r * cols + c)];
        }
    }
    return m;
}

}  // namespace ag
