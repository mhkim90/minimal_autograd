#pragma once
// extension/eigen.h — opt-in expert header for Eigen interoperation.
//
// This header is the Phase 2.2 / ARCHITECTURE_REFACTOR_PLAN.md §6.2
// expert boundary for CPU interoperation. The normal public
// `autograd/tensor.h` is intentionally Eigen-free; legacy code (and
// downstream projects such as CppResist) that still want the
// `ag::Mat` / `ag::Mats` / `shape(Mat)` / `numel(Mat)` aliases
// include this header explicitly to opt back into the Eigen-coupled
// view of the library.
//
// Use of this header is an explicit choice. It pulls in <Eigen/Dense>
// and the alias surface in this file. It is the only place where the
// Phase 2+ public API touches Eigen. Do not include it from headers
// that must compile without Eigen.
//
// The aliases here are exactly the contract that used to live in
// `autograd/tensor.h` before Phase 2.2. They are preserved verbatim
// so existing in-tree code (Var, Function, ops, modules) continues to
// compile and link without changing its call sites.
//
// Installed along with the rest of the public API by the top-level
// CMakeLists.txt install() rule.

#include <Eigen/Dense>

#include <cstdint>
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

}  // namespace ag
