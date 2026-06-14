#pragma once
// tensor.h — core tensor type alias.
//
// All data in this library is a 2D Eigen::MatrixXf. Higher-dimensional
// tensors (e.g. images for Conv2d) are flattened to (N, C*H*W) and the
// caller is responsible for tracking the original (C, H, W) shape.
// See conv.h for the conv-layer layout convention.

#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace ag {

using Mat = Eigen::MatrixXf;
using Mats = std::vector<Mat>;

// Return the shape of m as {rows, cols}.
inline std::vector<int64_t> shape(const Mat& m) {
    return {static_cast<int64_t>(m.rows()),
            static_cast<int64_t>(m.cols())};
}

// Number of elements in m.
inline int64_t numel(const Mat& m) {
    return static_cast<int64_t>(m.rows()) *
           static_cast<int64_t>(m.cols());
}

} // namespace ag
