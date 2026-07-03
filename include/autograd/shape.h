#pragma once
// shape.h — lightweight logical tensor shape metadata.
//
// Storage and compute remain Eigen::MatrixXf. Shape only records the logical
// view, e.g. a flat Mat(N, C*H*W) can carry Shape{N, C, H, W}.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace ag {

using Dims = std::vector<int64_t>;

struct Shape {
    Dims sizes;

    Shape() = default;
    Shape(std::initializer_list<int64_t> il) : sizes(il) {}
    Shape(int64_t a, int64_t b) : sizes({a, b}) {}
    explicit Shape(Dims s) : sizes(std::move(s)) {}

    int ndim() const noexcept { return static_cast<int>(sizes.size()); }

    int64_t numel() const noexcept {
        int64_t n = 1;
        for (auto s : sizes) n *= s;
        return n;
    }

    int64_t operator[](int i) const { return sizes.at(i); }

    bool operator==(const Shape& o) const noexcept { return sizes == o.sizes; }
    bool operator!=(const Shape& o) const noexcept { return !(*this == o); }
};

struct Stride {
    Dims strides;

    Stride() = default;
    explicit Stride(Dims s) : strides(std::move(s)) {}

    int64_t operator[](int i) const { return strides.at(i); }

    bool operator==(const Stride& o) const noexcept { return strides == o.strides; }
    bool operator!=(const Stride& o) const noexcept { return !(*this == o); }
};

inline Shape make_shape(std::initializer_list<int64_t> il) {
    return Shape(il);
}

inline Shape make_shape(int64_t a, int64_t b) {
    return Shape(a, b);
}

inline Stride contiguous_stride(const Shape& s) {
    Stride st;
    st.strides.assign(s.ndim(), 0);
    if (s.ndim() == 0) return st;
    st.strides[0] = 1;
    for (int i = 1; i < s.ndim(); ++i) {
        st.strides[i] = st.strides[i - 1] * s.sizes[i - 1];
    }
    return st;
}

inline void assert_same_numel(const Shape& s, int64_t n) {
    assert(s.numel() == n && "shape numel mismatch");
}

} // namespace ag
