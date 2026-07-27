#pragma once
// shape.h — lightweight logical tensor shape metadata.
//
// Storage and compute remain Eigen::MatrixXf. Shape only records the logical
// view, e.g. a flat Mat(N, C*H*W) can carry Shape{N, C, H, W}.
//
// Validation contract (Phase 2):
//   * Construction rejects negative dimensions with std::invalid_argument.
//   * Zero dimensions are allowed (reserved for future empty tensors).
//   * A rank-0 (default-constructed) shape represents a scalar; its
//     numel() is 1, the identity of the empty product.
//   * numel() and contiguous_stride() detect int64 overflow and throw
//     std::overflow_error instead of wrapping or producing UB.
//   * Indexing uses vector::at() and therefore throws std::out_of_range on
//     out-of-range access.
//   * The `sizes` and `strides` fields remain public for current source
//     compatibility. The "revalidating" member functions are:
//     Shape::numel, Shape::operator[], Shape::to_string,
//     contiguous_stride, assert_same_numel, Stride::operator[],
//     Stride::to_string. Each
//     re-validates the current contents so a stale or externally-mutated
//     field cannot bypass the contract. operator<< delegates to
//     to_string and therefore re-validates too. Equality (==, !=) is
//     explicitly a noexcept raw field compare and is NOT part of the
//     revalidation set; mutated-state equality follows from raw field
//     equality, by design.
//
// This header does not implement symbolic shapes, dimension inference,
// broadcasting, or arbitrary strides. See ARCHITECTURE_REFACTOR_PLAN.md
// §5.1 for the target API.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {

using Dims = std::vector<int64_t>;

namespace detail {

inline void validate_dims_non_negative(const Dims& dims, const char* what) {
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
            std::ostringstream os;
            os << what << ": dimension " << i << " is negative ("
               << dims[i] << ")";
            throw std::invalid_argument(os.str());
        }
    }
}

inline void validate_strides_non_negative(const Dims& strides,
                                          const char* what) {
    for (std::size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] < 0) {
            std::ostringstream os;
            os << what << ": stride " << i << " is negative ("
               << strides[i] << ")";
            throw std::invalid_argument(os.str());
        }
    }
}

inline int64_t mul_check_overflow(int64_t a, int64_t b, const char* what) {
    if (a == 0 || b == 0) return 0;
    const int64_t max = std::numeric_limits<int64_t>::max();
    if (a > 0 && b > 0 && a > max / b) {
        std::ostringstream os;
        os << what << ": int64 overflow computing " << a << " * " << b;
        throw std::overflow_error(os.str());
    }
    if (a < 0 || b < 0) {
        std::ostringstream os;
        os << what << ": negative factor (" << a << ", " << b
           << ") in element/stride product";
        throw std::invalid_argument(os.str());
    }
    return a * b;
}

} // namespace detail

struct Shape {
    Dims sizes;

    Shape() = default;

    Shape(std::initializer_list<int64_t> il) : sizes(il) {
        detail::validate_dims_non_negative(sizes, "Shape");
    }

    Shape(int64_t a, int64_t b) : sizes({a, b}) {
        detail::validate_dims_non_negative(sizes, "Shape");
    }

    explicit Shape(Dims s) : sizes(std::move(s)) {
        detail::validate_dims_non_negative(sizes, "Shape");
    }

    int ndim() const noexcept { return static_cast<int>(sizes.size()); }
    std::size_t rank() const noexcept { return sizes.size(); }

    int64_t numel() const {
        detail::validate_dims_non_negative(sizes, "Shape::numel");
        int64_t n = 1;
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            n = detail::mul_check_overflow(n, sizes[i], "Shape::numel");
        }
        return n;
    }

    int64_t operator[](int i) const {
        detail::validate_dims_non_negative(sizes, "Shape::operator[]");
        return sizes.at(i);
    }

    bool operator==(const Shape& o) const noexcept { return sizes == o.sizes; }
    bool operator!=(const Shape& o) const noexcept { return !(*this == o); }

    std::string to_string() const {
        detail::validate_dims_non_negative(sizes, "Shape::to_string");
        std::ostringstream os;
        os << "Shape[";
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            if (i) os << ", ";
            os << sizes[i];
        }
        os << "]";
        return os.str();
    }
};

inline std::ostream& operator<<(std::ostream& os, const Shape& s) {
    return os << s.to_string();
}

struct Stride {
    Dims strides;

    Stride() = default;

    explicit Stride(Dims s) : strides(std::move(s)) {
        detail::validate_strides_non_negative(strides, "Stride");
    }

    int64_t operator[](int i) const {
        detail::validate_strides_non_negative(strides, "Stride::operator[]");
        return strides.at(i);
    }

    bool operator==(const Stride& o) const noexcept {
        return strides == o.strides;
    }
    bool operator!=(const Stride& o) const noexcept { return !(*this == o); }

    std::string to_string() const {
        detail::validate_strides_non_negative(strides, "Stride::to_string");
        std::ostringstream os;
        os << "Stride[";
        for (std::size_t i = 0; i < strides.size(); ++i) {
            if (i) os << ", ";
            os << strides[i];
        }
        os << "]";
        return os.str();
    }
};

inline std::ostream& operator<<(std::ostream& os, const Stride& s) {
    return os << s.to_string();
}

inline Shape make_shape(std::initializer_list<int64_t> il) {
    return Shape(il);
}

inline Shape make_shape(int64_t a, int64_t b) {
    return Shape(a, b);
}

inline Stride contiguous_stride(const Shape& s) {
    detail::validate_dims_non_negative(s.sizes, "contiguous_stride");
    Stride st;
    st.strides.assign(s.ndim(), 0);
    if (s.ndim() == 0) return st;
    st.strides[0] = 1;
    for (int i = 1; i < s.ndim(); ++i) {
        st.strides[i] = detail::mul_check_overflow(
            st.strides[i - 1], s.sizes[i - 1], "contiguous_stride");
    }
    return st;
}

inline void assert_same_numel(const Shape& s, int64_t n) {
    const int64_t shape_numel = s.numel();
    if (shape_numel != n) {
        std::ostringstream os;
        os << "shape numel mismatch: shape has " << shape_numel
           << " elements, storage has " << n;
        throw std::invalid_argument(os.str());
    }
}

} // namespace ag
