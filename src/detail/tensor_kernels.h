#pragma once
// Private CPU tensor arithmetic for the autograd vertical slice.
//
// All kernels act on Tensor-backed dense storage laid out last-axis
// contiguous (canonical row-major): for shape (D0, D1, ..., D{n-1})
// the strides are stride[n-1] = 1, stride[i] = stride[i+1] * D[i+1].
// For rank-2 the flat index for (row, col) is `row * C + col`. Host
// copies go through Tensor::copy_to_host / copy_from_host; shapes
// and devices are validated by callers. New kernels are rank-agnostic
// and accept any rank + axis.

#include "autograd/tensor.h"

#include "detail/constants.h"
#include "detail/tensor_ops.h"

#ifdef AUTOGRAD_USE_CUDA
#include "detail/tensor_cuda_ops.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ag {
namespace detail {

namespace {

inline void require_same_shape(const char* op, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        std::ostringstream os;
        os << op << ": shape mismatch (" << a.shape() << " vs "
           << b.shape() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline void require_same_device(const char* op, const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        std::ostringstream os;
        os << op << ": device mismatch (" << a.device() << " vs "
           << b.device() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline int64_t checked_dim_sum(const char* op, int64_t a, int64_t b) {
    if (b > std::numeric_limits<int64_t>::max() - a) {
        throw std::overflow_error(std::string(op) + ": dimension overflow");
    }
    return a + b;
}

// ── Shape / stride / index helpers ─────────────────────────────────────
// Last-axis-contiguous strides (row-major): stride[n-1] = 1,
// stride[i] = stride[i+1] * shape[i+1]. For rank-2 (R, C): flat
// index = row * C + col.
inline std::vector<int64_t> contiguous_strides(const Shape& s) {
    return contiguous_stride(s).strides;
}

// Flat offset from a multi-index and stride vector.
inline int64_t linear_offset(const std::vector<int64_t>& idx,
                             const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        off += idx[i] * strides[i];
    }
    return off;
}

// Normalize a single axis against a rank. Negative axes wrap.
inline int normalize_axis(int axis, int rank, const char* what) {
    if (rank <= 0) {
        std::ostringstream os;
        os << what << ": empty rank";
        throw std::invalid_argument(os.str());
    }
    int n = axis;
    if (n < 0) n += rank;
    if (n < 0 || n >= rank) {
        std::ostringstream os;
        os << what << ": axis " << axis << " out of range for rank "
           << rank;
        throw std::invalid_argument(os.str());
    }
    return n;
}

// Normalize a list of axes; rejects duplicates and out-of-range.
inline std::vector<int> normalize_axes(const std::vector<int>& axes,
                                       int rank,
                                       const char* what) {
    std::vector<int> out;
    out.reserve(axes.size());
    for (int a : axes) {
        int n = normalize_axis(a, rank, what);
        for (int prev : out) {
            if (prev == n) {
                std::ostringstream os;
                os << what << ": duplicate axis " << n;
                throw std::invalid_argument(os.str());
            }
        }
        out.push_back(n);
    }
    return out;
}

// Increment a multi-index in last-axis-contiguous (row-major) order:
// the LAST index advances fastest, matching the byte order in
// contiguous row-major storage.
inline bool increment_index(std::vector<int64_t>& idx, const Shape& s) {
    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
        if (idx[i] + 1 < s.sizes[i]) {
            ++idx[i];
            return true;
        }
        idx[i] = 0;
    }
    return false;
}

}  // namespace

inline Tensor tensor_ones(const Shape& shape, Device device) {
#ifdef AUTOGRAD_USE_CUDA
    if (device.is_cuda()) return cuda_tensor_ones(shape, device);
#endif
    return Tensor::ones(shape, device);
}

// ── Shape ──────────────────────────────────────────────────────────────

inline Tensor tensor_sin(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::sin(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_cos(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::cos(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sin_backward(const Tensor& g, const Tensor& x) {
    require_same_shape("sin_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * std::cos(xv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_cos_backward(const Tensor& g, const Tensor& x) {
    require_same_shape("cos_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (-std::sin(xv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_clamp(const Tensor& a, float lo, float hi) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = av[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        ov[i] = v;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_clamp_backward(const Tensor& g,
                                    const Tensor& x,
                                    float lo,
                                    float hi) {
    require_same_shape("clamp_backward", g, x);
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = (xv[i] >= lo && xv[i] <= hi) ? gv[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

// ── N-D ops (rank-agnostic) ─────────────────────────────────────────────
//
// Storage is last-axis contiguous (row-major): stride[n-1] = 1,
// stride[i] = stride[i+1] * D[i+1]. For rank-2 the flat index for
// (row, col) is `row * C + col`.

// softmax_nd: per-row-equivalent reduce along an arbitrary axis. The
// shape of the output equals the shape of `a`. saved_softmax stores
// the post-softmax tensor for backward.
inline Tensor tensor_softmax_nd(const Tensor& a, int axis,
                                Tensor& saved_softmax) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "softmax");
#ifdef AUTOGRAD_USE_CUDA
    if (a.device().is_cuda()) {
        return cuda_tensor_softmax(a, ax, saved_softmax);
    }
#endif
    Tensor out = Tensor::empty(s, a.device());
    saved_softmax = Tensor::empty(s, a.device());
    if (a.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> av(a.elements()), ov(a.elements()), sv(a.elements());
    a.copy_to_host(av.data(), av.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        float maxval = -std::numeric_limits<float>::infinity();
        for (int64_t k = 0; k < inner_dim; ++k) {
            const float v = av[base + k * inner_stride];
            if (v > maxval) maxval = v;
        }
        float denom = 0.f;
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            const float e = std::exp(av[off] - maxval);
            sv[off] = e;
            denom += e;
        }
        const float inv = 1.f / denom;
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            ov[off] = sv[off] * inv;
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }

    out.copy_from_host(ov.data(), ov.size());
    saved_softmax.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_softmax_backward_nd(const Tensor& g,
                                         const Tensor& saved_softmax,
                                         int axis) {
    require_same_shape("softmax_backward", g, saved_softmax);
    require_same_device("softmax_backward", g, saved_softmax);
    const Shape& s = g.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "softmax_backward");
#ifdef AUTOGRAD_USE_CUDA
    if (g.device().is_cuda()) {
        return cuda_tensor_softmax_backward(g, saved_softmax, ax);
    }
#endif
    Tensor out = Tensor::empty(s, g.device());
    if (g.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> gv(g.elements()), sv(g.elements()), ov(g.elements());
    g.copy_to_host(gv.data(), gv.size());
    saved_softmax.copy_to_host(sv.data(), sv.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        float dot = 0.f;
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            dot += gv[off] * sv[off];
        }
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            ov[off] = sv[off] * (gv[off] - dot);
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }

    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_log_softmax_nd(const Tensor& a, int axis,
                                    Tensor& saved_lsm) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "log_softmax");
#ifdef AUTOGRAD_USE_CUDA
    if (a.device().is_cuda()) {
        return cuda_tensor_log_softmax(a, ax, saved_lsm);
    }
#endif
    Tensor out = Tensor::empty(s, a.device());
    saved_lsm = Tensor::empty(s, a.device());
    if (a.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> av(a.elements()), ov(a.elements()), lv(a.elements());
    a.copy_to_host(av.data(), av.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        float maxval = av[base];
        for (int64_t k = 1; k < inner_dim; ++k) {
            const float v = av[base + k * inner_stride];
            if (v > maxval) maxval = v;
        }
        float denom = 0.f;
        for (int64_t k = 0; k < inner_dim; ++k) {
            denom += std::exp(av[base + k * inner_stride] - maxval);
        }
        const float log_sum = maxval + std::log(denom);
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            const float v = av[off] - log_sum;
            lv[off] = v;
            ov[off] = v;
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }

    out.copy_from_host(ov.data(), ov.size());
    saved_lsm.copy_from_host(lv.data(), lv.size());
    return out;
}

inline Tensor tensor_log_softmax_backward_nd(const Tensor& g,
                                             const Tensor& saved_lsm,
                                             int axis) {
    require_same_shape("log_softmax_backward", g, saved_lsm);
    require_same_device("log_softmax_backward", g, saved_lsm);
    const Shape& s = g.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "log_softmax_backward");
#ifdef AUTOGRAD_USE_CUDA
    if (g.device().is_cuda()) {
        return cuda_tensor_log_softmax_backward(g, saved_lsm, ax);
    }
#endif
    Tensor out = Tensor::empty(s, g.device());
    if (g.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> gv(g.elements()), lv(g.elements()), ov(g.elements());
    g.copy_to_host(gv.data(), gv.size());
    saved_lsm.copy_to_host(lv.data(), lv.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        float row_sum = 0.f;
        for (int64_t k = 0; k < inner_dim; ++k) {
            row_sum += gv[base + k * inner_stride];
        }
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t off = base + k * inner_stride;
            const float sm = std::exp(lv[off]);
            ov[off] = gv[off] - sm * row_sum;
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }

    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_cumsum_nd(const Tensor& a, int axis) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "cumsum");
    Tensor out = Tensor::empty(s, a.device());
    if (a.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> av(a.elements()), ov(a.elements());
    a.copy_to_host(av.data(), av.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        if (inner_dim > 0) {
            float acc = av[base];
            ov[base] = acc;
            for (int64_t k = 1; k < inner_dim; ++k) {
                const int64_t off = base + k * inner_stride;
                acc += av[off];
                ov[off] = acc;
            }
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_cumsum_backward_nd(const Tensor& g, int axis) {
    const Shape& s = g.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "cumsum_backward");
    Tensor out = Tensor::empty(s, g.device());
    if (g.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> gv(g.elements()), ov(g.elements());
    g.copy_to_host(gv.data(), gv.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        if (inner_dim > 0) {
            float acc = gv[base + (inner_dim - 1) * inner_stride];
            ov[base + (inner_dim - 1) * inner_stride] = acc;
            for (int64_t k = inner_dim - 2; k >= 0; --k) {
                const int64_t off = base + k * inner_stride;
                acc += gv[off];
                ov[off] = acc;
            }
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_flip_nd(const Tensor& a, int axis) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = normalize_axis(axis, rank, "flip");
    Tensor out = Tensor::empty(s, a.device());
    if (a.elements() == 0) return out;

    const std::vector<int64_t> strides = contiguous_strides(s);
    const int64_t inner_dim = s.sizes[ax];
    const int64_t inner_stride = strides[ax];

    std::vector<float> av(a.elements()), ov(a.elements());
    a.copy_to_host(av.data(), av.size());

    std::vector<int64_t> outer_shape;
    outer_shape.reserve(rank - 1);
    for (int i = 0; i < rank; ++i) {
        if (i != ax) outer_shape.push_back(s.sizes[i]);
    }
    int64_t outer_numel = 1;
    for (int64_t d : outer_shape) outer_numel *= d;

    std::vector<int64_t> outer_idx(outer_shape.size(), 0);
    for (int64_t o = 0; o < outer_numel; ++o) {
        int64_t base = 0;
        int j = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != ax) {
                base += outer_idx[j] * strides[i];
                ++j;
            }
        }
        for (int64_t k = 0; k < inner_dim; ++k) {
            const int64_t dst = base + k * inner_stride;
            const int64_t src = base + (inner_dim - 1 - k) * inner_stride;
            ov[dst] = av[src];
        }
        for (int j = static_cast<int>(outer_idx.size()) - 1; j >= 0; --j) {
            if (++outer_idx[j] < outer_shape[j]) break;
            outer_idx[j] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

inline Tensor tensor_transpose_nd(const Tensor& a, int axis0, int axis1) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    int ax0 = normalize_axis(axis0, rank, "transpose");
    int ax1 = normalize_axis(axis1, rank, "transpose");
    if (ax0 == ax1) return a.clone();
    Dims out_sizes(s.sizes.begin(), s.sizes.end());
    std::swap(out_sizes[ax0], out_sizes[ax1]);
    Shape out_shape(out_sizes);
    Tensor out = Tensor::empty(out_shape, a.device());
    if (a.elements() == 0) return out;

    const std::vector<int64_t> in_strides = contiguous_strides(s);
    const std::vector<int64_t> out_strides = contiguous_strides(out_shape);

    std::vector<float> av(a.elements()), ov(a.elements());
    a.copy_to_host(av.data(), av.size());

    std::vector<int64_t> out_idx(rank, 0);
    std::vector<int64_t> in_idx(rank, 0);
    const int64_t total = a.elements();
    for (int64_t flat = 0; flat < total; ++flat) {
        for (int i = 0; i < rank; ++i) in_idx[i] = out_idx[i];
        std::swap(in_idx[ax0], in_idx[ax1]);
        const int64_t in_off = linear_offset(in_idx, in_strides);
        const int64_t out_off = linear_offset(out_idx, out_strides);
        ov[out_off] = av[in_off];
        for (int i = rank - 1; i >= 0; --i) {
            if (++out_idx[i] < out_sizes[i]) break;
            out_idx[i] = 0;
        }
    }
    out.copy_from_host(ov.data(), ov.size());
    return out;
}

// ── NCHW spatial kernels ───────────────────────────────────────────────
//
// All spatial kernels operate on rank-4 tensors in NCHW logical layout
// with last-axis-contiguous (row-major) storage: stride = (C*H*W, H*W,
// W, 1). The flat index for (n, c, h, w) is ((n*C + c)*H + h)*W + w.
//
// stride > 0 and pad >= 0 are required. The output extent formulas
// match the legacy plan: oH = (H + 2*pad - kH) / stride + 1, etc.
// Callers must validate these produce positive output extents before
// invoking the kernels.

inline int64_t nchw_output_extent(int64_t input, int64_t kernel,
                                  int64_t stride, int64_t pad,
                                  const char* what) {
    const int64_t span = input + 2 * pad - kernel;
    if (span < 0) {
        std::ostringstream os;
        os << what << ": input plus 2*pad smaller than kernel ("
           << input << " + 2*" << pad << " < " << kernel << ")";
        throw std::invalid_argument(os.str());
    }
    if (span % stride != 0) {
        std::ostringstream os;
        os << what << ": output extent non-integer (span " << span
           << " not divisible by stride " << stride << ")";
        throw std::invalid_argument(os.str());
    }
    return span / stride + 1;
}

inline Tensor tensor_im2col_nchw(const Tensor& input,
                                 int kH, int kW,
                                 int stride, int pad) {
    const Shape& s = input.shape();
    const int N = static_cast<int>(s[0]);
    const int C = static_cast<int>(s[1]);
    const int H = static_cast<int>(s[2]);
    const int W = static_cast<int>(s[3]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "im2col"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "im2col"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor col = Tensor::empty(Shape({N, K_flat, P_flat}), input.device());
    if (col.elements() == 0) return col;

    std::vector<float> in_data(input.elements()), col_data(col.elements());
    input.copy_to_host(in_data.data(), in_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    const int k = c * kH * kW + kh * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        const int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            const int iw = ow * stride + kw - pad;
                            const int p = oh * oW + ow;
                            float v = 0.f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                v = in_data[n * in_stride_n
                                          + c * in_stride_c
                                          + ih * in_stride_h
                                          + iw * in_stride_w];
                            }
                            col_data[n * col_stride_n
                                   + k * col_stride_k
                                   + p * col_stride_p] = v;
                        }
                    }
                }
            }
        }
    }
    col.copy_from_host(col_data.data(), col_data.size());
    return col;
}

inline Tensor tensor_col2im_nchw(const Tensor& col,
                                 int N, int C, int H, int W,
                                 int kH, int kW,
                                 int stride, int pad) {
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "col2im"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "col2im"));
    const int K_flat = C * kH * kW;
    const int P_flat = oH * oW;

    Tensor out = Tensor::zeros(Shape({N, C, H, W}), col.device());
    if (out.elements() == 0) return out;

    std::vector<float> col_data(col.elements()), out_data(out.elements());
    col.copy_to_host(col_data.data(), col_data.size());
    out.copy_to_host(out_data.data(), out_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    const int k = c * kH * kW + kh * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        const int ih = oh * stride + kh - pad;
                        for (int ow = 0; ow < oW; ++ow) {
                            const int iw = ow * stride + kw - pad;
                            const int p = oh * oW + ow;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            const int in_off = n * in_stride_n
                                             + c * in_stride_c
                                             + ih * in_stride_h
                                             + iw * in_stride_w;
                            const int col_off = n * col_stride_n
                                              + k * col_stride_k
                                              + p * col_stride_p;
                            out_data[in_off] += col_data[col_off];
                        }
                    }
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

// ── DepthwiseConv2d (NCHW) ────────────────────────────────────────────
//
// Per-channel convolution: each input channel c is convolved with its
// own (kH, kW) filter weight[c, :, :]. weight is rank-3 (C, kH, kW),
// bias is rank-1 (C,). The forward is decomposed through the existing
// im2col + col2im helpers by reusing the rank-3 col matrix view.
//
// Storage layout (last-axis contiguous / row-major):
//   col shape: (N, C*kH*kW, oH*oW), strides (C*kH*kW*oH*oW, oH*oW, 1)
//   For each (n, c, k, p): col[n, c*kH*kW + k, p] = input[n, c, ...]
//   weight[c, kh, kw] index: c*kH*kW + kh*kW + kw
//   For each (n, c, oh, ow):
//     out[n, c, oh, ow] = bias[c] + sum_{kh, kw} weight[c, kh, kw]
//                                                * col[n, c*kH*kW + kh*kW + kw, oh*oW + ow]
inline Tensor tensor_depthwise_conv2d_nchw_forward(
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        int stride, int pad, Tensor& saved_col) {
    const Shape& in_s = input.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    const Shape& w_s = weight.shape();
    const int kH = static_cast<int>(w_s[1]);
    const int kW = static_cast<int>(w_s[2]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "depthwise_conv2d"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "depthwise_conv2d"));
    const int ksz = kH * kW;
    const int K_flat = C * ksz;
    const int P_flat = oH * oW;

    saved_col = tensor_im2col_nchw(input, kH, kW, stride, pad);
    Tensor out = Tensor::empty(Shape({N, C, oH, oW}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> w_data(weight.elements());
    std::vector<float> b_data(bias.elements());
    std::vector<float> col_data(saved_col.elements());
    std::vector<float> out_data(out.elements());
    weight.copy_to_host(w_data.data(), w_data.size());
    bias.copy_to_host(b_data.data(), b_data.size());
    saved_col.copy_to_host(col_data.data(), col_data.size());

    const int w_stride_c = kH * kW;
    const int w_stride_kh = kW;
    const int w_stride_kw = 1;
    const int col_stride_n = K_flat * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;
    const int out_stride_n = C * oH * oW;
    const int out_stride_c = oH * oW;
    const int out_stride_oh = oW;
    const int out_stride_ow = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    float s = b_data[c];
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            const int k = c * ksz + kh * kW + kw;
                            const int w_off = c * w_stride_c
                                             + kh * w_stride_kh
                                             + kw * w_stride_kw;
                            const int col_off = n * col_stride_n
                                              + k * col_stride_k
                                              + p * col_stride_p;
                            s += w_data[w_off] * col_data[col_off];
                        }
                    }
                    out_data[n * out_stride_n
                           + c * out_stride_c
                           + oh * out_stride_oh
                           + ow * out_stride_ow] = s;
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

// d_input = col2im(W^T_col * d_out, ...). Because depthwise conv has
// per-channel filters, W^T_col multiplied by d_out is per-channel:
//   d_col[n, c*ksz + k, p] = weight[c, kh, kw] * d_out[n, c, oh, ow]
// where (kh, kw) corresponds to k. Then col2im accumulates back to
// the input plane.
inline Tensor tensor_depthwise_conv2d_nchw_backward_input(
        const Tensor& g, const Tensor& weight,
        int N, int C, int H, int W,
        int kH, int kW, int stride, int pad) {
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, pad, "depthwise_conv2d_backward_input"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, pad, "depthwise_conv2d_backward_input"));
    const int ksz = kH * kW;
    const int K_flat = C * ksz;
    const int P_flat = oH * oW;

    Tensor d_col = Tensor::zeros(Shape({N, K_flat, P_flat}), g.device());
    if (d_col.elements() == 0) {
        return Tensor::zeros(Shape({N, C, H, W}), g.device());
    }

    std::vector<float> g_data(g.elements());
    std::vector<float> w_data(weight.elements());
    std::vector<float> dc_data(d_col.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    weight.copy_to_host(w_data.data(), w_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int w_stride_c = kH * kW;
    const int w_stride_kh = kW;
    const int w_stride_kw = 1;
    const int dc_stride_n = K_flat * P_flat;
    const int dc_stride_k = P_flat;
    const int dc_stride_p = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const int p = oh * oW + ow;
                    const float g_val = g_data[n * g_stride_n
                                            + c * g_stride_c
                                            + oh * g_stride_oh
                                            + ow * g_stride_ow];
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            const int k = c * ksz + kh * kW + kw;
                            const int w_off = c * w_stride_c
                                             + kh * w_stride_kh
                                             + kw * w_stride_kw;
                            const int dc_off = n * dc_stride_n
                                              + k * dc_stride_k
                                              + p * dc_stride_p;
                            dc_data[dc_off] += g_val * w_data[w_off];
                        }
                    }
                }
            }
        }
    }
    d_col.copy_from_host(dc_data.data(), dc_data.size());
    return tensor_col2im_nchw(d_col, N, C, H, W, kH, kW, stride, pad);
}

inline Tensor tensor_depthwise_conv2d_nchw_backward_weight(
        const Tensor& g, const Tensor& col, int kH, int kW) {
    const Shape& cs = col.shape();
    const Shape& gs = g.shape();
    const int N = static_cast<int>(cs[0]);
    const int P_flat = static_cast<int>(cs[2]);
    const int C = static_cast<int>(gs[1]);
    const int oH = static_cast<int>(gs[2]);
    const int oW = static_cast<int>(gs[3]);
    const int ksz = kH * kW;

    Tensor d_w = Tensor::zeros(Shape({C, kH, kW}), g.device());
    if (d_w.elements() == 0) return d_w;

    std::vector<float> g_data(g.elements());
    std::vector<float> col_data(col.elements());
    std::vector<float> dw_data(d_w.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    col.copy_to_host(col_data.data(), col_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int col_stride_n = (C * ksz) * P_flat;
    const int col_stride_k = P_flat;
    const int col_stride_p = 1;
    const int dw_stride_c = kH * kW;
    const int dw_stride_kh = kW;
    const int dw_stride_kw = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    double acc = 0.0;
                    const int k = c * ksz + kh * kW + kw;
                    for (int oh = 0; oh < oH; ++oh) {
                        for (int ow = 0; ow < oW; ++ow) {
                            const int p = oh * oW + ow;
                            const float g_val = g_data[n * g_stride_n
                                                    + c * g_stride_c
                                                    + oh * g_stride_oh
                                                    + ow * g_stride_ow];
                            const int col_off = n * col_stride_n
                                              + k * col_stride_k
                                              + p * col_stride_p;
                            acc += static_cast<double>(g_val) *
                                   static_cast<double>(col_data[col_off]);
                        }
                    }
                    const int dw_off = c * dw_stride_c
                                     + kh * dw_stride_kh
                                     + kw * dw_stride_kw;
                    dw_data[dw_off] = static_cast<float>(acc);
                }
            }
        }
    }
    d_w.copy_from_host(dw_data.data(), dw_data.size());
    return d_w;
}

inline Tensor tensor_depthwise_conv2d_nchw_backward_bias(const Tensor& g) {
    const Shape& gs = g.shape();
    const int N = static_cast<int>(gs[0]);
    const int C = static_cast<int>(gs[1]);
    const int oH = static_cast<int>(gs[2]);
    const int oW = static_cast<int>(gs[3]);

    Tensor d_b = Tensor::zeros(Shape({C}), g.device());
    if (d_b.elements() == 0) return d_b;

    std::vector<float> g_data(g.elements());
    std::vector<float> db_data(d_b.elements());
    g.copy_to_host(g_data.data(), g_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;

    for (int c = 0; c < C; ++c) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    s += static_cast<double>(
                        g_data[n * g_stride_n
                             + c * g_stride_c
                             + oh * g_stride_oh
                             + ow * g_stride_ow]);
                }
            }
        }
        db_data[c] = static_cast<float>(s);
    }
    d_b.copy_from_host(db_data.data(), db_data.size());
    return d_b;
}

// ── AvgPool2d (NCHW) ──────────────────────────────────────────────────
//
// Average pooling: out[n, c, oh, ow] = (1/(kH*kW)) * sum_{kh, kw}
// input[n, c, oh*stride+kh, ow*stride+kw]. Backward distributes the
// upstream gradient across the kernel positions of each window, so
// each input pixel receives (1/(kH*kW)) times the sum of upstream
// gradients over every window containing it.
inline Tensor tensor_avgpool2d_nchw_forward(const Tensor& input,
                                            int kH, int kW,
                                            int stride) {
    const Shape& s = input.shape();
    const int N = static_cast<int>(s[0]);
    const int C = static_cast<int>(s[1]);
    const int H = static_cast<int>(s[2]);
    const int W = static_cast<int>(s[3]);
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, 0, "avg_pool2d"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, 0, "avg_pool2d"));
    const int ksz = kH * kW;
    const float inv_k = 1.f / static_cast<float>(ksz);

    Tensor out = Tensor::empty(Shape({N, C, oH, oW}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> in_data(input.elements());
    std::vector<float> out_data(out.elements());
    input.copy_to_host(in_data.data(), in_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int out_stride_n = C * oH * oW;
    const int out_stride_c = oH * oW;
    const int out_stride_oh = oW;
    const int out_stride_ow = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float s = 0.f;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * stride + kh;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = ow * stride + kw;
                            s += in_data[n * in_stride_n
                                       + c * in_stride_c
                                       + ih * in_stride_h
                                       + iw * in_stride_w];
                        }
                    }
                    out_data[n * out_stride_n
                           + c * out_stride_c
                           + oh * out_stride_oh
                           + ow * out_stride_ow] = s * inv_k;
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

inline Tensor tensor_avgpool2d_nchw_backward(const Tensor& g,
                                             int N, int C, int H, int W,
                                             int kH, int kW, int stride) {
    const int oH = static_cast<int>(nchw_output_extent(
        H, kH, stride, 0, "avg_pool2d_backward"));
    const int oW = static_cast<int>(nchw_output_extent(
        W, kW, stride, 0, "avg_pool2d_backward"));
    const int ksz = kH * kW;
    const float inv_k = 1.f / static_cast<float>(ksz);

    Tensor d_in = Tensor::zeros(Shape({N, C, H, W}), g.device());
    if (d_in.elements() == 0) return d_in;

    std::vector<float> g_data(g.elements());
    std::vector<float> d_in_data(d_in.elements());
    g.copy_to_host(g_data.data(), g_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    const float g_val = g_data[n * g_stride_n
                                            + c * g_stride_c
                                            + oh * g_stride_oh
                                            + ow * g_stride_ow] * inv_k;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * stride + kh;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = ow * stride + kw;
                            d_in_data[n * in_stride_n
                                    + c * in_stride_c
                                    + ih * in_stride_h
                                    + iw * in_stride_w] += g_val;
                        }
                    }
                }
            }
        }
    }
    d_in.copy_from_host(d_in_data.data(), d_in_data.size());
    return d_in;
}

// ── NearestUpsample2d (NCHW) ──────────────────────────────────────────
//
// Each input pixel (n, c, h, w) is replicated to a scale x scale block
// of output positions (h*scale + sh, w*scale + sw) for sh, sw in
// [0, scale). Backward sums the upstream gradients over the scale x
// scale block for each input pixel.
inline Tensor tensor_nearest_upsample2d_nchw_forward(const Tensor& input,
                                                      int scale) {
    const Shape& s = input.shape();
    const int N = static_cast<int>(s[0]);
    const int C = static_cast<int>(s[1]);
    const int H = static_cast<int>(s[2]);
    const int W = static_cast<int>(s[3]);
    const int oH = H * scale;
    const int oW = W * scale;

    Tensor out = Tensor::empty(Shape({N, C, oH, oW}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> in_data(input.elements());
    std::vector<float> out_data(out.elements());
    input.copy_to_host(in_data.data(), in_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int out_stride_n = C * oH * oW;
    const int out_stride_c = oH * oW;
    const int out_stride_oh = oW;
    const int out_stride_ow = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const float v = in_data[n * in_stride_n
                                          + c * in_stride_c
                                          + h * in_stride_h
                                          + w * in_stride_w];
                    for (int sh = 0; sh < scale; ++sh) {
                        const int oh = h * scale + sh;
                        for (int sw = 0; sw < scale; ++sw) {
                            const int ow = w * scale + sw;
                            out_data[n * out_stride_n
                                   + c * out_stride_c
                                   + oh * out_stride_oh
                                   + ow * out_stride_ow] = v;
                        }
                    }
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    return out;
}

inline Tensor tensor_nearest_upsample2d_nchw_backward(const Tensor& g,
                                                      int N, int C,
                                                      int H, int W,
                                                      int scale) {
    const int oH = H * scale;
    const int oW = W * scale;

    Tensor d_in = Tensor::zeros(Shape({N, C, H, W}), g.device());
    if (d_in.elements() == 0) return d_in;

    std::vector<float> g_data(g.elements());
    std::vector<float> d_in_data(d_in.elements());
    g.copy_to_host(g_data.data(), g_data.size());

    const int g_stride_n = C * oH * oW;
    const int g_stride_c = oH * oW;
    const int g_stride_oh = oW;
    const int g_stride_ow = 1;
    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    double s = 0.0;
                    for (int sh = 0; sh < scale; ++sh) {
                        const int oh = h * scale + sh;
                        for (int sw = 0; sw < scale; ++sw) {
                            const int ow = w * scale + sw;
                            s += static_cast<double>(
                                g_data[n * g_stride_n
                                     + c * g_stride_c
                                     + oh * g_stride_oh
                                     + ow * g_stride_ow]);
                        }
                    }
                    d_in_data[n * in_stride_n
                             + c * in_stride_c
                             + h * in_stride_h
                             + w * in_stride_w] = static_cast<float>(s);
                }
            }
        }
    }
    d_in.copy_from_host(d_in_data.data(), d_in_data.size());
    return d_in;
}

// ── GroupNorm (NCHW) ─────────────────────────────────────────────────
//
// Normalizes over (channels_per_group * H * W) elements per (sample,
// group). Channels are partitioned into num_groups contiguous groups:
// group g contains channels [g * ch_per_g, (g+1) * ch_per_g).
//
//   mean[n, g]      = (1/M) * sum x[n, c, h, w]
//   var[n, g]       = (1/M) * sum (x[n, c, h, w] - mean[n, g])^2
//   inv_std[n, g]   = 1 / sqrt(var[n, g] + eps)
//   xhat[n, c, h, w] = (x[n, c, h, w] - mean[n, g]) * inv_std[n, g]
//   y[n, c, h, w]   = gamma[c] * xhat[n, c, h, w] + beta[c]
//
// Backward uses the closed-form mean/var reduction, expressed in
// terms of xhat to avoid recomputing mean/var from the input.
//
//   dyh[n, c, h, w] = grad[n, c, h, w] * gamma[c]
//   sum_dyh[n, g]   = sum dyh over the group
//   sum_dyh_x[n, g] = sum dyh * xhat over the group
//   d_x[n, c, h, w] = inv_std[n, g] *
//                      (dyh[n, c, h, w]
//                       - (1/M) * (sum_dyh[n, g]
//                                  + xhat[n, c, h, w] * sum_dyh_x[n, g]))
//   d_gamma[c]      = sum_{n, h, w} grad[n, c, h, w] * xhat[n, c, h, w]
//   d_beta[c]       = sum_{n, h, w} grad[n, c, h, w]
//
// Saved tensors for backward:
//   saved_xhat : (N, C, H, W) - normalized pre-affine values
//   saved_inv_std : (N, num_groups) - per-(sample, group) inv_std
inline Tensor tensor_group_norm_nchw_forward(
        const Tensor& input, const Tensor& gamma, const Tensor& beta,
        int num_groups, float eps,
        Tensor& saved_xhat, Tensor& saved_inv_std) {
    const Shape& in_s = input.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    const int ch_per_g = C / num_groups;
    const int M = ch_per_g * H * W;

    Tensor out = Tensor::empty(in_s, input.device());
    saved_xhat = Tensor::empty(in_s, input.device());
    saved_inv_std = Tensor::empty(Shape({N, num_groups}), input.device());
    if (out.elements() == 0) return out;

    std::vector<float> in_data(input.elements());
    std::vector<float> gamma_data(gamma.elements());
    std::vector<float> beta_data(beta.elements());
    std::vector<float> out_data(out.elements());
    std::vector<float> xhat_data(out.elements());
    std::vector<float> inv_std_data(static_cast<std::size_t>(N) * num_groups);
    input.copy_to_host(in_data.data(), in_data.size());
    gamma.copy_to_host(gamma_data.data(), gamma_data.size());
    beta.copy_to_host(beta_data.data(), beta_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int out_stride_n = C * H * W;
    const int out_stride_c = H * W;
    const int out_stride_h = W;
    const int out_stride_w = 1;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int ch0 = g * ch_per_g;
            double mean = 0.0;
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        mean += static_cast<double>(
                            in_data[n * in_stride_n
                                  + c * in_stride_c
                                  + h * in_stride_h
                                  + w * in_stride_w]);
                    }
                }
            }
            mean /= static_cast<double>(M);
            double var = 0.0;
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const double diff = static_cast<double>(
                            in_data[n * in_stride_n
                                  + c * in_stride_c
                                  + h * in_stride_h
                                  + w * in_stride_w]) - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= static_cast<double>(M);
            const double inv_std = 1.0 /
                std::sqrt(var + static_cast<double>(eps));
            inv_std_data[n * num_groups + g] = static_cast<float>(inv_std);
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                const float gc = gamma_data[c];
                const float bc = beta_data[c];
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const int off = n * in_stride_n
                                      + c * in_stride_c
                                      + h * in_stride_h
                                      + w * in_stride_w;
                        const double diff = static_cast<double>(
                            in_data[off]) - mean;
                        const float xn = static_cast<float>(diff * inv_std);
                        xhat_data[off] = xn;
                        out_data[off] = gc * xn + bc;
                    }
                }
            }
        }
    }
    out.copy_from_host(out_data.data(), out_data.size());
    saved_xhat.copy_from_host(xhat_data.data(), xhat_data.size());
    saved_inv_std.copy_from_host(inv_std_data.data(), inv_std_data.size());
    return out;
}

inline Tensor tensor_group_norm_nchw_backward_input(
        const Tensor& g, const Tensor& xhat, const Tensor& inv_std,
        const Tensor& gamma, int num_groups) {
    const Shape& in_s = g.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);
    const int ch_per_g = C / num_groups;
    const int M = ch_per_g * H * W;

    Tensor d_in = Tensor::empty(in_s, g.device());
    if (d_in.elements() == 0) return d_in;

    std::vector<float> g_data(g.elements());
    std::vector<float> xhat_data(xhat.elements());
    std::vector<float> gamma_data(gamma.elements());
    std::vector<float> inv_std_data(inv_std.elements());
    std::vector<float> d_in_data(d_in.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    xhat.copy_to_host(xhat_data.data(), xhat_data.size());
    gamma.copy_to_host(gamma_data.data(), gamma_data.size());
    inv_std.copy_to_host(inv_std_data.data(), inv_std_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;

    for (int n = 0; n < N; ++n) {
        for (int grp = 0; grp < num_groups; ++grp) {
            const int ch0 = grp * ch_per_g;
            const float inv_std = inv_std_data[n * num_groups + grp];
            double sum_dyh = 0.0;
            double sum_dyh_x = 0.0;
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                const float gc = gamma_data[c];
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const int off = n * in_stride_n
                                      + c * in_stride_c
                                      + h * in_stride_h
                                      + w * in_stride_w;
                        const double dyh = static_cast<double>(
                            g_data[off]) * static_cast<double>(gc);
                        sum_dyh += dyh;
                        sum_dyh_x += dyh *
                            static_cast<double>(xhat_data[off]);
                    }
                }
            }
            const double inv_M = 1.0 / static_cast<double>(M);
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                const float gc = gamma_data[c];
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const int off = n * in_stride_n
                                      + c * in_stride_c
                                      + h * in_stride_h
                                      + w * in_stride_w;
                        const double dyh = static_cast<double>(
                            g_data[off]) * static_cast<double>(gc);
                        const double xn = static_cast<double>(
                            xhat_data[off]);
                        const double v = dyh
                            - inv_M * (sum_dyh + xn * sum_dyh_x);
                        d_in_data[off] = static_cast<float>(
                            inv_std * v);
                    }
                }
            }
        }
    }
    d_in.copy_from_host(d_in_data.data(), d_in_data.size());
    return d_in;
}

inline Tensor tensor_group_norm_nchw_backward_weight(
        const Tensor& g, const Tensor& xhat) {
    const Shape& in_s = g.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);

    Tensor d_gamma = Tensor::zeros(Shape({C}), g.device());
    if (d_gamma.elements() == 0) return d_gamma;

    std::vector<float> g_data(g.elements());
    std::vector<float> xhat_data(xhat.elements());
    std::vector<float> dg_data(d_gamma.elements());
    g.copy_to_host(g_data.data(), g_data.size());
    xhat.copy_to_host(xhat_data.data(), xhat_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;

    for (int c = 0; c < C; ++c) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const int off = n * in_stride_n
                                  + c * in_stride_c
                                  + h * in_stride_h
                                  + w * in_stride_w;
                    s += static_cast<double>(g_data[off]) *
                         static_cast<double>(xhat_data[off]);
                }
            }
        }
        dg_data[c] = static_cast<float>(s);
    }
    d_gamma.copy_from_host(dg_data.data(), dg_data.size());
    return d_gamma;
}

inline Tensor tensor_group_norm_nchw_backward_bias(const Tensor& g) {
    const Shape& in_s = g.shape();
    const int N = static_cast<int>(in_s[0]);
    const int C = static_cast<int>(in_s[1]);
    const int H = static_cast<int>(in_s[2]);
    const int W = static_cast<int>(in_s[3]);

    Tensor d_beta = Tensor::zeros(Shape({C}), g.device());
    if (d_beta.elements() == 0) return d_beta;

    std::vector<float> g_data(g.elements());
    std::vector<float> db_data(d_beta.elements());
    g.copy_to_host(g_data.data(), g_data.size());

    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;

    for (int c = 0; c < C; ++c) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const int off = n * in_stride_n
                                  + c * in_stride_c
                                  + h * in_stride_h
                                  + w * in_stride_w;
                    s += static_cast<double>(g_data[off]);
                }
            }
        }
        db_data[c] = static_cast<float>(s);
    }
    d_beta.copy_from_host(db_data.data(), db_data.size());
    return d_beta;
}

// ── 2-D DFT over the final two axes (CPU reference) ──────────────────
//
// Direct O((HW)^2) DFT applied independently to the final two axes of
// the input for every leading batch coordinate. Storage is row-major,
// last-axis contiguous. The DFT is linear, so batched outputs equal
// the element-wise DFT of each (H, W) plane.
//
// `inverse=false` is the unscaled forward DFT
// (exp(-i * 2 * pi * ...)); `inverse=true` is the inverse DFT
// (exp(+i * 2 * pi * ...)) which the caller may further scale by
// 1/(H*W). The same kernel is used for the forward pass and for the
// backward adjoint (with inverse and scale_output swapped); see
// src/core/fft.cpp.
struct TensorDFT2Result {
    Tensor real;
    Tensor imag;
};

inline TensorDFT2Result tensor_dft2_last2(
        const Tensor& real_in,
        const Tensor& imag_in,
        bool inverse,
        bool scale_output) {
    const Shape& s = real_in.shape();
    if (s != imag_in.shape()) {
        std::ostringstream os;
        os << "tensor_dft2_last2: real/imag shape mismatch ("
           << s << " vs " << imag_in.shape() << ")";
        throw std::invalid_argument(os.str());
    }
    if (real_in.device() != imag_in.device()) {
        std::ostringstream os;
        os << "tensor_dft2_last2: real/imag device mismatch";
        throw std::invalid_argument(os.str());
    }
    const int rank = static_cast<int>(s.rank());
    if (rank < 2) {
        std::ostringstream os;
        os << "tensor_dft2_last2: input must have rank >= 2 (got "
           << s << ")";
        throw std::invalid_argument(os.str());
    }
    const int H = static_cast<int>(s[rank - 2]);
    const int W = static_cast<int>(s[rank - 1]);
    if (H <= 0 || W <= 0) {
        std::ostringstream os;
        os << "tensor_dft2_last2: last two dimensions must be "
              "positive (got " << H << " x " << W << ")";
        throw std::invalid_argument(os.str());
    }
#ifdef AUTOGRAD_USE_CUDA
    if (real_in.device().is_cuda()) {
        const CudaTensorDFT2Result cuda_out = cuda_tensor_dft2_last2(
            real_in, imag_in, inverse, scale_output);
        TensorDFT2Result out{cuda_out.real, cuda_out.imag};
        return out;
    }
#endif
    const int64_t plane = static_cast<int64_t>(H) * W;
    const int64_t batch_count = s.numel() / plane;

    TensorDFT2Result out;
    out.real = Tensor::empty(s, real_in.device());
    out.imag = Tensor::empty(imag_in.shape(), imag_in.device());
    if (real_in.elements() == 0) return out;

    std::vector<float> r_in(real_in.elements());
    std::vector<float> i_in(imag_in.elements());
    std::vector<float> r_out(real_in.elements(), 0.f);
    std::vector<float> i_out(imag_in.elements(), 0.f);
    real_in.copy_to_host(r_in.data(), r_in.size());
    imag_in.copy_to_host(i_in.data(), i_in.size());

    const float norm = scale_output
        ? 1.f / static_cast<float>(H * W) : 1.f;

    for (int64_t b = 0; b < batch_count; ++b) {
        const float* rp = r_in.data() + b * plane;
        const float* ip = i_in.data() + b * plane;
        float* ro = r_out.data() + b * plane;
        float* io = i_out.data() + b * plane;
        for (int kr = 0; kr < H; ++kr) {
            for (int kc = 0; kc < W; ++kc) {
                float sum_r = 0.f;
                float sum_i = 0.f;
                for (int r = 0; r < H; ++r) {
                    for (int c = 0; c < W; ++c) {
                        const float angle = 2.f * kPi *
                            (static_cast<float>(kr * r) /
                                 static_cast<float>(H) +
                             static_cast<float>(kc * c) /
                                 static_cast<float>(W));
                        const float wr = std::cos(angle);
                        const float wi = inverse
                            ? std::sin(angle) : -std::sin(angle);
                        const float xr = rp[r * W + c];
                        const float xi = ip[r * W + c];
                        sum_r += xr * wr - xi * wi;
                        sum_i += xr * wi + xi * wr;
                    }
                }
                ro[kr * W + kc] = norm * sum_r;
                io[kr * W + kc] = norm * sum_i;
            }
        }
    }
    out.real.copy_from_host(r_out.data(), r_out.size());
    out.imag.copy_from_host(i_out.data(), i_out.size());
    return out;
}

}  // namespace detail
}  // namespace ag
