#include "detail/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ag {
namespace detail {
namespace cpu_ops {

namespace {

inline std::vector<int64_t> contiguous_strides(const Shape& s) {
    return contiguous_stride(s).strides;
}

}  // namespace

Tensor tensor_ones(const Shape& shape, Device device) {
    return Tensor::ones(shape, device);
}

Tensor tensor_softmax_nd(const Tensor& a, int axis, Tensor& saved_softmax) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = axis;
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

Tensor tensor_softmax_backward_nd(const Tensor& g,
                                  const Tensor& saved_softmax,
                                  int axis) {
    const Shape& s = g.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = axis;
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

Tensor tensor_log_softmax_nd(const Tensor& a, int axis, Tensor& saved_lsm) {
    const Shape& s = a.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = axis;
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

Tensor tensor_log_softmax_backward_nd(const Tensor& g,
                                      const Tensor& saved_lsm,
                                      int axis) {
    const Shape& s = g.shape();
    const int rank = static_cast<int>(s.rank());
    const int ax = axis;
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

Tensor tensor_add(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] + bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_mul(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), bv(n), ov(n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_scale(const Tensor& a, float scalar) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), ov(n);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * scalar;
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_relu(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = av[i] > 0.f ? av[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_relu_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = sv[i] > 0.f ? gv[i] : 0.f;
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sigmoid(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = 1.f / (1.f + std::exp(-av[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sigmoid_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i] * (1.f - sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_tanh(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::tanh(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_tanh_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (1.f - sv[i] * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_exp(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::exp(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_exp_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_log(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::log(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_log_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / sv[i];
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sqrt(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = std::sqrt(av[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sqrt_backward(const Tensor& g, const Tensor& saved) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    saved.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (2.f * sv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_silu_forward(const Tensor& a, Tensor& sigmoid_out) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    sigmoid_out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), so(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        so[i] = 1.f / (1.f + std::exp(-av[i]));
        ov[i] = av[i] * so[i];
    }
    out.copy_from_host(ov.data(), n);
    sigmoid_out.copy_from_host(so.data(), n);
    return out;
}

Tensor tensor_silu_backward(const Tensor& g,
                            const Tensor& x,
                            const Tensor& sig) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), sv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    sig.copy_to_host(sv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] * (sv[i] + xv[i] * sv[i] * (1.f - sv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_softplus(const Tensor& a) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = av[i];
        const float ax = std::fabs(x);
        ov[i] = std::max(x, 0.f) + std::log(1.f + std::exp(-ax));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_softplus_backward(const Tensor& g, const Tensor& x) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), xv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    x.copy_to_host(xv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = gv[i] / (1.f + std::exp(-xv[i]));
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sub(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] - bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_sub_backward_a(const Tensor& g) {
    return g.clone();
}

Tensor tensor_sub_backward_b(const Tensor& g) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = -gv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div_backward_a(const Tensor& g, const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = gv[i] / bv[i];
    out.copy_from_host(ov.data(), n);
    return out;
}

Tensor tensor_div_backward_b(const Tensor& g,
                             const Tensor& a,
                             const Tensor& b) {
    Tensor out = Tensor::empty(g.shape(), g.device());
    const std::size_t n = g.elements();
    if (n == 0) return out;
    std::vector<float> gv(n, 0.f), av(n, 0.f), bv(n, 0.f), ov(n, 0.f);
    g.copy_to_host(gv.data(), n);
    a.copy_to_host(av.data(), n);
    b.copy_to_host(bv.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = -(gv[i] * av[i]) / (bv[i] * bv[i]);
    }
    out.copy_from_host(ov.data(), n);
    return out;
}

}  // namespace cpu_ops
}  // namespace detail
}  // namespace ag
