#include "detail/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ag {
namespace detail {
namespace cpu_ops {

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
