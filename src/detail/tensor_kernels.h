#pragma once
// Private CPU tensor arithmetic for the autograd vertical slice.

#include "autograd/tensor.h"

#include <cstddef>
#include <vector>

namespace ag {
namespace detail {

inline Tensor tensor_add(const Tensor& a, const Tensor& b) {
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

inline Tensor tensor_mul(const Tensor& a, const Tensor& b) {
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

inline Tensor tensor_scale(const Tensor& a, float s) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::size_t n = a.elements();
    if (n == 0) return out;
    std::vector<float> av(n), ov(n);
    a.copy_to_host(av.data(), n);
    for (std::size_t i = 0; i < n; ++i) ov[i] = av[i] * s;
    out.copy_from_host(ov.data(), n);
    return out;
}

inline Tensor tensor_sum(const Tensor& a) {
    Tensor out = Tensor::empty(Shape{}, a.device());
    if (a.elements() == 0) {
        float z = 0.f;
        out.copy_from_host(&z, 1);
        return out;
    }
    std::vector<float> av(a.elements());
    a.copy_to_host(av.data(), av.size());
    float s = 0.f;
    for (float v : av) s += v;
    out.copy_from_host(&s, 1);
    return out;
}

// Broadcast a scalar to a target shape (used by sum's backward).
inline Tensor tensor_broadcast_scalar(const Tensor& scalar,
                                      const Shape& target) {
    Tensor out = Tensor::empty(target, scalar.device());
    if (out.elements() == 0) return out;
    std::vector<float> seed(1);
    scalar.copy_to_host(seed.data(), 1);
    std::vector<float> buf(out.elements(), seed[0]);
    out.copy_from_host(buf.data(), buf.size());
    return out;
}

}  // namespace detail
}  // namespace ag
