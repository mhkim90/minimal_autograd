#pragma once
// Numerical gradient checks on public Tensor storage and fresh Variable graphs.
#include "autograd/core/ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>

namespace ag_test {
inline std::vector<float> host(const ag::Tensor& t) {
    std::vector<float> out(t.elements());
    t.copy_to_host(out.data(), out.size());
    return out;
}
inline ag::Tensor tensor(const ag::Shape& shape, const std::vector<float>& data) {
    if (data.size() != static_cast<size_t>(shape.numel()))
        throw std::invalid_argument("test tensor size mismatch");
    return ag::Tensor::from_host(data.data(), shape);
}
inline ag::Tensor filled(const ag::Shape& shape, float value) {
    return tensor(shape, std::vector<float>(shape.numel(), value));
}
inline ag::Tensor random_tensor(const ag::Shape& shape, float scale = 1.f,
                                float positive_offset = -1.f) {
    std::vector<float> data(shape.numel());
    for (float& v : data) {
        v = 2.f * static_cast<float>(std::rand()) / RAND_MAX - 1.f;
        v = (positive_offset >= 0.f ? std::fabs(v) + positive_offset : v) * scale;
    }
    return tensor(shape, data);
}
inline float at(const ag::Tensor& t, size_t index = 0) { return host(t).at(index); }
// Inspect a sample's flattened row-major slab (also useful for NCHW fixtures).
inline float at(const ag::Tensor& t, int row, int col) {
    const size_t width = t.shape().rank() == 0 ? 1 : t.elements() / t.shape()[0];
    return at(t, row * width + col);
}
inline void put(ag::Tensor t, size_t index, float value) {
    auto data = host(t); data.at(index) = value;
    t.copy_from_host(data.data(), data.size());
}
inline float max_abs(const ag::Tensor& t) {
    float out = 0.f;
    for (float v : host(t)) {
        if (!std::isfinite(v)) return NAN;
        out = std::max(out, std::fabs(v));
    }
    return out;
}
inline float max_diff(const ag::Tensor& a, const ag::Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("test shape mismatch");
    auto av = host(a), bv = host(b); float out = 0.f;
    for (size_t i = 0; i < av.size(); ++i) {
        if (!std::isfinite(av[i]) || !std::isfinite(bv[i])) return INFINITY;
        out = std::max(out, std::fabs(av[i] - bv[i]));
    }
    return out;
}
inline float host_mean(const ag::Tensor& t) {
    float out = 0.f; for (float v : host(t)) out += v;
    return out / t.elements();
}
inline float host_variance(const ag::Tensor& t, float center) {
    float out = 0.f; for (float v : host(t)) out += (v-center)*(v-center);
    return out / t.elements();
}
inline float min_value(const ag::Tensor& t) {
    auto v = host(t); return *std::min_element(v.begin(), v.end());
}
inline float max_value(const ag::Tensor& t) {
    auto v = host(t); return *std::max_element(v.begin(), v.end());
}
inline float row_sum(const ag::Tensor& t, int row) {
    auto v = host(t); float out = 0.f;
    for (int j = 0; j < t.shape()[1]; ++j) out += v[row*t.shape()[1]+j];
    return out;
}
// Install an upstream gradient through the public backward contract, not fields.
inline void seed_grad(ag::Variable v, const ag::Tensor& gradient) {
    v.zero_grad(); v.backward(gradient.to(v.device()));
}
}  // namespace ag_test

inline bool grad_check(std::function<ag::Variable(ag::Variable)> f,
                       ag::Variable x, float eps = 1e-3f, float tol = 5e-2f) {
    x.zero_grad();
    ag::sum(f(x)).backward();
    const auto analytic = ag_test::host(x.grad());
    auto data = ag_test::host(x.value());
    ag::Tensor value = x.value();  // shared storage, no graph adapter
    float diff = 0.f;
    for (size_t i = 0; i < data.size(); ++i) {
        const float orig = data[i];
        float plus = 0.f, minus = 0.f;
        try {
            data[i] = orig + eps; value.copy_from_host(data.data(), data.size());
            plus = ag_test::at(ag::sum(f(ag::Variable(value, true))).value());
            data[i] = orig - eps; value.copy_from_host(data.data(), data.size());
            minus = ag_test::at(ag::sum(f(ag::Variable(value, true))).value());
        } catch (...) {
            data[i] = orig; value.copy_from_host(data.data(), data.size());
            throw;
        }
        data[i] = orig; value.copy_from_host(data.data(), data.size());
        const float error = std::fabs(analytic[i] - (plus-minus)/(2.f*eps));
        if (!std::isfinite(error)) return false;
        diff = std::max(diff, error);
    }
    if (diff > tol) std::printf("    grad_check FAIL: max abs diff = %g (tol %g)\n", diff, tol);
    return diff <= tol;
}

inline bool grad_check_multi(
        std::function<ag::Variable(std::vector<ag::Variable>)> f,
        std::vector<ag::Variable> xs, float eps = 1e-3f, float tol = 5e-2f) {
    for (auto& x : xs) x.zero_grad();
    ag::sum(f(xs)).backward();
    std::vector<std::vector<float>> analytic;
    for (auto& x : xs) analytic.push_back(ag_test::host(x.grad()));
    for (size_t vi = 0; vi < xs.size(); ++vi) {
        ag::Tensor value = xs[vi].value();
        auto data = ag_test::host(value); float diff = 0.f;
        auto probe = [&]() {
            std::vector<ag::Variable> fresh;
            for (auto& x : xs) fresh.emplace_back(x.value(), true);
            return ag_test::at(ag::sum(f(fresh)).value());
        };
        for (size_t i = 0; i < data.size(); ++i) {
            const float orig = data[i];
            float plus = 0.f, minus = 0.f;
            try {
                data[i] = orig + eps; value.copy_from_host(data.data(), data.size());
                plus = probe();
                data[i] = orig - eps; value.copy_from_host(data.data(), data.size());
                minus = probe();
            } catch (...) {
                data[i] = orig; value.copy_from_host(data.data(), data.size());
                throw;
            }
            data[i] = orig; value.copy_from_host(data.data(), data.size());
            const float error = std::fabs(analytic[vi][i] - (plus-minus)/(2.f*eps));
            if (!std::isfinite(error)) return false;
            diff = std::max(diff, error);
        }
        if (diff > tol) {
            std::printf("    grad_check_multi FAIL on input %zu: diff = %g\n", vi, diff);
            return false;
        }
    }
    return true;
}
