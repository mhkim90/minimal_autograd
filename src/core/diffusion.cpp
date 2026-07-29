// src/core/diffusion.cpp — diffusion-model primitives on the
// replacement Tensor/Variable API.
//
//   randn, randn_like            : Tensor-producing RNG helpers.
//   sinusoidal_time_embedding    : Tensor-producing host-int helper.
//   q_sample                     : differentiable composition of
//                                  existing public scale/add ops.
//
// All three reuse the existing public Tensor factories (Tensor::empty,
// Tensor::from_host) and existing Variable free functions (add,
// scale). No new OpKind is added; no kernel changes are needed.
//
// The helpers live in the transitional namespace `ag::diffusion`
// to keep the same signature space as the legacy `ag::diffusion.h`
// free functions (which expose `ag::sinusoidal_time_embedding(int,
// int)`). See include/autograd/core/diffusion.h for the rationale.

#include "autograd/core/diffusion.h"
#include "autograd/core/ops.h"
#include "autograd/core/variable.h"
#include "autograd/tensor.h"
#include "autograd/shape.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ag {
namespace diffusion {
namespace {

void validate_scalar_coefficient(const char* what, float v) {
    if (!std::isfinite(v)) {
        std::ostringstream os;
        os << what << ": must be finite (got " << v << ")";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

Tensor randn(const Shape& shape, uint32_t seed) {
    const int64_t n = shape.numel();
    if (n <= 0) {
        // Zero-element shape: produce an empty Tensor; the RNG is
        // never sampled.
        return Tensor::empty(shape);
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> values(static_cast<std::size_t>(n));
    for (auto& v : values) v = nd(rng);
    return Tensor::from_host(values.data(), shape);
}

Tensor randn_like(const Tensor& x, uint32_t seed) {
    if (x.device().is_cuda()) {
        throw std::runtime_error(
            "ag::diffusion::randn_like: CUDA tensors are not supported");
    }
    return randn(x.shape(), seed);
}

Tensor sinusoidal_time_embedding(int t, int dim) {
    if (dim <= 0) {
        std::ostringstream os;
        os << "ag::diffusion::sinusoidal_time_embedding: dim must be "
              "positive (got " << dim << ")";
        throw std::invalid_argument(os.str());
    }
    Tensor pe = Tensor::empty(Shape({dim}));
    std::vector<float> values(static_cast<std::size_t>(dim));
    // half_sin = ceil(dim / 2), half_cos = floor(dim / 2).
    // For dim=4 -> (2, 2); dim=5 -> (3, 2); dim=1 -> (1, 0).
    const int half_sin = (dim + 1) / 2;
    const int half_cos = dim / 2;
    const float t_f = static_cast<float>(t);
    const float log_10000 = std::log(10000.f);
    for (int i = 0; i < half_sin; ++i) {
        const float freq = std::exp(-log_10000 *
                                    (2.f * static_cast<float>(i) /
                                     static_cast<float>(dim)));
        values[i] = std::sin(t_f * freq);
    }
    for (int i = 0; i < half_cos; ++i) {
        const float freq = std::exp(-log_10000 *
                                    (2.f * static_cast<float>(i) /
                                     static_cast<float>(dim)));
        values[half_sin + i] = std::cos(t_f * freq);
    }
    pe.copy_from_host(values.data(), values.size());
    return pe;
}

Variable q_sample(const Variable& x0,
                  float sqrt_alpha_bar,
                  float sqrt_one_minus_alpha_bar,
                  const Variable* noise,
                  uint32_t seed) {
    validate_scalar_coefficient(
        "ag::diffusion::q_sample: sqrt_alpha_bar", sqrt_alpha_bar);
    validate_scalar_coefficient(
        "ag::diffusion::q_sample: sqrt_one_minus_alpha_bar",
        sqrt_one_minus_alpha_bar);
    if (x0.device().is_cuda() ||
        (noise != nullptr && noise->device().is_cuda())) {
        std::ostringstream os;
        os << "ag::diffusion::q_sample: CUDA tensors are not supported "
              "in this build";
        throw std::runtime_error(os.str());
    }
    if (noise == nullptr) {
        Variable drawn(randn_like(x0.value(), seed), false);
        return add(scale(x0, sqrt_alpha_bar),
                   scale(drawn, sqrt_one_minus_alpha_bar));
    }
    const Variable& nv = *noise;
    if (nv.value().shape() != x0.value().shape()) {
        std::ostringstream os;
        os << "ag::diffusion::q_sample: noise shape ("
           << nv.value().shape() << ") does not match x0 shape ("
           << x0.value().shape() << ")";
        throw std::invalid_argument(os.str());
    }
    return add(scale(x0, sqrt_alpha_bar),
               scale(nv, sqrt_one_minus_alpha_bar));
}

}  // namespace diffusion
}  // namespace ag
