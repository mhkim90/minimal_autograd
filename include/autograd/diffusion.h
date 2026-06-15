#pragma once
// diffusion.h — diffusion-model primitives on top of the autograd engine.
//
// These helpers are intentionally minimal and compose with the existing
// differentiable ops (add / mul / scale / cumsum / silu). They exist to
// (1) generate random noise tensors as leaf Vars without going through
// the autograd graph, and (2) provide the standard building blocks every
// DDPM / DDIM / score-based model needs:
//
//   randn, randn_like, sinusoidal_time_embedding, q_sample
//
// No new Function subclass is required — the differentiable pieces are
// plain compositions; the non-differentiable ones (RNG, time embedding
// from a host int) are leaf-producing helpers.

#include "autograd/tensor.h"
#include "autograd/variable.h"
#include "autograd/ops.h"
#include <cmath>
#include <cstdint>
#include <random>

namespace ag {

// ── Random noise (leaf Vars, no back_fn) ───────────────────────────────────

// randn(r, c, seed=0) — draw (r, c) standard normal samples using
// std::mt19937 seeded with `seed`. Returns a leaf Var (no parents,
// no back_fn, no grad contribution on backward).
inline VarPtr randn(int rows, int cols, uint32_t seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    Mat m(rows, cols);
    for (int i = 0; i < m.size(); ++i) *(m.data() + i) = nd(rng);
    return Var::make(std::move(m));
}

// randn_like(x) — same shape as x, independent std-normal samples.
inline VarPtr randn_like(VarPtr x, uint32_t seed = 0) {
    return randn(x->data.rows(), x->data.cols(), seed);
}

// ── Sinusoidal time embedding ──────────────────────────────────────────────

// sinusoidal_time_embedding(t, dim) — host-int timestep t → Var of shape
// (1, dim) using the standard transformer-style positional encoding:
//
//   pe[i]      = sin(t * freq_i)  for i in [0, dim/2)
//   pe[i+dim/2]= cos(t * freq_i)  for i in [0, dim/2)
//   freq_i     = exp(-log(10000) * (2*i / dim))
//
// Mirrors torch.nn.Embedding / positional encoding used in every diffusion
// U-Net (DDPM, Stable Diffusion, GUDM, etc.). `dim` must be even.
//
// Leaf Var — no back_fn. `t` is not differentiated; pass an int.
inline VarPtr sinusoidal_time_embedding(int t, int dim) {
    assert(dim % 2 == 0 && "sinusoidal_time_embedding: dim must be even");
    int half = dim / 2;
    Mat pe(1, dim);
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(10000.f) * (2.f * i / dim));
        pe(0, i)        = std::sin(static_cast<float>(t) * freq);
        pe(0, i + half) = std::cos(static_cast<float>(t) * freq);
    }
    return Var::make(std::move(pe));
}

// ── DDPM forward (q_sample) ────────────────────────────────────────────────

// q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise, seed=0)
//
// Standard DDPM forward diffusion step:
//   x_t = sqrt_alpha_bar(t) * x0
//       + sqrt_one_minus_alpha_bar(t) * noise
//
// where `noise` is std-normal of the same shape as x0. If `noise` is null,
// a fresh randn_like(x0) is drawn with the given seed.
//
//   x0:                VarPtr (N, *)
//   t:                 int — single timestep applied to the whole batch
//   sqrt_alpha_bar:    host float scalar
//   sqrt_one_minus_ab: host float scalar
//   noise:             optional VarPtr; if null we draw fresh noise
//   seed:              RNG seed used only when noise is null
//
// Returns Var of the same shape as x0. Differentiated w.r.t. x0 and noise
// via the existing add / scale chain — no new Function subclass required.
inline VarPtr q_sample(VarPtr x0, int /*t*/,
                       float sqrt_alpha_bar,
                       float sqrt_one_minus_alpha_bar,
                       VarPtr noise = nullptr,
                       uint32_t seed = 0) {
    if (noise == nullptr) noise = randn_like(x0, seed);
    VarPtr a = scale(x0, sqrt_alpha_bar);
    VarPtr b = scale(noise, sqrt_one_minus_alpha_bar);
    return add(a, b);
}

} // namespace ag
