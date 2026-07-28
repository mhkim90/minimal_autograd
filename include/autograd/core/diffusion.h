#pragma once
// diffusion.h — diffusion-model primitives on the replacement
// Tensor/Variable API.
//
// The helpers live in the transitional namespace `ag::diffusion`
// to avoid signature collisions with the legacy `ag::diffusion.h`
// free functions (which expose `ag::sinusoidal_time_embedding(int,
// int)`). Until the legacy facade is retired, callers of the
// replacement surface must write `ag::diffusion::randn(...)` etc.,
// mirroring the `ag::nn::*` and `ag::optim::*` collision strategy.
// The flat `ag::*` aliases will be re-introduced at the end of the
// refactor.
//
// The helpers split into two groups:
//
//   * Tensor-producing utilities (randn, randn_like,
//     sinusoidal_time_embedding). These never carry an autograd
//     graph: randn draws fresh std-normal samples that are noise
//     (no gradient to propagate), and the time embedding is built
//     from a host-int timestep so there is nothing to differentiate.
//
//   * A differentiable op (q_sample) composed entirely from existing
//     public ops (scale, add). No new OpKind is introduced.
//
// API contract notes:
//   * randn takes an explicit Shape and an explicit seed. The RNG is
//     constructed per call from the seed, so there is no global
//     mutable RNG state. The default seed (0) produces a fixed
//     deterministic sequence; passing the same seed to two
//     successive calls produces identical values.
//   * sinusoidal_time_embedding takes a host-int timestep t and a
//     positive int dim. Output is a rank-1 Tensor of shape (dim,)
//     with sin components in [0, half_sin) and cos components in
//     [half_sin, dim), where half_sin = (dim + 1) / 2. Odd dim is
//     handled safely: the total element count is always dim.
//   * q_sample is rank-agnostic. x0 and noise can be any rank
//     (including rank-0 / rank-1) and zero-element shapes, as long
//     as they share the same shape. Scalar coefficients broadcast
//     naturally through the underlying scale/add ops.

#include "autograd/core/variable.h"

#include <cstdint>

namespace ag {
namespace diffusion {

// ── Random noise ──────────────────────────────────────────────────────

// randn(shape, seed=0): returns a Tensor of standard-normal samples
// with the requested Shape. The RNG is constructed locally per call
// from `seed`; the default seed (0) is fixed and deterministic.
// A zero-element shape produces an empty Tensor without sampling.
Tensor randn(const Shape& shape, uint32_t seed = 0);

// randn_like(x, seed=0): same shape as the source Tensor, independent
// samples. Wraps randn(source.shape(), seed).
Tensor randn_like(const Tensor& x, uint32_t seed = 0);

// ── Time embedding ────────────────────────────────────────────────────

// sinusoidal_time_embedding(t, dim): returns a rank-1 Tensor of
// shape (dim,) holding the standard transformer-style positional
// encoding at integer timestep t. Output values lie in [-1, 1].
// `dim` must be positive; if dim == 1 only a sin component is
// produced. The output is not differentiable: `t` is a host int.
Tensor sinusoidal_time_embedding(int t, int dim);

// ── DDPM forward ──────────────────────────────────────────────────────

// q_sample(x0, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise=nullptr,
//          seed=0): standard DDPM forward diffusion step
//
//     x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
//
// Rank-agnostic: x0 and noise can be any rank (rank-0 / rank-1 /
// higher) and zero-element shapes, as long as they share the same
// shape. The result is a Variable whose backward propagates through
// both x0 and noise via the existing scale/add ops. When `noise` is
// null a fresh randn sample of x0's shape is drawn using `seed`
// (the default seed=0 produces a fixed deterministic sequence).
//
// Coefficients must be finite real numbers (NaN / +-inf rejected).
// Mismatched shapes raise std::invalid_argument.
Variable q_sample(const Variable& x0,
                  float sqrt_alpha_bar,
                  float sqrt_one_minus_alpha_bar,
                  const Variable* noise = nullptr,
                  uint32_t seed = 0);

}  // namespace diffusion
}  // namespace ag
