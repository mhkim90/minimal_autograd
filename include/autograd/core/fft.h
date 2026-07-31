#pragma once
// core/fft.h — 2-D complex FFT API on the Tensor/Variable surface.
//
// ag::fft2 / ag::ifft2 overload the legacy free functions in
// include/autograd/fft.h by parameter type: the legacy surface takes
// ag::ComplexVar, this surface takes ag::ComplexVariable. Both share
// the canonical ag::FftNorm enum from include/autograd/core/fft_norm.h.
//
// Shape / device contract:
//   * fft2 / ifft2 accept any rank >= 2 ComplexVariable; the 2-D
//     complex DFT is applied independently to the final two axes for
//     every leading batch coordinate.
//   * Last two axes must be positive. Rank-1 and zero-extent inputs
//     are rejected with std::invalid_argument.
//   * CPU and CUDA devices are supported; mixed real/imag devices are
//     rejected with std::invalid_argument.
//   * Only FftNorm::Backward is supported; any other value throws
//     std::runtime_error. Backward normalization: forward is the
//     unscaled DFT, inverse is the inverse DFT scaled by 1/(H*W).
//   * Each output component accumulates gradients to both input
//     components (real and imag) through ag::make_custom_variable,
//     so repeated / shared branches accumulate gradients correctly
//     under the existing custom-op backward contract.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No public graph fields, raw pointers, or mutable element refs.

#include "autograd/core/complex.h"
#include "autograd/core/fft_norm.h"

namespace ag {

// Forward 2-D complex FFT (backward normalization, unscaled DFT).
// Returns a ComplexVariable whose real and imag components are new
// graph nodes with both real/imag of `z` as parents; backward
// accumulates into both parent components.
ComplexVariable fft2(const ComplexVariable& z,
                     FftNorm norm = FftNorm::Backward);

// Inverse 2-D complex FFT (backward normalization, scaled by 1/(H*W)).
// The same shape and parent contract as fft2 above.
ComplexVariable ifft2(const ComplexVariable& z,
                      FftNorm norm = FftNorm::Backward);

}  // namespace ag
