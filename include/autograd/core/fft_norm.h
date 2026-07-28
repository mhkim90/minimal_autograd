#pragma once
// fft_norm.h — canonical Eigen-free normalization enum for FFT ops.
//
// This header is the single source of truth for the ag::FftNorm
// enumeration used by both the legacy ag::fft2 / ag::ifft2 free
// functions (in include/autograd/fft.h) and the replacement
// ag::fft2 / ag::ifft2 free functions (in include/autograd/core/fft.h).
// Both surfaces include this header so the enum type lives in exactly
// one place.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No dependency on Tensor / Variable / Shape / Device.
//
// Only the "backward" normalization (matching PyTorch's default) is
// currently supported: forward DFT is unscaled, inverse DFT is scaled
// by 1/(H*W). Other normalizations throw at runtime from the free
// functions; this matches the legacy behavior.

namespace ag {

enum class FftNorm {
    Backward,
};

}  // namespace ag