#pragma once
// fft.h — legacy differentiable 2-D complex FFT API on Var/Mat.
//
// FftNorm is shared with the replacement surface through the
// canonical Eigen-free header autograd/core/fft_norm.h.

#include "autograd/complex.h"
#include "autograd/core/fft_norm.h"

namespace ag {

ComplexVar fft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);
ComplexVar ifft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);

#ifdef AUTOGRAD_USE_CUDA
namespace detail {
ComplexVar cuda_fft2_forward(const ComplexVar& z, bool inverse);
}
#endif

} // namespace ag
