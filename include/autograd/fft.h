#pragma once
// fft.h — differentiable 2-D complex FFT API.

#include "autograd/complex.h"

namespace ag {

enum class FftNorm {
    Backward
};

ComplexVar fft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);
ComplexVar ifft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);

#ifdef AUTOGRAD_USE_CUDA
namespace detail {
ComplexVar cuda_fft2_forward(const ComplexVar& z, bool inverse);
}
#endif

} // namespace ag
