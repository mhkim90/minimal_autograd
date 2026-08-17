// src/core/fft.cpp — 2-D complex FFT on the Tensor/Variable API.
//
// Implements ag::fft2 / ag::ifft2 declared in autograd/core/fft.h.
// Forward and adjoint use the same private detail::tensor_dft2_last2
// kernel in src/detail/tensor_ops.h, with inverse and scale_output
// swapped for the adjoint. Each output component (real, imag) is a
// custom-op node built through the public ag::make_custom_variable
// boundary, so no new OpKind or speculative graph fields are
// introduced; repeated / shared branches accumulate gradients through
// the existing custom-op contract.
//
// Shape / device contract:
//   * rank >= 2; the last two axes are the (H, W) plane the DFT acts
//     on; every leading axis is a batch dimension. Output shape
//     equals input shape.
//   * H and W must be positive integers; rank < 2 or zero H/W are
//     rejected with std::invalid_argument.
//   * Real and imag must share shape and device.
//   * CPU and CUDA devices use the replacement Tensor kernels directly;
//     CUDA support is not silently forwarded to the legacy path.
//
// Normalization: FftNorm::Backward only. Forward is the unscaled DFT,
// inverse is the inverse DFT scaled by 1/(H*W), so composing them is
// the identity on the input. Any other normalization throws
// std::runtime_error.

#include "autograd/core/complex.h"
#include "autograd/core/fft.h"
#include "autograd/core/fft_norm.h"
#include "autograd/extension/custom_op.h"
#include "autograd/tensor.h"

#include "detail/tensor_ops.h"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ag {
namespace {

void validate_fft_input(const ComplexVariable& z, const char* op) {
    validate_complex_pair(z, op);
    const Shape& s = z.real.value().shape();
    if (s.rank() < 2) {
        std::ostringstream os;
        os << op << ": input must have rank >= 2 (got " << s << ")";
        throw std::invalid_argument(os.str());
    }
    const int64_t H = s[s.rank() - 2];
    const int64_t W = s[s.rank() - 1];
    if (H <= 0 || W <= 0) {
        std::ostringstream os;
        os << op << ": last two dimensions must be positive (got "
           << H << " x " << W << ")";
        throw std::invalid_argument(os.str());
    }
}

std::pair<Variable, Variable> fft_impl_variables(
        const ComplexVariable& z,
        Tensor out_real,
        Tensor out_imag,
        bool forward_was_inverse) {
    const Device dev = out_real.device();

    // Backward for each output component reads that component's
    // upstream gradient and fills the other slot with zeros, then
    // calls the same DFT kernel with swapped flags. This avoids a
    // separate adjoint implementation.
    auto real_backward =
        [forward_was_inverse, dev](
        const Tensor& upstream) -> std::vector<Tensor> {
        Tensor zero = detail::tensor_zeros(upstream.shape(), dev);
        auto adj = detail::tensor_dft2_last2(
            upstream, zero,
            /*inverse=*/!forward_was_inverse,
            /*scale_output=*/forward_was_inverse);
        return {std::move(adj.real), std::move(adj.imag)};
    };
    auto imag_backward =
        [forward_was_inverse, dev](
        const Tensor& upstream) -> std::vector<Tensor> {
        Tensor zero = detail::tensor_zeros(upstream.shape(), dev);
        auto adj = detail::tensor_dft2_last2(
            zero, upstream,
            /*inverse=*/!forward_was_inverse,
            /*scale_output=*/forward_was_inverse);
        return {std::move(adj.real), std::move(adj.imag)};
    };

    Variable out_real_var = make_custom_variable(
        std::move(out_real), {z.real, z.imag}, std::move(real_backward));
    Variable out_imag_var = make_custom_variable(
        std::move(out_imag), {z.real, z.imag}, std::move(imag_backward));
    return {std::move(out_real_var), std::move(out_imag_var)};
}

ComplexVariable fft_impl(const ComplexVariable& z,
                          bool inverse,
                          FftNorm norm,
                          const char* op) {
    if (norm != FftNorm::Backward) {
        std::ostringstream os;
        os << op << ": unsupported normalization (only FftNorm::Backward)";
        throw std::runtime_error(os.str());
    }
    validate_fft_input(z, op);

    auto fwd = detail::tensor_dft2_last2(
        z.real.value(), z.imag.value(),
        /*inverse=*/inverse,
        /*scale_output=*/inverse);
    auto vars = fft_impl_variables(z, std::move(fwd.real), std::move(fwd.imag),
                                    inverse);
    return make_complex(std::move(vars.first), std::move(vars.second));
}

}  // namespace

ComplexVariable fft2(const ComplexVariable& z, FftNorm norm) {
    return fft_impl(z, /*inverse=*/false, norm, "fft2");
}

ComplexVariable ifft2(const ComplexVariable& z, FftNorm norm) {
    return fft_impl(z, /*inverse=*/true, norm, "ifft2");
}

}  // namespace ag
