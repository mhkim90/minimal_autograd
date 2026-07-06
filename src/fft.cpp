#include "autograd/fft.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ag {
namespace {

constexpr float kPi = 3.14159265358979323846f;

struct ComplexMats {
    Mat real;
    Mat imag;
};

void validate_fft_input(const ComplexVar& z, const char* op) {
    validate_complex_pair(z, op);
    if (z.real->data.rows() <= 0 || z.real->data.cols() <= 0) {
        throw std::runtime_error(std::string(op) + ": empty input");
    }
}

ComplexMats dft2_reference(const Mat& real,
                           const Mat& imag,
                           bool inverse,
                           bool scale_output) {
    const int rows = real.rows();
    const int cols = real.cols();
    const float norm = scale_output ? 1.f / static_cast<float>(rows * cols) : 1.f;
    ComplexMats out{Mat::Zero(rows, cols), Mat::Zero(rows, cols)};

    for (int kr = 0; kr < rows; ++kr) {
        for (int kc = 0; kc < cols; ++kc) {
            float sum_r = 0.f;
            float sum_i = 0.f;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const float angle = 2.f * kPi *
                        (static_cast<float>(kr * r) / static_cast<float>(rows) +
                         static_cast<float>(kc * c) / static_cast<float>(cols));
                    const float wr = std::cos(angle);
                    const float wi = inverse ? std::sin(angle) : -std::sin(angle);
                    const float xr = real(r, c);
                    const float xi = imag(r, c);
                    sum_r += xr * wr - xi * wi;
                    sum_i += xr * wi + xi * wr;
                }
            }
            out.real(kr, kc) = norm * sum_r;
            out.imag(kr, kc) = norm * sum_i;
        }
    }
    return out;
}

ComplexMats fft_adjoint(const Mat& grad_real,
                        const Mat& grad_imag,
                        bool forward_was_inverse) {
    if (forward_was_inverse) {
        return dft2_reference(grad_real, grad_imag, false, true);
    }
    return dft2_reference(grad_real, grad_imag, true, false);
}

ComplexVar fft2_impl(const ComplexVar& z,
                     bool inverse,
                     FftNorm norm,
                     const char* op) {
    if (norm != FftNorm::Backward) {
        throw std::runtime_error(std::string(op) + ": unsupported normalization");
    }
    validate_fft_input(z, op);
#ifdef AUTOGRAD_USE_CUDA
    if (z.real->is_cuda() || z.imag->is_cuda()) {
        return detail::cuda_fft2_forward(z, inverse);
    }
#endif

    const auto out = dft2_reference(z.real->data, z.imag->data,
                                    inverse, inverse);
    auto out_real = Var::make(out.real);
    auto out_imag = Var::make(out.imag);
    out_real->set_shape(z.real->shape());
    out_imag->set_shape(z.imag->shape());
    out_real->parents = {z.real, z.imag};
    out_imag->parents = {z.real, z.imag};

    auto input_real = z.real;
    auto input_imag = z.imag;
    out_real->back_fn = [input_real, input_imag, inverse,
                         wp = std::weak_ptr<Var>(out_real)]() {
        auto self = wp.lock();
        if (!self) return;
        const Mat zero = Mat::Zero(self->grad.rows(), self->grad.cols());
        const auto grad = fft_adjoint(self->grad, zero, inverse);
        input_real->grad += grad.real;
        input_imag->grad += grad.imag;
    };
    out_imag->back_fn = [input_real, input_imag, inverse,
                         wp = std::weak_ptr<Var>(out_imag)]() {
        auto self = wp.lock();
        if (!self) return;
        const Mat zero = Mat::Zero(self->grad.rows(), self->grad.cols());
        const auto grad = fft_adjoint(zero, self->grad, inverse);
        input_real->grad += grad.real;
        input_imag->grad += grad.imag;
    };

    return make_complex(std::move(out_real), std::move(out_imag));
}

} // namespace

ComplexVar fft2(const ComplexVar& z, FftNorm norm) {
    return fft2_impl(z, false, norm, "fft2");
}

ComplexVar ifft2(const ComplexVar& z, FftNorm norm) {
    return fft2_impl(z, true, norm, "ifft2");
}

} // namespace ag
