#pragma once
// core/complex.h — minimal complex-valued wrapper over pairs of real
// ag::Variable components.
//
// The free functions real / imag / conj / complex_mul / complex_scale
// / abs2 / make_complex / real_to_complex overload the legacy
// ag::ComplexVar surface; this file uses ag::ComplexVariable. The
// real and imag accessors return ag::Variable by value (a cheap
// handle) so they are safe to bind to the result of a temporary.
//
// Header hygiene:
//   * No Eigen include.
//   * No CUDA runtime include.
//   * No public graph fields, raw pointers, or writable element refs.
//   * All forward ops reuse the existing public ag::core N-D free
//     ops (add, mul, scale, sub, etc.), so no new autograd nodes are
//     introduced for the trivial complex ops.

#include "autograd/core/ops.h"
#include "autograd/core/variable.h"
#include "autograd/device.h"
#include "autograd/shape.h"
#include "autograd/tensor.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ag {

struct ComplexVariable {
    Variable real;
    Variable imag;
};

inline void validate_complex_pair(const ComplexVariable& z, const char* op) {
    if (z.real.value().shape() != z.imag.value().shape()) {
        std::ostringstream os;
        os << op << ": real/imag shape mismatch ("
           << z.real.value().shape() << " vs " << z.imag.value().shape() << ")";
        throw std::invalid_argument(os.str());
    }
    if (z.real.device() != z.imag.device()) {
        std::ostringstream os;
        os << op << ": real/imag device mismatch ("
           << z.real.device() << " vs " << z.imag.device() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline void validate_same_complex_shape(const ComplexVariable& a,
                                        const ComplexVariable& b,
                                        const char* op) {
    validate_complex_pair(a, op);
    validate_complex_pair(b, op);
    if (a.real.value().shape() != b.real.value().shape()) {
        std::ostringstream os;
        os << op << ": shape mismatch ("
           << a.real.value().shape() << " vs " << b.real.value().shape() << ")";
        throw std::invalid_argument(os.str());
    }
}

inline ComplexVariable make_complex(Variable real_part, Variable imag_part) {
    ComplexVariable z{std::move(real_part), std::move(imag_part)};
    validate_complex_pair(z, "make_complex");
    return z;
}

inline ComplexVariable real_to_complex(Variable real_part) {
    const Tensor& src = real_part.value();
    Tensor zero = Tensor::zeros(src.shape(), src.device());
    Variable imag_part(std::move(zero), /*requires_grad=*/false);
    return make_complex(std::move(real_part), std::move(imag_part));
}

inline Variable real(const ComplexVariable& z) {
    validate_complex_pair(z, "real");
    return z.real;
}

inline Variable imag(const ComplexVariable& z) {
    validate_complex_pair(z, "imag");
    return z.imag;
}

inline ComplexVariable conj(const ComplexVariable& z) {
    validate_complex_pair(z, "conj");
    return make_complex(z.real, scale(z.imag, -1.f));
}

inline ComplexVariable complex_mul(const ComplexVariable& a,
                                   const ComplexVariable& b) {
    validate_same_complex_shape(a, b, "complex_mul");
    Variable rr = mul(a.real, b.real);
    Variable ii = mul(a.imag, b.imag);
    Variable ri = mul(a.real, b.imag);
    Variable ir = mul(a.imag, b.real);
    return make_complex(sub(rr, ii), add(ri, ir));
}

inline ComplexVariable complex_scale(const ComplexVariable& z, float s) {
    validate_complex_pair(z, "complex_scale");
    return make_complex(scale(z.real, s), scale(z.imag, s));
}

inline Variable abs2(const ComplexVariable& z) {
    validate_complex_pair(z, "abs2");
    return add(mul(z.real, z.real), mul(z.imag, z.imag));
}

}  // namespace ag