#pragma once
// complex.h — small complex-valued wrapper over pairs of real Var nodes.

#include "autograd/ops.h"
#include "autograd/variable.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ag {

struct ComplexVar {
    VarPtr real;
    VarPtr imag;
};

inline void validate_complex_pair(const ComplexVar& z, const char* op) {
    if (!z.real || !z.imag) {
        throw std::runtime_error(std::string(op) + ": null complex component");
    }
    if (z.real->shape() != z.imag->shape() ||
        z.real->data.rows() != z.imag->data.rows() ||
        z.real->data.cols() != z.imag->data.cols()) {
        throw std::runtime_error(std::string(op) + ": real/imag shape mismatch");
    }
#ifdef AUTOGRAD_USE_CUDA
    if (z.real->is_cuda() != z.imag->is_cuda()) {
        throw std::runtime_error(std::string(op) + ": mixed CPU/CUDA complex pair");
    }
    if (z.real->is_cuda() && z.real->cuda_device() != z.imag->cuda_device()) {
        throw std::runtime_error(std::string(op) + ": real/imag device mismatch");
    }
#endif
}

inline void validate_same_complex_shape(const ComplexVar& a,
                                        const ComplexVar& b,
                                        const char* op) {
    validate_complex_pair(a, op);
    validate_complex_pair(b, op);
    if (a.real->shape() != b.real->shape() ||
        a.real->data.rows() != b.real->data.rows() ||
        a.real->data.cols() != b.real->data.cols()) {
        throw std::runtime_error(std::string(op) + ": shape mismatch");
    }
}

inline ComplexVar make_complex(VarPtr real_part, VarPtr imag_part) {
    ComplexVar z{std::move(real_part), std::move(imag_part)};
    validate_complex_pair(z, "make_complex");
    return z;
}

inline ComplexVar real_to_complex(VarPtr real_part) {
    if (!real_part) {
        throw std::runtime_error("real_to_complex: null input");
    }
#ifdef AUTOGRAD_USE_CUDA
    if (real_part->is_cuda()) {
        auto imag_part = Var::make(Mat::Zero(real_part->data.rows(), real_part->data.cols()))
            ->cuda(real_part->cuda_device());
        imag_part->set_shape(real_part->shape());
        return make_complex(std::move(real_part), imag_part);
    }
#endif
    auto imag_part = Var::make(Mat::Zero(real_part->data.rows(), real_part->data.cols()));
    imag_part->set_shape(real_part->shape());
    return make_complex(std::move(real_part), imag_part);
}

inline VarPtr real(const ComplexVar& z) {
    validate_complex_pair(z, "real");
    return z.real;
}

inline VarPtr imag(const ComplexVar& z) {
    validate_complex_pair(z, "imag");
    return z.imag;
}

inline ComplexVar conj(const ComplexVar& z) {
    validate_complex_pair(z, "conj");
    return make_complex(z.real, scale(z.imag, -1.f));
}

inline ComplexVar complex_mul(const ComplexVar& a, const ComplexVar& b) {
    validate_same_complex_shape(a, b, "complex_mul");
    auto rr = mul(a.real, b.real);
    auto ii = mul(a.imag, b.imag);
    auto ri = mul(a.real, b.imag);
    auto ir = mul(a.imag, b.real);
    return make_complex(sub(rr, ii), add(ri, ir));
}

inline ComplexVar complex_scale(const ComplexVar& z, float s) {
    validate_complex_pair(z, "complex_scale");
    return make_complex(scale(z.real, s), scale(z.imag, s));
}

inline VarPtr abs2(const ComplexVar& z) {
    validate_complex_pair(z, "abs2");
    return add(mul(z.real, z.real), mul(z.imag, z.imag));
}

} // namespace ag
