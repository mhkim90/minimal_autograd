#include "autograd/core/fft.h"
#include "grad_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace ag;
using namespace ag_test;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (!std::isfinite(_a) || !std::isfinite(_b) || std::fabs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "FAIL: %s (%g) vs %s (%g) at %s:%d\n", \
                     #a, _a, #b, _b, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

void test_real_to_complex_and_accessors() {
    auto x = Variable(tensor(Shape{1, 3}, {1.f, -2.f, 3.f}), true);
    auto z = real_to_complex(x);

    auto alias = real(z);
    put(alias.value(), 0, 4.f);
    CHECK_NEAR(at(x.value()), 4.f, 1e-6f);
    sum(alias).backward();
    CHECK(x.has_grad());
    CHECK_NEAR(at(x.grad()), 1.f, 1e-6f);
    x.zero_grad();
    CHECK(!alias.has_grad());
    CHECK(z.imag.value().shape()[0] == 1);
    CHECK(z.imag.value().shape()[1] == 3);
    CHECK_NEAR(max_abs(z.imag.value()), 0.f, 1e-6f);

    std::printf("[ok] real_to_complex creates zero imaginary part\n");
}

void test_complex_forward_ops() {
    auto ar = Variable(filled(Shape{1, 1}, 1.f), true);
    auto ai = Variable(filled(Shape{1, 1}, 2.f), true);
    auto br = Variable(filled(Shape{1, 1}, 3.f), true);
    auto bi = Variable(filled(Shape{1, 1}, -4.f), true);

    auto a = make_complex(ar, ai);
    auto b = make_complex(br, bi);
    auto p = complex_mul(a, b);
    CHECK_NEAR(at(p.real.value(), 0, 0), 11.f, 1e-5f);
    CHECK_NEAR(at(p.imag.value(), 0, 0), 2.f, 1e-5f);

    auto c = conj(a);
    CHECK_NEAR(at(c.real.value(), 0, 0), 1.f, 1e-5f);
    CHECK_NEAR(at(c.imag.value(), 0, 0), -2.f, 1e-5f);

    auto n = abs2(a);
    CHECK_NEAR(at(n.value(), 0, 0), 5.f, 1e-5f);

    std::printf("[ok] complex forward ops\n");
}

void test_complex_grad_check() {
    auto xr = Variable(tensor(Shape{2, 2}, {0.2f, -0.4f, 0.6f, -0.8f}), true);
    auto xi = Variable(tensor(Shape{2, 2}, {0.1f, 0.3f, -0.5f, 0.7f}), true);
    auto wr = Variable(tensor(Shape{2, 2}, {1.0f, -0.5f, 0.25f, 0.75f}), true);
    auto wi = Variable(tensor(Shape{2, 2}, {-0.2f, 0.4f, 0.6f, -0.8f}), true);

    auto f = [wr, wi](std::vector<Variable> xs) {
        auto z = make_complex(xs[0], xs[1]);
        auto w = make_complex(wr, wi);
        return sum(abs2(complex_mul(z, w)));
    };

    CHECK(grad_check_multi(f, {xr, xi}, 1e-3f, 8e-2f));
    std::printf("[ok] complex_mul + abs2 grad_check\n");
}

void test_repeated_branch_gradient() {
    auto xr = Variable(filled(Shape{1, 1}, 2.f), true);
    auto xi = Variable(filled(Shape{1, 1}, -1.f), true);
    auto z = make_complex(xr, xi);

    auto y = sum(abs2(complex_mul(z, z)));
    y.backward();

    CHECK_NEAR(at(xr.grad(), 0, 0), 40.f, 1e-3f);
    CHECK_NEAR(at(xi.grad(), 0, 0), -20.f, 1e-3f);
    std::printf("[ok] repeated complex branch gradient accumulates once\n");
}

void test_shape_validation() {
    bool threw = false;
    try {
        make_complex(Variable(Tensor::zeros(Shape{1, 2}), true),
                     Variable(Tensor::zeros(Shape{2, 1}), true));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("[ok] complex shape validation rejects mismatches\n");
}

void test_fft_round_trip() {
    auto zr = Variable(tensor(Shape{4, 4}, {0.1f, -0.2f, 0.3f, -0.4f,
        0.5f, -0.6f, 0.7f, -0.8f,
        0.9f, -1.0f, 1.1f, -1.2f,
        1.3f, -1.4f, 1.5f, -1.6f}), true);
    auto zi = Variable(tensor(Shape{4, 4}, {-0.3f, 0.2f, -0.1f, 0.0f,
        0.1f, -0.2f, 0.3f, -0.4f,
        0.5f, -0.6f, 0.7f, -0.8f,
        0.9f, -1.0f, 1.1f, -1.2f}), true);

    auto z = make_complex(zr, zi);
    auto y = ifft2(fft2(z));

    CHECK((max_diff(y.real.value(), zr.value())) < 1e-4f);
    CHECK((max_diff(y.imag.value(), zi.value())) < 1e-4f);
    std::printf("[ok] ifft2(fft2(z)) round-trip\n");
}

void test_fft_known_fixtures() {
    auto ones = real_to_complex(Variable(Tensor::ones(Shape{4, 4}), true));
    auto spectrum = fft2(ones);

    CHECK_NEAR(at(spectrum.real.value(), 0, 0), 16.f, 1e-4f);
    CHECK_NEAR(max_abs(spectrum.imag.value()), 0.f, 1e-4f);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (r == 0 && c == 0) continue;
            CHECK_NEAR(at(spectrum.real.value(), r, c), 0.f, 1e-4f);
        }
    }

    Tensor delta = Tensor::zeros(Shape{4, 4});
    put(delta, 0, 1.f);
    auto delta_spectrum = fft2(real_to_complex(Variable(delta, true)));
    CHECK_NEAR(max_diff(delta_spectrum.real.value(), Tensor::ones(Shape{4, 4})), 0.f, 1e-4f);
    CHECK_NEAR(max_abs(delta_spectrum.imag.value()), 0.f, 1e-4f);

    std::printf("[ok] fft2 known fixtures\n");
}

void test_fft_non_square_round_trip() {
    auto zr = Variable(tensor(Shape{4, 8}, {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f,
        2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f, 3.1f}), true);
    auto y = ifft2(fft2(real_to_complex(zr)));
    CHECK((max_diff(y.real.value(), zr.value())) < 2e-4f);
    CHECK_NEAR(max_abs(y.imag.value()), 0.f, 2e-4f);
    std::printf("[ok] fft2 non-square round-trip\n");
}

void test_fft_spectral_filter_grad_check() {
    auto input = Variable(tensor(Shape{2, 2}, {0.2f, -0.4f,
        0.6f, -0.8f}), true);
    auto fr = Variable(tensor(Shape{2, 2}, {1.0f, 0.5f,
        0.25f, -0.75f}), true);
    auto fi = Variable(tensor(Shape{2, 2}, {0.0f, -0.2f,
        0.3f, 0.1f}), true);

    auto f = [fr, fi](Variable x) {
        auto filter = make_complex(fr, fi);
        auto field = ifft2(complex_mul(filter, fft2(real_to_complex(x))));
        return sum(abs2(field));
    };

    CHECK(grad_check(f, input, 1e-3f, 1e-1f));
    std::printf("[ok] fft2 spectral filter grad_check\n");
}

void test_fft_component_grad_checks() {
    auto zr = Variable(tensor(Shape{2, 2}, {0.2f, -0.4f,
        0.6f, -0.8f}), true);
    auto zi = Variable(tensor(Shape{2, 2}, {-0.1f, 0.3f,
        -0.5f, 0.7f}), true);

    auto real_f = [](std::vector<Variable> xs) {
        auto y = fft2(make_complex(xs[0], xs[1]));
        return sum(real(y));
    };
    CHECK(grad_check_multi(real_f, {zr, zi}, 1e-3f, 1e-1f));

    zr.zero_grad();
    zi.zero_grad();
    auto imag_f = [](std::vector<Variable> xs) {
        auto y = fft2(make_complex(xs[0], xs[1]));
        return sum(imag(y));
    };
    CHECK(grad_check_multi(imag_f, {zr, zi}, 1e-3f, 1e-1f));

    std::printf("[ok] fft2 real-only and imag-only grad_checks\n");
}

int main() {
    test_real_to_complex_and_accessors();
    test_complex_forward_ops();
    test_complex_grad_check();
    test_repeated_branch_gradient();
    test_shape_validation();
    test_fft_round_trip();
    test_fft_known_fixtures();
    test_fft_non_square_round_trip();
    test_fft_spectral_filter_grad_check();
    test_fft_component_grad_checks();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
