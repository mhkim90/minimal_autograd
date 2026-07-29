// Replacement consumer tests for the Tensor/Variable ComplexVariable
// and fft2 / ifft2 API.
//
// The translation unit compiles against the new core/* headers and the
// public umbrella autograd.h. It does not depend on the legacy
// Var/Mat ComplexVar / fft2 / ifft2 surface and never includes
// "autograd/complex.h" or "autograd/fft.h" (legacy). It does not
// include Eigen or CUDA, by the hygiene guards below.
//
// Coverage:
//   * ag::ComplexVariable construction, validation, accessors,
//     conj, complex_mul, complex_scale, abs2 on rank-2 and rank-3 /
//     rank-4 inputs.
//   * ag::real_to_complex builds an all-zero imaginary component and
//     propagates shape.
//   * FFT forward on rank-2 fixtures (all-ones, delta, generic
//     input) and an ifft2(fft2(z)) round-trip on square and
//     non-square grids.
//   * FFT forward on rank-3 and rank-4 batched inputs with explicit
//     per-batch independent reference comparisons.
//   * Full 2x2 real / imag Jacobian blocks of fft2 against central
//     finite differences for both loss = sum(real(fft2(z))) and
//     loss = sum(imag(fft2(z))) -- off-diagonal blocks catch sign
//     errors in the real↔imag cross terms of the DFT.
//   * Rank-3 batched (B=2, H=2, W=3) finite-difference gradient on
//     an objective that mixes real and imag components, exercising
//     both cross-term signs and per-batch gradient isolation.
//   * Spectral-filter end-to-end gradient finite-difference check
//     (ifft2(complex_mul(filter, fft2(x)))).
//   * Repeated / shared-branch gradient accumulation: each branch is
//     measured in an isolated fresh graph and the combined graph's
//     gradient is verified against the elementwise sum of the two
//     isolated branch gradients (independent oracle).
//   * Invalid arguments: rank < 2, last-two-axis dim = 0,
//     mismatched real/imag shape, non-Backward FftNorm.
//   * Normalization check: ifft2 includes 1/(H*W) scaling, fft2 does
//     not, and the two compose to the identity on a rank-2 input.
//   * Row-major batch isolation: changes in one batch element do not
//     leak into FFT outputs of other batches.
//
// Non-CPU device rejection is enforced in code (src/core/fft.cpp) but
// cannot be exercised through this test until the CUDA Tensor
// foundation makes a non-CPU replacement Tensor constructible. The
// device branch is covered by code inspection, not by an executable
// test.

#include "autograd/core/complex.h"
#include "autograd/core/fft.h"
#include "autograd/core/fft_norm.h"
#include "autograd/core/ops.h"
#include "autograd/core/variable.h"
#include "autograd/tensor.h"
#include "autograd/shape.h"
#include "autograd/device.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Public core/* headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Public core/* headers must not include CUDA runtime"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int passed = 0;

#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #__VA_ARGS__, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_THROWS_AS(expr, ex_type) do { \
    bool threw = false; \
    try { (void)(expr); } catch (const ex_type&) { threw = true; } \
    catch (...) { /* wrong type */ } \
    CHECK(threw); \
} while (0)

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

ag::Tensor make_tensor(const std::vector<float>& v, const ag::Shape& shape) {
    CHECK(static_cast<std::size_t>(shape.numel()) == v.size());
    return ag::Tensor::from_host(v.empty() ? nullptr : v.data(), shape);
}

std::vector<float> read_values(const ag::Tensor& t) {
    std::vector<float> out(t.elements());
    t.copy_to_host(out.empty() ? nullptr : out.data(), out.size());
    return out;
}

void check_close(const std::vector<float>& a, const std::vector<float>& b,
                 float tol) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) {
            std::fprintf(stderr,
                         "value mismatch at %zu: actual=%g expected=%g\n",
                         i, a[i], b[i]);
            CHECK(false);
        }
    }
}

// Central finite-difference gradient for any scalar-output op. The
// parameter `data` is a flat host buffer; `param_alias` is the live
// Tensor storage view; `f` rebuilds the graph each call.
std::vector<float> finite_difference(
        std::vector<float> data, ag::Tensor param_alias,
        const std::function<ag::Variable()>& f,
        float eps = 1e-3f) {
    const std::size_t n = data.size();
    std::vector<float> g(n, 0.f);
    for (std::size_t i = 0; i < n; ++i) {
        const float orig = data[i];
        data[i] = orig + eps;
        param_alias.copy_from_host(data.data(), data.size());
        const float v_plus = read_values(ag::sum(f()).value())[0];
        data[i] = orig - eps;
        param_alias.copy_from_host(data.data(), data.size());
        const float v_minus = read_values(ag::sum(f()).value())[0];
        data[i] = orig;
        param_alias.copy_from_host(data.data(), data.size());
        g[i] = (v_plus - v_minus) / (2.f * eps);
    }
    return g;
}

// ── ComplexVariable: construction + accessors ─────────────────────────

void test_make_complex_validates_shape() {
    ag::Variable r(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}), true);
    ag::Variable i(make_tensor({5.f, 6.f, 7.f, 8.f}, ag::Shape{2, 2}), true);
    ag::ComplexVariable z = ag::make_complex(r, i);
    CHECK((z.real.value().shape() == ag::Shape{2, 2}));
    CHECK((z.imag.value().shape() == ag::Shape{2, 2}));
    CHECK(z.real.device().is_cpu());

    // Shape mismatch throws.
    ag::Variable bad(make_tensor({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                  ag::Shape{2, 3}), true);
    CHECK_THROWS_AS(ag::make_complex(r, bad), std::invalid_argument);
    report("ag::ComplexVariable validates equal shape");
}

void test_real_to_complex_zero_imag() {
    ag::Variable x(make_tensor({1.f, -2.f, 3.f, -4.f}, ag::Shape{2, 2}), true);
    ag::ComplexVariable z = ag::real_to_complex(x);
    CHECK((z.real.value().shape() == ag::Shape{2, 2}));
    CHECK((z.imag.value().shape() == ag::Shape{2, 2}));
    std::vector<float> imag = read_values(z.imag.value());
    for (float v : imag) CHECK(std::fabs(v) <= 1e-6f);
    // Real component is the same data.
    std::vector<float> real = read_values(z.real.value());
    std::vector<float> x_data = read_values(x.value());
    CHECK(real == x_data);
    // Accessors.
    CHECK(ag::real(z).value().shape() == x.value().shape());
    CHECK(ag::imag(z).value().shape() == x.value().shape());
    report("ag::real_to_complex builds zero imaginary and preserves shape");
}

void test_complex_ops_rank2() {
    ag::Variable ar(make_tensor({1.f}, ag::Shape{1, 1}), true);
    ag::Variable ai(make_tensor({2.f}, ag::Shape{1, 1}), true);
    ag::Variable br(make_tensor({3.f}, ag::Shape{1, 1}), true);
    ag::Variable bi(make_tensor({-4.f}, ag::Shape{1, 1}), true);

    ag::ComplexVariable a = ag::make_complex(ar, ai);
    ag::ComplexVariable b = ag::make_complex(br, bi);

    // (1 + 2i) * (3 - 4i) = (3 + 8) + (6 - 4)i = 11 + 2i.
    ag::ComplexVariable p = ag::complex_mul(a, b);
    CHECK(std::fabs(read_values(p.real.value())[0] - 11.f) <= 1e-5f);
    CHECK(std::fabs(read_values(p.imag.value())[0] - 2.f) <= 1e-5f);

    ag::ComplexVariable c = ag::conj(a);
    CHECK(std::fabs(read_values(c.real.value())[0] - 1.f) <= 1e-5f);
    CHECK(std::fabs(read_values(c.imag.value())[0] - -2.f) <= 1e-5f);

    ag::ComplexVariable s = ag::complex_scale(a, 3.f);
    CHECK(std::fabs(read_values(s.real.value())[0] - 3.f) <= 1e-5f);
    CHECK(std::fabs(read_values(s.imag.value())[0] - 6.f) <= 1e-5f);

    ag::Variable n = ag::abs2(a);
    CHECK(std::fabs(read_values(n.value())[0] - 5.f) <= 1e-5f);

    report("ag::ComplexVariable ops correct on rank-2");
}

void test_complex_ops_rank3_rank4() {
    // Rank-3 batched (B, H, W).
    const int B = 2, H = 3, W = 2;
    std::vector<float> r1(B * H * W), i1(B * H * W), r2(B * H * W),
                       i2(B * H * W);
    std::mt19937 rng(0xa1b2'c3d4u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : r1) v = dist(rng);
    for (auto& v : i1) v = dist(rng);
    for (auto& v : r2) v = dist(rng);
    for (auto& v : i2) v = dist(rng);

    ag::Variable rv1(make_tensor(r1, ag::Shape{B, H, W}), true);
    ag::Variable iv1(make_tensor(i1, ag::Shape{B, H, W}), true);
    ag::Variable rv2(make_tensor(r2, ag::Shape{B, H, W}), true);
    ag::Variable iv2(make_tensor(i2, ag::Shape{B, H, W}), true);
    ag::ComplexVariable a = ag::make_complex(rv1, iv1);
    ag::ComplexVariable b = ag::make_complex(rv2, iv2);

    ag::ComplexVariable p = ag::complex_mul(a, b);
    CHECK((p.real.value().shape() == ag::Shape{B, H, W}));
    CHECK((p.imag.value().shape() == ag::Shape{B, H, W}));

    ag::ComplexVariable c = ag::conj(a);
    CHECK((c.real.value().shape() == ag::Shape{B, H, W}));
    std::vector<float> c_real = read_values(c.real.value());
    std::vector<float> c_imag = read_values(c.imag.value());
    for (int idx = 0; idx < B * H * W; ++idx) {
        CHECK(std::fabs(c_real[idx] - r1[idx]) <= 1e-6f);
        CHECK(std::fabs(c_imag[idx] + i1[idx]) <= 1e-6f);
    }

    ag::Variable n = ag::abs2(a);
    CHECK((n.value().shape() == ag::Shape{B, H, W}));
    std::vector<float> nv = read_values(n.value());
    for (int idx = 0; idx < B * H * W; ++idx) {
        const float expected = r1[idx] * r1[idx] + i1[idx] * i1[idx];
        CHECK(std::fabs(nv[idx] - expected) <= 1e-5f);
    }

    // Rank-4 (N, C, H, W) just exercises shape preservation.
    const int N = 1, C = 2, H4 = 2, W4 = 2;
    std::vector<float> buf(N * C * H4 * W4, 0.f);
    ag::Variable r4(make_tensor(buf, ag::Shape{N, C, H4, W4}), true);
    ag::Variable i4(make_tensor(buf, ag::Shape{N, C, H4, W4}), true);
    ag::ComplexVariable z4 = ag::make_complex(r4, i4);
    CHECK((z4.real.value().shape() == ag::Shape{N, C, H4, W4}));
    CHECK((z4.imag.value().shape() == ag::Shape{N, C, H4, W4}));
    ag::ComplexVariable s4 = ag::complex_scale(z4, 2.f);
    CHECK((s4.real.value().shape() == ag::Shape{N, C, H4, W4}));
    report("ag::ComplexVariable ops preserve shape on rank-3 / rank-4");
}

// ── FFT: rank-2 fixtures and round trip ───────────────────────────────

void test_fft_rank2_round_trip_square() {
    const int H = 4, W = 4;
    std::vector<float> zr(H * W), zi(H * W);
    std::mt19937 rng(0x5f37'5f37u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{H, W}), true);
    ag::ComplexVariable z = ag::make_complex(rv, iv);

    ag::ComplexVariable y = ag::ifft2(ag::fft2(z));
    check_close(read_values(y.real.value()), zr, 2e-4f);
    check_close(read_values(y.imag.value()), zi, 2e-4f);
    report("ifft2(fft2(z)) rank-2 round trip on square grid");
}

void test_fft_rank2_round_trip_non_square() {
    const int H = 3, W = 5;
    std::vector<float> zr(H * W);
    std::mt19937 rng(0x6e44'6e44u);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto& v : zr) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{H, W}), true);
    ag::ComplexVariable z = ag::real_to_complex(rv);

    ag::ComplexVariable y = ag::ifft2(ag::fft2(z));
    check_close(read_values(y.real.value()), zr, 2e-4f);
    std::vector<float> yi = read_values(y.imag.value());
    for (float v : yi) CHECK(std::fabs(v) <= 2e-4f);
    report("ifft2(fft2(z)) rank-2 round trip on non-square grid");
}

void test_fft_rank2_known_fixtures() {
    const int H = 4, W = 4;
    // All-ones: spectrum is (H*W, 0, 0, ..., 0).
    {
        ag::Variable ones(make_tensor(std::vector<float>(H * W, 1.f),
                                        ag::Shape{H, W}), true);
        ag::ComplexVariable z = ag::real_to_complex(ones);
        ag::ComplexVariable s = ag::fft2(z);
        std::vector<float> sr = read_values(s.real.value());
        std::vector<float> si = read_values(s.imag.value());
        CHECK(std::fabs(sr[0] - static_cast<float>(H * W)) <= 1e-3f);
        for (int i = 1; i < H * W; ++i) CHECK(std::fabs(sr[i]) <= 1e-3f);
        for (float v : si) CHECK(std::fabs(v) <= 1e-3f);
    }
    // Delta at (0, 0): spectrum is all 1.
    {
        std::vector<float> delta(H * W, 0.f);
        delta[0] = 1.f;
        ag::Variable d(make_tensor(delta, ag::Shape{H, W}), true);
        ag::ComplexVariable z = ag::real_to_complex(d);
        ag::ComplexVariable s = ag::fft2(z);
        std::vector<float> sr = read_values(s.real.value());
        std::vector<float> si = read_values(s.imag.value());
        for (float v : sr) CHECK(std::fabs(v - 1.f) <= 1e-4f);
        for (float v : si) CHECK(std::fabs(v) <= 1e-4f);
    }
    report("fft2 known fixtures (all-ones, delta)");
}

void test_fft_normalization() {
    // fft2 produces unscaled DFT; ifft2 divides by H*W; composing
    // them yields identity. The same check in the round-trip test
    // already exercises composition; this test additionally verifies
    // the per-op scaling directly.
    const int H = 4, W = 4;
    std::vector<float> zr(H * W);
    std::mt19937 rng(0x12c4'8f73u);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (auto& v : zr) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{H, W}), true);
    ag::ComplexVariable z = ag::real_to_complex(rv);
    ag::ComplexVariable s = ag::fft2(z);
    std::vector<float> sr = read_values(s.real.value());
    std::vector<float> si = read_values(s.imag.value());
    // Each s[k] is sum_r sum_c zr[r,c] * cos(angle) (no 1/(HW)).
    // Compute reference FFT for the real-valued input.
    std::vector<float> ref_sr(H * W, 0.f);
    std::vector<float> ref_si(H * W, 0.f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int kr = 0; kr < H; ++kr) {
        for (int kc = 0; kc < W; ++kc) {
            float sum_r = 0.f, sum_i = 0.f;
            for (int r = 0; r < H; ++r) {
                for (int c = 0; c < W; ++c) {
                    const float a = 2.f * kPi *
                        (static_cast<float>(kr * r) / static_cast<float>(H) +
                         static_cast<float>(kc * c) / static_cast<float>(W));
                    sum_r += zr[r * W + c] * std::cos(a);
                    sum_i += zr[r * W + c] * (-std::sin(a));
                }
            }
            ref_sr[kr * W + kc] = sum_r;
            ref_si[kr * W + kc] = sum_i;
        }
    }
    check_close(sr, ref_sr, 1e-4f);
    check_close(si, ref_si, 1e-4f);
    report("fft2 forward normalization is unscaled DFT (Backward norm)");
}

// ── FFT: batched rank > 2 with independent reference ──────────────────

// Reference DFT on a single (H, W) plane.
std::pair<std::vector<float>, std::vector<float>> dft2_reference_plane(
        const std::vector<float>& real_in, const std::vector<float>& imag_in,
        int H, int W, bool inverse, bool scale_output) {
    constexpr float kPi = 3.14159265358979323846f;
    const float norm = scale_output ? 1.f / static_cast<float>(H * W) : 1.f;
    std::vector<float> out_r(H * W, 0.f), out_i(H * W, 0.f);
    for (int kr = 0; kr < H; ++kr) {
        for (int kc = 0; kc < W; ++kc) {
            float sum_r = 0.f, sum_i = 0.f;
            for (int r = 0; r < H; ++r) {
                for (int c = 0; c < W; ++c) {
                    const float a = 2.f * kPi *
                        (static_cast<float>(kr * r) / static_cast<float>(H) +
                         static_cast<float>(kc * c) / static_cast<float>(W));
                    const float wr = std::cos(a);
                    const float wi = inverse ? std::sin(a) : -std::sin(a);
                    sum_r += real_in[r * W + c] * wr -
                             imag_in[r * W + c] * wi;
                    sum_i += real_in[r * W + c] * wi +
                             imag_in[r * W + c] * wr;
                }
            }
            out_r[kr * W + kc] = norm * sum_r;
            out_i[kr * W + kc] = norm * sum_i;
        }
    }
    return {out_r, out_i};
}

void test_fft_rank3_batched_independent_per_batch() {
    const int B = 2, H = 4, W = 3;
    std::vector<float> zr(B * H * W), zi(B * H * W);
    std::mt19937 rng(0x9911'aa22u);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{B, H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{B, H, W}), true);
    ag::ComplexVariable z = ag::make_complex(rv, iv);

    ag::ComplexVariable s = ag::fft2(z);
    CHECK((s.real.value().shape() == ag::Shape{B, H, W}));
    std::vector<float> sr = read_values(s.real.value());
    std::vector<float> si = read_values(s.imag.value());
    for (int b = 0; b < B; ++b) {
        std::vector<float> plane_r(zr.begin() + b * H * W,
                                    zr.begin() + (b + 1) * H * W);
        std::vector<float> plane_i(zi.begin() + b * H * W,
                                    zi.begin() + (b + 1) * H * W);
        auto ref = dft2_reference_plane(plane_r, plane_i, H, W,
                                          /*inverse=*/false,
                                          /*scale_output=*/false);
        std::vector<float> got_r(sr.begin() + b * H * W,
                                  sr.begin() + (b + 1) * H * W);
        std::vector<float> got_i(si.begin() + b * H * W,
                                  si.begin() + (b + 1) * H * W);
        check_close(got_r, ref.first, 1e-4f);
        check_close(got_i, ref.second, 1e-4f);
    }
    report("fft2 rank-3 batched FFT matches per-batch independent reference");
}

void test_fft_rank4_batched_independent_per_batch() {
    const int N = 2, C = 2, H = 3, W = 3;
    std::vector<float> zr(N * C * H * W), zi(N * C * H * W);
    std::mt19937 rng(0x7733'4411u);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{N, C, H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{N, C, H, W}), true);
    ag::ComplexVariable z = ag::make_complex(rv, iv);

    ag::ComplexVariable s = ag::fft2(z);
    CHECK((s.real.value().shape() == ag::Shape{N, C, H, W}));
    std::vector<float> sr = read_values(s.real.value());
    std::vector<float> si = read_values(s.imag.value());

    const int plane = H * W;
    const int batch_count = N * C;
    for (int b = 0; b < batch_count; ++b) {
        std::vector<float> plane_r(zr.begin() + b * plane,
                                    zr.begin() + (b + 1) * plane);
        std::vector<float> plane_i(zi.begin() + b * plane,
                                    zi.begin() + (b + 1) * plane);
        auto ref = dft2_reference_plane(plane_r, plane_i, H, W,
                                          /*inverse=*/false,
                                          /*scale_output=*/false);
        std::vector<float> got_r(sr.begin() + b * plane,
                                  sr.begin() + (b + 1) * plane);
        std::vector<float> got_i(si.begin() + b * plane,
                                  si.begin() + (b + 1) * plane);
        check_close(got_r, ref.first, 1e-4f);
        check_close(got_i, ref.second, 1e-4f);
    }
    // Round trip per batch slice.
    ag::ComplexVariable back = ag::ifft2(s);
    CHECK((back.real.value().shape() == ag::Shape{N, C, H, W}));
    check_close(read_values(back.real.value()), zr, 2e-4f);
    check_close(read_values(back.imag.value()), zi, 2e-4f);
    report("fft2 rank-4 batched FFT matches independent reference and round-trips");
}

void test_fft_row_major_batch_isolation() {
    // Build a batched input where batch 0 is all-1 and batch 1 is all-2.
    // Compute fft2 and verify each batch plane matches the reference
    // for that plane alone. This confirms the batched kernel iterates
    // leading axes independently and does not leak across batches.
    const int B = 2, H = 4, W = 4;
    std::vector<float> zr(B * H * W, 0.f);
    std::vector<float> zi(B * H * W, 0.f);
    for (int i = 0; i < H * W; ++i) zr[0 * H * W + i] = 1.f;
    for (int i = 0; i < H * W; ++i) zr[1 * H * W + i] = 2.f;
    ag::Variable rv(make_tensor(zr, ag::Shape{B, H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{B, H, W}), true);
    ag::ComplexVariable z = ag::make_complex(rv, iv);

    ag::ComplexVariable s = ag::fft2(z);
    std::vector<float> sr = read_values(s.real.value());
    std::vector<float> si = read_values(s.imag.value());

    auto ref_ones = dft2_reference_plane(
        std::vector<float>(H * W, 1.f), std::vector<float>(H * W, 0.f),
        H, W, /*inverse=*/false, /*scale_output=*/false);
    auto ref_twos = dft2_reference_plane(
        std::vector<float>(H * W, 2.f), std::vector<float>(H * W, 0.f),
        H, W, /*inverse=*/false, /*scale_output=*/false);

    std::vector<float> got_ones_r(sr.begin(), sr.begin() + H * W);
    std::vector<float> got_ones_i(si.begin(), si.begin() + H * W);
    std::vector<float> got_twos_r(sr.begin() + H * W, sr.end());
    std::vector<float> got_twos_i(si.begin() + H * W, si.end());
    check_close(got_ones_r, ref_ones.first, 1e-4f);
    check_close(got_ones_i, ref_ones.second, 1e-4f);
    check_close(got_twos_r, ref_twos.first, 1e-4f);
    check_close(got_twos_i, ref_twos.second, 1e-4f);
    report("fft2 row-major batched: per-batch planes are isolated");
}

void test_fft_rank3_batched_gradient_finite_diff() {
    // Small batched rank-3 (B=2, H=2, W=3) FD gradient test on an
    // objective that mixes real and imag components. Exercises both
    // cross-term signs and the batched-batch isolation of the
    // backward adjoint over leading axes.
    const int B = 2, H = 2, W = 3;
    std::vector<float> zr(B * H * W), zi(B * H * W);
    std::mt19937 rng(0xb2c3'd4e5u);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{B, H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{B, H, W}), true);

    auto f = [&] {
        ag::ComplexVariable y = ag::fft2(ag::make_complex(rv, iv));
        // |y|^2 + real(y) * imag(y) exercises both the square of
        // each component and the cross product.
        ag::Variable mag = ag::abs2(y);
        ag::Variable cross = ag::mul(ag::real(y), ag::imag(y));
        return ag::sum(ag::add(mag, cross));
    };
    auto fd_r = finite_difference(zr, rv.value(), f);
    auto fd_i = finite_difference(zi, iv.value(), f);
    rv.zero_grad();
    iv.zero_grad();
    ag::sum(f()).backward();
    check_close(read_values(rv.grad()), fd_r, 5e-2f);
    check_close(read_values(iv.grad()), fd_i, 5e-2f);
    report("rank-3 batched FFT FD gradient matches FD (real and imag)");
}

// ── FFT: gradients ────────────────────────────────────────────────────

void test_fft_real_only_output_gradient_finite_diff() {
    const int H = 4, W = 4;
    std::vector<float> zr(H * W), zi(H * W);
    std::mt19937 rng(0xfeed'beefu);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{H, W}), true);

    // Loss = sum(real(fft2(z))): full 2x2 Jacobian block wrt (rv, iv).
    // The off-diagonal terms (d_loss/d_iv) catch sign errors in the
    // real↔imag cross terms of the DFT.
    {
        auto f = [&] {
            ag::ComplexVariable y = ag::fft2(ag::make_complex(rv, iv));
            return ag::sum(ag::real(y));
        };
        auto fd_r = finite_difference(zr, rv.value(), f);
        auto fd_i = finite_difference(zi, iv.value(), f);
        rv.zero_grad();
        iv.zero_grad();
        ag::sum(f()).backward();
        check_close(read_values(rv.grad()), fd_r, 5e-2f);
        check_close(read_values(iv.grad()), fd_i, 5e-2f);
    }

    // Loss = sum(imag(fft2(z))): full 2x2 Jacobian block. Off-diagonal
    // (d_loss/d_rv) catches the inverse sign pairing.
    {
        auto f = [&] {
            ag::ComplexVariable y = ag::fft2(ag::make_complex(rv, iv));
            return ag::sum(ag::imag(y));
        };
        auto fd_r = finite_difference(zr, rv.value(), f);
        auto fd_i = finite_difference(zi, iv.value(), f);
        rv.zero_grad();
        iv.zero_grad();
        ag::sum(f()).backward();
        check_close(read_values(rv.grad()), fd_r, 5e-2f);
        check_close(read_values(iv.grad()), fd_i, 5e-2f);
    }
    report("fft2 full 2x2 real/imag Jacobian blocks match FD for sum(real) and sum(imag)");
}

void test_fft_spectral_filter_gradient_finite_diff() {
    // ifft2(complex_mul(filter, fft2(real_to_complex(x)))).sum() wrt x.
    const int H = 4, W = 4;
    std::vector<float> zx(H * W);
    std::vector<float> fr(H * W), fi(H * W);
    std::mt19937 rng(0xfeed'c0deu);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (auto& v : zx) v = dist(rng);
    for (auto& v : fr) v = dist(rng);
    for (auto& v : fi) v = dist(rng);
    ag::Variable xv(make_tensor(zx, ag::Shape{H, W}), true);
    ag::Variable fvr(make_tensor(fr, ag::Shape{H, W}), true);
    ag::Variable fvi(make_tensor(fi, ag::Shape{H, W}), true);

    auto f = [&] {
        ag::ComplexVariable filter = ag::make_complex(fvr, fvi);
        ag::ComplexVariable field = ag::ifft2(ag::complex_mul(
            filter, ag::fft2(ag::real_to_complex(xv))));
        return ag::sum(ag::abs2(field));
    };
    auto fd = finite_difference(zx, xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("spectral filter ifft2(complex_mul(filter, fft2(x))) gradient matches FD");
}

void test_fft_repeated_branch_gradient_accumulates() {
    const int H = 2, W = 2;
    std::vector<float> zr(H * W), zi(H * W);
    std::mt19937 rng(0x1357'2468u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : zr) v = dist(rng);
    for (auto& v : zi) v = dist(rng);
    ag::Variable rv(make_tensor(zr, ag::Shape{H, W}), true);
    ag::Variable iv(make_tensor(zi, ag::Shape{H, W}), true);

    // Measure branch 1 in an isolated fresh graph:
    //   loss_a = sum(abs2(fft2(z))).
    ag::sum(ag::abs2(ag::fft2(ag::make_complex(rv, iv)))).backward();
    std::vector<float> g_a_r = read_values(rv.grad());
    std::vector<float> g_a_i = read_values(iv.grad());

    rv.zero_grad();
    iv.zero_grad();

    // Measure branch 2 in an isolated fresh graph:
    //   loss_b = sum(abs2(ifft2(z))).
    ag::sum(ag::abs2(ag::ifft2(ag::make_complex(rv, iv)))).backward();
    std::vector<float> g_b_r = read_values(rv.grad());
    std::vector<float> g_b_i = read_values(iv.grad());

    rv.zero_grad();
    iv.zero_grad();

    // Combined graph: both branches consume the same z parents. The
    // resulting gradient must equal the elementwise sum of the two
    // isolated branch gradients; this is the autograd accumulation
    // contract and is independent of the analytic FFT Jacobian.
    ag::ComplexVariable z_c = ag::make_complex(rv, iv);
    ag::Variable branch1 = ag::sum(ag::abs2(ag::fft2(z_c)));
    ag::Variable branch2 = ag::sum(ag::abs2(ag::ifft2(z_c)));
    ag::sum(ag::add(branch1, branch2)).backward();

    std::vector<float> expected_r(g_a_r.size()), expected_i(g_a_i.size());
    for (std::size_t i = 0; i < g_a_r.size(); ++i) {
        expected_r[i] = g_a_r[i] + g_b_r[i];
    }
    for (std::size_t i = 0; i < g_a_i.size(); ++i) {
        expected_i[i] = g_a_i[i] + g_b_i[i];
    }
    check_close(read_values(rv.grad()), expected_r, 1e-4f);
    check_close(read_values(iv.grad()), expected_i, 1e-4f);
    report("fft2 / ifft2 combined-branch gradient equals sum of isolated branch gradients");
}

// ── FFT: invalid arguments ───────────────────────────────────────────

void test_fft_invalid_arguments() {
    // Rank-1 input rejected.
    {
        ag::Variable x(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{4}), true);
        ag::ComplexVariable z = ag::real_to_complex(x);
        CHECK_THROWS_AS(ag::fft2(z), std::invalid_argument);
        CHECK_THROWS_AS(ag::ifft2(z), std::invalid_argument);
    }
    // Zero-extent final axis rejected.
    {
        ag::Variable x(make_tensor(std::vector<float>{}, ag::Shape{2, 0}),
                        true);
        ag::ComplexVariable z = ag::real_to_complex(x);
        CHECK_THROWS_AS(ag::fft2(z), std::invalid_argument);
    }
    // Real/imag shape mismatch.
    {
        ag::Variable r(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                        true);
        ag::Variable i(make_tensor({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                    ag::Shape{2, 3}), true);
        // make_complex itself rejects the mismatched pair.
        CHECK_THROWS_AS(ag::make_complex(r, i), std::invalid_argument);
    }
    // Non-Backward normalization rejected.
    {
        ag::Variable x(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                        true);
        ag::ComplexVariable z = ag::real_to_complex(x);
        // Cast through int to bypass static_assert on enum class.
        const ag::FftNorm bogus = static_cast<ag::FftNorm>(42);
        CHECK_THROWS_AS(ag::fft2(z, bogus), std::runtime_error);
        CHECK_THROWS_AS(ag::ifft2(z, bogus), std::runtime_error);
    }
    report("fft2 / ifft2 reject invalid arguments");
}

}  // namespace

int main() {
    test_make_complex_validates_shape();
    test_real_to_complex_zero_imag();
    test_complex_ops_rank2();
    test_complex_ops_rank3_rank4();
    test_fft_rank2_round_trip_square();
    test_fft_rank2_round_trip_non_square();
    test_fft_rank2_known_fixtures();
    test_fft_normalization();
    test_fft_rank3_batched_independent_per_batch();
    test_fft_rank4_batched_independent_per_batch();
    test_fft_row_major_batch_isolation();
    test_fft_real_only_output_gradient_finite_diff();
    test_fft_rank3_batched_gradient_finite_diff();
    test_fft_spectral_filter_gradient_finite_diff();
    test_fft_repeated_branch_gradient_accumulates();
    test_fft_invalid_arguments();

    std::printf("\nALL OOP FFT TESTS PASSED (%d)\n", passed);
    return 0;
}