// Replacement consumer tests for the Tensor/Variable diffusion
// helpers on the replacement public API.
//
// This translation unit compiles against:
//   * the new core/* public headers (Eigen-free), guarded against
//     leakage of Eigen / CUDA into the public replacement API;
//   * the legacy umbrella autograd.h, after the hygiene guards, to
//     prove that the legacy flat ag::* functions and replacement
//     ag::diffusion::* functions coexist in the same TU without
//     signature collisions.
//
// The replacement helpers live in the transitional namespace
// ag::diffusion to avoid signature collisions with the legacy
// ag::diffusion.h free functions (which expose
// ag::sinusoidal_time_embedding(int, int)).
//
// Coverage:
//   * ag::diffusion::randn: shape (rank-agnostic), mean ~ 0 /
//     std ~ 1 over large samples, deterministic for a fixed seed,
//     different seeds produce independent sequences.
//   * ag::diffusion::sinusoidal_time_embedding: shape (dim,), value
//     range in [-1, 1], sin/cos pairing at index 0 / dim/2 (even
//     dim) and safe behaviour for odd dim.
//   * ag::diffusion::q_sample: forward shape, zero-noise -> scaled
//     x0, analytic vs central-finite-difference gradients through
//     x0 and noise on rank-0 / rank-1 / rank-2 / rank-4 inputs
//     and on a zero-element shape.
//   * Invalid argument paths.
//   * Public-header hygiene: no Eigen / CUDA leakage from core/*.
//   * Legacy + replacement namespace coexistence.

#include "autograd/core/diffusion.h"
#include "autograd/core/ops.h"
#include "autograd/core/variable.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Public diffusion/ops headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Public diffusion/ops headers must not include CUDA runtime"
#endif

// The legacy umbrella is included AFTER the hygiene guards so the
// guards verify that the replacement core/* headers do not leak
// Eigen / CUDA into a clean public-API consumer TU. autograd.h
// legitimately brings in Eigen; that is fine because it lives
// behind a name alias for the legacy surface. Once included, the
// coexistence test below exercises both namespaces in the same TU.
#include "autograd.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
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

float mean_of(const std::vector<float>& v) {
    if (v.empty()) return 0.f;
    double s = 0.0;
    for (float x : v) s += x;
    return static_cast<float>(s / static_cast<double>(v.size()));
}

float stddev_of(const std::vector<float>& v, float m) {
    if (v.empty()) return 0.f;
    double s = 0.0;
    for (float x : v) {
        const double d = x - m;
        s += d * d;
    }
    return static_cast<float>(
        std::sqrt(s / static_cast<double>(v.size())));
}

void check_close(const std::vector<float>& a, const std::vector<float>& b,
                 float tol = 5e-2f) {
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

// Central finite-difference gradient. Mutates the parameter `data`
// in place via its alias `param_alias`, evaluates f(data) for the
// +/- perturbations, and returns the gradient.
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

// ── randn ────────────────────────────────────────────────────────────

void test_randn_shape_rank4() {
    // rank-agnostic Shape: (2, 3, 4, 5).
    ag::Tensor t = ag::diffusion::randn(ag::Shape{2, 3, 4, 5}, 7);
    CHECK((t.shape() == ag::Shape{2, 3, 4, 5}));
    CHECK(static_cast<std::size_t>(t.elements()) == 2 * 3 * 4 * 5);
    CHECK(t.device().is_cpu());
    report("ag::diffusion::randn rank-4 shape and CPU device");
}

void test_randn_stats() {
    // Mean ~ 0, std ~ 1 over a large sample, deterministic for a seed.
    ag::Tensor t = ag::diffusion::randn(ag::Shape{1, 20000}, 42);
    std::vector<float> v = read_values(t);
    const float m = mean_of(v);
    const float s = stddev_of(v, m);
    CHECK(std::fabs(m) < 0.05f);
    CHECK(std::fabs(s - 1.f) < 0.05f);
    report("ag::diffusion::randn mean ~ 0 and std ~ 1 (large sample)");
}

void test_randn_reproducible_same_seed() {
    ag::Tensor a = ag::diffusion::randn(ag::Shape{2, 8}, 123);
    ag::Tensor b = ag::diffusion::randn(ag::Shape{2, 8}, 123);
    std::vector<float> va = read_values(a);
    std::vector<float> vb = read_values(b);
    for (std::size_t i = 0; i < va.size(); ++i) {
        CHECK(std::fabs(va[i] - vb[i]) <= 1e-6f);
    }
    report("ag::diffusion::randn deterministic for fixed seed");
}

void test_randn_different_seeds_independent() {
    ag::Tensor a = ag::diffusion::randn(ag::Shape{2, 64}, 1);
    ag::Tensor b = ag::diffusion::randn(ag::Shape{2, 64}, 2);
    std::vector<float> va = read_values(a);
    std::vector<float> vb = read_values(b);
    // Probability of all 128 pairs being within 0.01 is essentially
    // zero for two independent standard-normal sequences.
    int near = 0;
    for (std::size_t i = 0; i < va.size(); ++i) {
        if (std::fabs(va[i] - vb[i]) <= 0.01f) ++near;
    }
    CHECK(near < 8);
    report("ag::diffusion::randn different seeds produce independent sequences");
}

void test_randn_like_shape() {
    ag::Tensor x = make_tensor(std::vector<float>(1 * 2 * 3 * 3, 0.f),
                               ag::Shape{1, 2, 3, 3});
    ag::Tensor n = ag::diffusion::randn_like(x, 1);
    CHECK((n.shape() == ag::Shape{1, 2, 3, 3}));
    CHECK(static_cast<std::size_t>(n.elements()) ==
          static_cast<std::size_t>(x.elements()));
    report("ag::diffusion::randn_like matches source shape");
}

void test_randn_invalid_arguments() {
    // Empty shape: zero elements is valid and produces an empty Tensor.
    {
        ag::Tensor t = ag::diffusion::randn(ag::Shape{0}, 0);
        CHECK(t.elements() == 0);
    }
    // Negative dim is rejected at Shape construction.
    bool threw = false;
    try {
        ag::Shape bad = ag::Shape({-1, 2});
        (void)bad;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    report("ag::diffusion::randn / Shape rejects invalid arguments");
}

// ── sinusoidal_time_embedding ───────────────────────────────────────

void test_time_emb_shape_even() {
    ag::Tensor pe = ag::diffusion::sinusoidal_time_embedding(10, 16);
    CHECK((pe.shape() == ag::Shape{16}));
    CHECK(pe.elements() == 16);
    CHECK(pe.device().is_cpu());
    report("ag::diffusion::sinusoidal_time_embedding rank-1 shape (dim,)");
}

void test_time_emb_shape_odd() {
    ag::Tensor pe = ag::diffusion::sinusoidal_time_embedding(5, 7);
    CHECK((pe.shape() == ag::Shape{7}));
    CHECK(pe.elements() == 7);
    std::vector<float> v = read_values(pe);
    // Values stay in [-1, 1] even for odd dim.
    for (float x : v) {
        CHECK(x >= -1.f - 1e-6f);
        CHECK(x <= 1.f + 1e-6f);
    }
    report("ag::diffusion::sinusoidal_time_embedding handles odd dim safely");
}

void test_time_emb_period() {
    // freq_0 = exp(0) = 1; first element must be sin(t).
    ag::Tensor pe = ag::diffusion::sinusoidal_time_embedding(7, 8);
    std::vector<float> v = read_values(pe);
    const int half = 4;
    CHECK(std::fabs(v[0] - std::sin(7.f)) <= 1e-5f);
    CHECK(std::fabs(v[half] - std::cos(7.f)) <= 1e-5f);
    report("ag::diffusion::sinusoidal_time_embedding pe[0]=sin(t), pe[half]=cos(t)");
}

void test_time_emb_range_even() {
    ag::Tensor pe = ag::diffusion::sinusoidal_time_embedding(123, 32);
    std::vector<float> v = read_values(pe);
    for (float x : v) {
        CHECK(x >= -1.f - 1e-6f);
        CHECK(x <= 1.f + 1e-6f);
    }
    report("ag::diffusion::sinusoidal_time_embedding values in [-1, 1]");
}

void test_time_emb_invalid_arguments() {
    CHECK_THROWS_AS(ag::diffusion::sinusoidal_time_embedding(0, 0),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::diffusion::sinusoidal_time_embedding(0, -4),
                    std::invalid_argument);
    report("ag::diffusion::sinusoidal_time_embedding rejects non-positive dim");
}

// ── q_sample ────────────────────────────────────────────────────────

void test_q_sample_shape_rank4() {
    const int N = 2, C = 3, H = 4, W = 4;
    std::vector<float> x0(N * C * H * W);
    std::vector<float> noise(N * C * H * W);
    for (auto& v : x0) v = 0.1f * static_cast<float>(&v - x0.data()) - 0.5f;
    for (auto& v : noise) v = 0.05f * static_cast<float>(&v - noise.data());

    ag::Variable xv(make_tensor(x0, ag::Shape{N, C, H, W}), true);
    ag::Variable nv(make_tensor(noise, ag::Shape{N, C, H, W}), true);
    ag::Variable xt = ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv);
    CHECK((xt.value().shape() == ag::Shape{N, C, H, W}));
    report("ag::diffusion::q_sample rank-4 output shape matches x0");
}

void test_q_sample_zero_noise_equals_scale() {
    const int N = 1, C = 2, H = 2, W = 2;
    std::vector<float> x0(N * C * H * W, 2.f);
    std::vector<float> noise(N * C * H * W, 0.f);
    ag::Variable xv(make_tensor(x0, ag::Shape{N, C, H, W}), true);
    ag::Variable nv(make_tensor(noise, ag::Shape{N, C, H, W}), true);
    ag::Variable xt = ag::diffusion::q_sample(xv, /*sqrt_ab=*/0.5f,
                                    /*sqrt_1mab=*/1.f, &nv);
    std::vector<float> v = read_values(xt.value());
    for (float x : v) {
        CHECK(std::fabs(x - 1.f) <= 1e-5f);
    }
    report("ag::diffusion::q_sample with zero noise = sqrt_ab * x0");
}

void test_q_sample_grad_x0() {
    const int N = 1, C = 2, H = 3, W = 3;
    std::mt19937 rng(0x9e37'79b9u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x0(N * C * H * W);
    std::vector<float> noise(N * C * H * W);
    for (auto& v : x0) v = dist(rng);
    for (auto& v : noise) v = dist(rng);

    ag::Variable xv(make_tensor(x0, ag::Shape{N, C, H, W}), true);
    ag::Variable nv(make_tensor(noise, ag::Shape{N, C, H, W}), true);
    auto f = [&] { return ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv); };
    auto fd = finite_difference(
        read_values(xv.value()), xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("ag::diffusion::q_sample gradient: d_x0 matches FD (rank-4)");
}

void test_q_sample_grad_noise() {
    const int N = 1, C = 2, H = 3, W = 3;
    std::mt19937 rng(0x12c4'8f73u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x0(N * C * H * W);
    std::vector<float> noise(N * C * H * W);
    for (auto& v : x0) v = dist(rng);
    for (auto& v : noise) v = dist(rng);

    ag::Variable xv(make_tensor(x0, ag::Shape{N, C, H, W}), true);
    ag::Variable nv(make_tensor(noise, ag::Shape{N, C, H, W}), true);
    auto f = [&] { return ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv); };
    auto fd = finite_difference(
        read_values(nv.value()), nv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(nv.grad()), fd, 5e-2f);
    report("ag::diffusion::q_sample gradient: d_noise matches FD (rank-4)");
}

void test_q_sample_grad_rank2() {
    // Same coverage as the legacy rank-2 path.
    const int N = 2, D = 5;
    std::mt19937 rng(0x77d3'4b91u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x0(N * D);
    std::vector<float> noise(N * D);
    for (auto& v : x0) v = dist(rng);
    for (auto& v : noise) v = dist(rng);

    ag::Variable xv(make_tensor(x0, ag::Shape{N, D}), true);
    ag::Variable nv(make_tensor(noise, ag::Shape{N, D}), true);
    auto f = [&] { return ag::diffusion::q_sample(xv, 0.7f, 0.7f, &nv); };
    auto fd = finite_difference(
        read_values(xv.value()), xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("ag::diffusion::q_sample gradient: d_x0 matches FD (rank-2)");
}

void test_q_sample_default_noise_uses_seed() {
    // When noise is null, q_sample draws fresh randn_like(x0) with
    // the given seed. Two equivalent calls must agree element-wise.
    const int N = 1, D = 4;
    std::vector<float> x0(N * D);
    for (std::size_t i = 0; i < x0.size(); ++i) x0[i] = 0.1f * static_cast<float>(i);
    ag::Variable xv_a(make_tensor(x0, ag::Shape{N, D}), true);
    ag::Variable xv_b(make_tensor(x0, ag::Shape{N, D}), true);
    ag::Variable ya = ag::diffusion::q_sample(xv_a, 0.8f, 0.6f, nullptr, 999);
    ag::Variable yb = ag::diffusion::q_sample(xv_b, 0.8f, 0.6f, nullptr, 999);
    std::vector<float> va = read_values(ya.value());
    std::vector<float> vb = read_values(yb.value());
    for (std::size_t i = 0; i < va.size(); ++i) {
        CHECK(std::fabs(va[i] - vb[i]) <= 1e-6f);
    }
    report("ag::diffusion::q_sample default noise is deterministic for fixed seed");
}

void test_q_sample_invalid_arguments() {
    // Shape mismatch between x0 and noise.
    {
        ag::Variable xv(make_tensor(std::vector<float>(8, 0.f),
                                    ag::Shape{2, 4}), true);
        ag::Variable nv(make_tensor(std::vector<float>(12, 0.f),
                                    ag::Shape{3, 4}), true);
        CHECK_THROWS_AS(ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv),
                        std::invalid_argument);
    }
    // Non-finite scalars rejected.
    {
        ag::Variable xv(make_tensor(std::vector<float>(4, 0.f),
                                    ag::Shape{2, 2}), true);
        CHECK_THROWS_AS(ag::diffusion::q_sample(xv, std::nanf(""), 0.6f),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::diffusion::q_sample(xv, 0.8f, std::nanf("")),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::diffusion::q_sample(xv,
                                    std::numeric_limits<float>::infinity(),
                                    0.6f),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::diffusion::q_sample(xv, 0.8f,
                                    -std::numeric_limits<float>::infinity()),
                        std::invalid_argument);
    }
    report("ag::diffusion::q_sample rejects invalid arguments");
}

void test_q_sample_rank1_forward_and_gradient() {
    // Rank-1 (vector-shaped) input. q_sample composes scale/add and
    // must support any rank, including rank-1.
    const std::size_t N = 8;
    std::vector<float> x0(N);
    std::vector<float> noise(N);
    for (std::size_t i = 0; i < N; ++i) {
        x0[i] = 0.1f * static_cast<float>(i) - 0.4f;
        noise[i] = 0.05f * static_cast<float>(i) - 0.1f;
    }
    ag::Variable xv(make_tensor(x0, ag::Shape{static_cast<int64_t>(N)}),
                     true);
    ag::Variable nv(make_tensor(noise, ag::Shape{static_cast<int64_t>(N)}),
                     true);
    ag::Variable xt = ag::diffusion::q_sample(xv, 0.7f, 0.7f, &nv);
    CHECK((xt.value().shape() == ag::Shape{static_cast<int64_t>(N)}));

    // Forward: y_i = 0.7 * x0_i + 0.7 * noise_i.
    for (std::size_t i = 0; i < N; ++i) {
        const float expected = 0.7f * x0[i] + 0.7f * noise[i];
        CHECK(std::fabs(read_values(xt.value())[i] - expected) <= 1e-6f);
    }

    // FD gradient check.
    auto f = [&] { return ag::diffusion::q_sample(xv, 0.7f, 0.7f, &nv); };
    auto fd = finite_difference(
        read_values(xv.value()), xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("ag::diffusion::q_sample rank-1 forward + gradient");
}

void test_q_sample_rank0_scalar() {
    // Rank-0 (scalar) input. Shape{}.numel() == 1 by the Shape
    // contract. q_sample must accept this shape and produce a scalar
    // output with the expected analytic value.
    ag::Variable xv(make_tensor({2.f}, ag::Shape{}), true);
    ag::Variable nv(make_tensor({0.5f}, ag::Shape{}), true);
    ag::Variable xt = ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv);
    CHECK((xt.value().shape() == ag::Shape{}));
    // y = 0.8 * 2 + 0.6 * 0.5 = 1.6 + 0.3 = 1.9.
    CHECK(std::fabs(read_values(xt.value())[0] - 1.9f) <= 1e-6f);

    ag::sum(xt).backward();
    // d_x0 = 0.8, d_noise = 0.6 (chain rule).
    CHECK(std::fabs(read_values(xv.grad())[0] - 0.8f) <= 1e-6f);
    CHECK(std::fabs(read_values(nv.grad())[0] - 0.6f) <= 1e-6f);
    report("ag::diffusion::q_sample rank-0 scalar forward + gradients");
}

void test_q_sample_zero_element_shape() {
    // Zero-element shape (rank-3 with one zero dim): empty result,
    // empty grads, no throw.
    ag::Variable xv(ag::Tensor::empty(ag::Shape{2, 3, 0}), true);
    ag::Variable nv(ag::Tensor::empty(ag::Shape{2, 3, 0}), true);
    ag::Variable xt = ag::diffusion::q_sample(xv, 0.8f, 0.6f, &nv);
    CHECK((xt.value().shape() == ag::Shape{2, 3, 0}));
    CHECK(xt.value().elements() == 0);
    ag::sum(xt).backward();
    CHECK(xv.has_grad());
    CHECK(nv.has_grad());
    CHECK(xv.grad().elements() == 0);
    CHECK(nv.grad().elements() == 0);
    report("ag::diffusion::q_sample zero-element shape (rank-3)");
}

// Legacy and replacement surfaces coexist in the same TU. This
// helper is invoked from main() once the public-API tests have
// already exercised the replacement. It includes the legacy umbrella
// header and calls into both the legacy ag::sinusoidal_time_embedding
// and the replacement ag::diffusion::sinusoidal_time_embedding. If
// the namespaces collide, compilation of this TU would have failed
// before reaching this point.
static void test_legacy_and_replacement_coexist() {
    {
        // Legacy ag::sinusoidal_time_embedding: rank-2 (1, dim).
        ag::VarPtr legacy_pe = ag::sinusoidal_time_embedding(
            /*t=*/3, /*dim=*/8);
        CHECK(legacy_pe->data.rows() == 1);
        CHECK(legacy_pe->data.cols() == 8);
        // Replacement ag::diffusion::sinusoidal_time_embedding: rank-1 (dim,).
        ag::Tensor replacement_pe = ag::diffusion::sinusoidal_time_embedding(
            /*t=*/3, /*dim=*/8);
        CHECK((replacement_pe.shape() == ag::Shape{8}));
        // Same value at the same position: legacy pe(0, 0) == sin(t),
        // replacement pe[0] == sin(t).
        CHECK(std::fabs(legacy_pe->data(0, 0) -
                        read_values(replacement_pe)[0]) <= 1e-6f);
    }
    report("legacy ag::sinusoidal_time_embedding and "
           "replacement ag::diffusion::sinusoidal_time_embedding coexist");
}

}  // namespace

int main() {
    test_randn_shape_rank4();
    test_randn_stats();
    test_randn_reproducible_same_seed();
    test_randn_different_seeds_independent();
    test_randn_like_shape();
    test_randn_invalid_arguments();

    test_time_emb_shape_even();
    test_time_emb_shape_odd();
    test_time_emb_period();
    test_time_emb_range_even();
    test_time_emb_invalid_arguments();

    test_q_sample_shape_rank4();
    test_q_sample_zero_noise_equals_scale();
    test_q_sample_grad_x0();
    test_q_sample_grad_noise();
    test_q_sample_grad_rank2();
    test_q_sample_default_noise_uses_seed();
    test_q_sample_invalid_arguments();
    test_q_sample_rank1_forward_and_gradient();
    test_q_sample_rank0_scalar();
    test_q_sample_zero_element_shape();

    test_legacy_and_replacement_coexist();

    std::printf("\nALL OOP DIFFUSION TESTS PASSED (%d)\n", passed);
    return 0;
}
