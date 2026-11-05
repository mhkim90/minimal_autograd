// test_diffusion.cpp — checks for diffusion.h primitives.
//
// Covers:
//   - randn / randn_like: shape, mean ~ 0, std ~ 1, leaf semantics
//   - sinusoidal_time_embedding: shape, period, value range
//   - q_sample: grad_check through x0, through noise (when supplied as a Variable)

#include "autograd/core/diffusion.h"
#include "grad_check.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace ag;
using namespace ag_test;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++passed; }
    else     { std::printf("  FAIL  %s\n", name); ++failed; }
}

// ── randn / randn_like ──────────────────────────────────────────────────────

static void test_randn_shape() {
    auto x = diffusion::randn(Shape{3, 5}, /*seed=*/7);
    check("randn shape", x.shape()[0] == 3 && x.shape()[1] == 5);
    check("randn is a Tensor utility", x.device().is_cpu());
}

static void test_randn_stats() {
    // Mean ~ 0, std ~ 1 over a large sample, deterministic for a fixed seed.
    auto x = diffusion::randn(Shape{1, 10000}, /*seed=*/42);
    float m = host_mean(x);
    float s = std::sqrt(host_variance(x, m));
    check("randn mean ~ 0",   std::fabs(m) < 0.05f);
    check("randn std ~ 1",    std::fabs(s - 1.f) < 0.05f);
}

static void test_randn_like_shape() {
    auto x = Variable(Tensor::zeros(Shape{4, 7}), true);
    auto n = diffusion::randn_like(x.value(), /*seed=*/1);
    check("randn_like shape matches", n.shape()[0] == 4 && n.shape()[1] == 7);
}

static void test_randn_reproducible() {
    auto a = diffusion::randn(Shape{2, 8}, /*seed=*/123);
    auto b = diffusion::randn(Shape{2, 8}, /*seed=*/123);
    bool eq = max_diff(a, b) < 1e-6f;
    check("randn deterministic for fixed seed", eq);
}

// ── sinusoidal_time_embedding ───────────────────────────────────────────────

static void test_time_emb_shape() {
    auto pe = diffusion::sinusoidal_time_embedding(/*t=*/10, /*dim=*/16);
    check("time_emb rank1 shape", pe.shape() == Shape{16});
    check("time_emb is a Tensor utility", pe.device().is_cpu());
}

static void test_time_emb_range() {
    auto pe = diffusion::sinusoidal_time_embedding(/*t=*/5, /*dim=*/32);
    float lo = min_value(pe);
    float hi = max_value(pe);
    check("time_emb values in [-1, 1]", lo >= -1.f - 1e-6f && hi <= 1.f + 1e-6f);
}

static void test_time_emb_period() {
    // freq_0 = exp(0) = 1, so the first dim should be sin(t) and cos(t).
    auto pe = diffusion::sinusoidal_time_embedding(/*t=*/7, /*dim=*/8);
    int half = 4;
    float sv = at(pe, 0);
    float cv = at(pe, half);
    check("time_emb pe[0] = sin(t)",
          std::fabs(sv - std::sin(7.f)) < 1e-5f);
    check("time_emb pe[half] = cos(t)",
          std::fabs(cv - std::cos(7.f)) < 1e-5f);
}

// ── q_sample ────────────────────────────────────────────────────────────────

static void test_q_sample_shape() {
    auto x0 = Variable(random_tensor(Shape{2, 6}), true);
    auto noise = Variable(random_tensor(Shape{2, 6}), true);
    auto xt = diffusion::q_sample(x0, 0.5f, 0.5f, &noise);
    check("q_sample shape matches x0",
          xt.value().shape()[0] == 2 && xt.value().shape()[1] == 6);
}

static void test_q_sample_grad_through_x0() {
    // d/dx0 q_sample = sqrt_alpha_bar.
    // Noise must be a FRESH leaf on every f(x) call so its accumulated grad
    // does not bleed across the fwd/bwd probes that grad_check runs.
    Tensor noise_src = random_tensor(Shape{2, 4});
    auto x0 = Variable(random_tensor(Shape{2, 4}), true);
    bool ok = grad_check(
        [&](Variable x) {
            auto n = Variable(noise_src, true);  // fresh leaf, same data
            return sum(diffusion::q_sample(x, 0.8f, 0.6f, &n));
        },
        x0);
    check("q_sample grad w.r.t. x0 (analytic vs numeric)", ok);
}

static void test_q_sample_grad_through_noise() {
    // d/dn q_sample = sqrt_one_minus_alpha_bar.
    // x0 must be a fresh leaf per call for the same reason as above.
    Tensor x0_src = random_tensor(Shape{2, 4});
    auto n = Variable(random_tensor(Shape{2, 4}), true);
    bool ok = grad_check(
        [&](Variable nv) {
            auto x0 = Variable(x0_src, true);  // fresh leaf, same data
            return sum(diffusion::q_sample(x0, 0.8f, 0.6f, &nv));
        },
        n);
    check("q_sample grad w.r.t. noise (analytic vs numeric)", ok);
}

static void test_q_sample_zero_noise_equals_scale() {
    // With noise == 0 the output must equal sqrt_alpha_bar * x0.
    auto x0 = Variable(filled(Shape{1, 3}, 2.f), true);
    Tensor zero = Tensor::zeros(Shape{1, 3});
    auto n = Variable(zero, true);
    auto xt = diffusion::q_sample(x0, /*sqrt_ab=*/0.5f, /*sqrt_1mab=*/1.f, &n);
    // Expected: 0.5 * 2.0 = 1.0
    bool ok = std::fabs(at(xt.value(), 0, 0) - 1.f) < 1e-6f &&
              std::fabs(at(xt.value(), 0, 1) - 1.f) < 1e-6f &&
              std::fabs(at(xt.value(), 0, 2) - 1.f) < 1e-6f;
    check("q_sample with zero noise = scale(x0, sqrt_ab)", ok);
}

int main() {
    std::printf("=== diffusion.h tests ===\n");

    std::printf("[randn / randn_like]\n");
    test_randn_shape();
    test_randn_stats();
    test_randn_like_shape();
    test_randn_reproducible();

    std::printf("[sinusoidal_time_embedding]\n");
    test_time_emb_shape();
    test_time_emb_range();
    test_time_emb_period();

    std::printf("[q_sample]\n");
    test_q_sample_shape();
    test_q_sample_grad_through_x0();
    test_q_sample_grad_through_noise();
    test_q_sample_zero_noise_equals_scale();

    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return failed == 0 ? 0 : 1;
}
