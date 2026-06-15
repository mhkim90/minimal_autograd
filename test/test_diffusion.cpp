// test_diffusion.cpp — checks for diffusion.h primitives.
//
// Covers:
//   - randn / randn_like: shape, mean ~ 0, std ~ 1, leaf semantics
//   - sinusoidal_time_embedding: shape, period, value range
//   - q_sample: grad_check through x0, through noise (when supplied as Var)

#include "autograd.h"
#include "grad_check.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace ag;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++passed; }
    else     { std::printf("  FAIL  %s\n", name); ++failed; }
}

// ── randn / randn_like ──────────────────────────────────────────────────────

static void test_randn_shape() {
    auto x = randn(3, 5, /*seed=*/7);
    check("randn shape", x->data.rows() == 3 && x->data.cols() == 5);
    check("randn leaf (no parents)", x->parents.empty());
    check("randn leaf (no back_fn)", !x->back_fn);
}

static void test_randn_stats() {
    // Mean ~ 0, std ~ 1 over a large sample, deterministic for a fixed seed.
    auto x = randn(1, 10000, /*seed=*/42);
    float m = x->data.mean();
    Mat centered = x->data.array() - m;
    float s = std::sqrt((centered.array().square()).sum() / 10000.f);
    check("randn mean ~ 0",   std::fabs(m) < 0.05f);
    check("randn std ~ 1",    std::fabs(s - 1.f) < 0.05f);
}

static void test_randn_like_shape() {
    auto x = Var::make(Mat::Zero(4, 7));
    auto n = randn_like(x, /*seed=*/1);
    check("randn_like shape matches", n->data.rows() == 4 && n->data.cols() == 7);
}

static void test_randn_reproducible() {
    auto a = randn(2, 8, /*seed=*/123);
    auto b = randn(2, 8, /*seed=*/123);
    bool eq = (a->data - b->data).cwiseAbs().maxCoeff() < 1e-6f;
    check("randn deterministic for fixed seed", eq);
}

// ── sinusoidal_time_embedding ───────────────────────────────────────────────

static void test_time_emb_shape() {
    auto pe = sinusoidal_time_embedding(/*t=*/10, /*dim=*/16);
    check("time_emb shape (1, dim)", pe->data.rows() == 1 && pe->data.cols() == 16);
    check("time_emb leaf (no parents)", pe->parents.empty());
    check("time_emb leaf (no back_fn)", !pe->back_fn);
}

static void test_time_emb_range() {
    auto pe = sinusoidal_time_embedding(/*t=*/5, /*dim=*/32);
    float lo = pe->data.minCoeff();
    float hi = pe->data.maxCoeff();
    check("time_emb values in [-1, 1]", lo >= -1.f - 1e-6f && hi <= 1.f + 1e-6f);
}

static void test_time_emb_period() {
    // freq_0 = exp(0) = 1, so the first dim should be sin(t) and cos(t).
    auto pe = sinusoidal_time_embedding(/*t=*/7, /*dim=*/8);
    int half = 4;
    float sv = pe->data(0, 0);
    float cv = pe->data(0, half);
    check("time_emb pe[0] = sin(t)",
          std::fabs(sv - std::sin(7.f)) < 1e-5f);
    check("time_emb pe[half] = cos(t)",
          std::fabs(cv - std::cos(7.f)) < 1e-5f);
}

// ── q_sample ────────────────────────────────────────────────────────────────

static void test_q_sample_shape() {
    auto x0 = Var::make(Mat::Random(2, 6));
    auto noise = Var::make(Mat::Random(2, 6));
    auto xt = q_sample(x0, /*t=*/100, 0.5f, 0.5f, noise);
    check("q_sample shape matches x0",
          xt->data.rows() == 2 && xt->data.cols() == 6);
}

static void test_q_sample_grad_through_x0() {
    // d/dx0 q_sample = sqrt_alpha_bar.
    // Noise must be a FRESH leaf on every f(x) call so its accumulated grad
    // does not bleed across the fwd/bwd probes that grad_check runs.
    Mat noise_src = Mat::Random(2, 4);
    auto x0 = Var::make(Mat::Random(2, 4));
    bool ok = grad_check(
        [&](VarPtr x) {
            auto n = Var::make(noise_src);  // fresh leaf, same data
            return sum(q_sample(x, /*t=*/0, 0.8f, 0.6f, n));
        },
        x0);
    check("q_sample grad w.r.t. x0 (analytic vs numeric)", ok);
}

static void test_q_sample_grad_through_noise() {
    // d/dn q_sample = sqrt_one_minus_alpha_bar.
    // x0 must be a fresh leaf per call for the same reason as above.
    Mat x0_src = Mat::Random(2, 4);
    auto n = Var::make(Mat::Random(2, 4));
    bool ok = grad_check(
        [&](VarPtr nv) {
            auto x0 = Var::make(x0_src);  // fresh leaf, same data
            return sum(q_sample(x0, /*t=*/0, 0.8f, 0.6f, nv));
        },
        n);
    check("q_sample grad w.r.t. noise (analytic vs numeric)", ok);
}

static void test_q_sample_zero_noise_equals_scale() {
    // With noise == 0 the output must equal sqrt_alpha_bar * x0.
    auto x0 = Var::make(Mat::Constant(1, 3, 2.f));
    Mat zero = Mat::Zero(1, 3);
    auto n = Var::make(zero);
    auto xt = q_sample(x0, 0, /*sqrt_ab=*/0.5f, /*sqrt_1mab=*/1.f, n);
    // Expected: 0.5 * 2.0 = 1.0
    bool ok = std::fabs(xt->data(0, 0) - 1.f) < 1e-6f &&
              std::fabs(xt->data(0, 1) - 1.f) < 1e-6f &&
              std::fabs(xt->data(0, 2) - 1.f) < 1e-6f;
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
