// Replacement consumer tests for the Tensor/Variable GroupNorm API.
//
// Translation unit compiles against the new headers under
// autograd/core and the public umbrella autograd.h. It does not depend
// on the legacy ag::GroupNorm / Var/Mat surface.
//
// Coverage:
//   * ag::group_norm forward matches an independent naive NCHW
//     reference, including affine (gamma, beta) and zero-mean
//     unit-variance per-group statistics when gamma=1, beta=0.
//   * Output shape metadata (N, C, H, W) is preserved.
//   * Numerical gradient checks against central finite differences for
//     input, gamma, and beta.
//   * ag::nn::GroupNorm parameter names and order, zero_grad, and
//     alias-visible SGD update.
//   * Invalid argument paths: non-rank-4 input, non-rank-1 gamma/beta,
//     channel-not-divisible-by-groups, non-positive groups, non-finite
//     or non-positive epsilon.
//   * Public-header hygiene: the consumer TU must compile without ever
//     pulling Eigen or CUDA macros through the new headers.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"
#include "autograd/core/optim.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Public module/ops headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Public module/ops headers must not include CUDA runtime"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
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

void check_near(const std::vector<float>& a, const std::vector<float>& b,
                float tol = 1e-4f) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(std::fabs(a[i] - b[i]) <= tol);
    }
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

// Central finite-difference gradient for a scalar-output op. Mutates
// the parameter `data` in place via its alias `param_alias`, evaluates
// f(data) for the +/- perturbations, and returns the gradient.
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

// Naive NCHW GroupNorm reference. Input (N, C, H, W), gamma/beta (C,),
// num_groups divides C, eps > 0. Returns flat (N*C*H*W) in row-major,
// last-axis-contiguous order.
std::vector<float> group_norm_reference(
        const std::vector<float>& in, int N, int C, int H, int W,
        const std::vector<float>& gamma, const std::vector<float>& beta,
        int num_groups, float eps) {
    const int ch_per_g = C / num_groups;
    const int M = ch_per_g * H * W;
    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    const int out_stride_n = C * H * W;
    const int out_stride_c = H * W;
    const int out_stride_h = W;
    const int out_stride_w = 1;
    std::vector<float> out(static_cast<std::size_t>(N) * C * H * W, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int ch0 = g * ch_per_g;
            double mean = 0.0;
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        mean += static_cast<double>(
                            in[n * in_stride_n
                             + c * in_stride_c
                             + h * in_stride_h
                             + w * in_stride_w]);
                    }
                }
            }
            mean /= static_cast<double>(M);
            double var = 0.0;
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const double diff = static_cast<double>(
                            in[n * in_stride_n
                             + c * in_stride_c
                             + h * in_stride_h
                             + w * in_stride_w]) - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= static_cast<double>(M);
            const double inv_std = 1.0 /
                std::sqrt(var + static_cast<double>(eps));
            for (int c = ch0; c < ch0 + ch_per_g; ++c) {
                const float gc = gamma[c];
                const float bc = beta[c];
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const double diff = static_cast<double>(
                            in[n * in_stride_n
                             + c * in_stride_c
                             + h * in_stride_h
                             + w * in_stride_w]) - mean;
                        const float xn = static_cast<float>(diff * inv_std);
                        out[n * out_stride_n
                          + c * out_stride_c
                          + h * out_stride_h
                          + w * out_stride_w] = gc * xn + bc;
                    }
                }
            }
        }
    }
    return out;
}

void test_group_norm_forward_matches_naive() {
    const int N = 2, C = 4, H = 3, W = 3, G = 2;
    const float eps = 1e-5f;
    const int ch_per_g = C / G;
    std::mt19937 rng(0x6c1f'a8d2u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> in(N * C * H * W);
    std::vector<float> gamma(C, 1.f);
    std::vector<float> beta(C, 0.f);
    // Mildly non-trivial affine: gamma!=1, beta!=0.
    for (int c = 0; c < C; ++c) {
        gamma[c] = 0.5f + 0.1f * static_cast<float>(c);
        beta[c]  = -0.2f + 0.05f * static_cast<float>(c);
    }
    for (auto& v : in) v = dist(rng);

    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable gv(make_tensor(gamma, ag::Shape{C}), true);
    ag::Variable bv(make_tensor(beta, ag::Shape{C}), true);
    ag::Variable y = ag::group_norm(xv, gv, bv, G, eps);
    CHECK((y.value().shape() == ag::Shape{N, C, H, W}));

    std::vector<float> got = read_values(y.value());
    std::vector<float> ref = group_norm_reference(
        in, N, C, H, W, gamma, beta, G, eps);
    check_close(got, ref, 1e-4f);

    // Sanity: per-group mean and variance with gamma=1, beta=0 must
    // be ~0 and ~1 respectively.
    ag::Variable xv2(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable ones(make_tensor(std::vector<float>(C, 1.f), ag::Shape{C}),
                       true);
    ag::Variable zeros(make_tensor(std::vector<float>(C, 0.f), ag::Shape{C}),
                        true);
    ag::Variable y2 = ag::group_norm(xv2, ones, zeros, G, eps);
    std::vector<float> y2v = read_values(y2.value());
    const int in_stride_n = C * H * W;
    const int in_stride_c = H * W;
    const int in_stride_h = W;
    const int in_stride_w = 1;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            double mean = 0.0;
            for (int c = g * ch_per_g; c < (g + 1) * ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        mean += y2v[n * in_stride_n
                                   + c * in_stride_c
                                   + h * in_stride_h
                                   + w * in_stride_w];
                    }
                }
            }
            mean /= static_cast<double>(ch_per_g * H * W);
            double var = 0.0;
            for (int c = g * ch_per_g; c < (g + 1) * ch_per_g; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const double d = y2v[n * in_stride_n
                                          + c * in_stride_c
                                          + h * in_stride_h
                                          + w * in_stride_w] - mean;
                        var += d * d;
                    }
                }
            }
            var /= static_cast<double>(ch_per_g * H * W);
            CHECK(std::fabs(mean) < 1e-4f);
            CHECK(std::fabs(var - 1.0) < 1e-3f);
        }
    }
    report("ag::group_norm forward matches naive reference");
}

void test_group_norm_gradients() {
    const int N = 1, C = 4, H = 4, W = 4, G = 2;
    const float eps = 1e-5f;
    std::mt19937 rng(0x77d3'4b91u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto fresh = [&](ag::Variable& in_var, ag::Variable& gamma_var,
                     ag::Variable& beta_var) {
        std::vector<float> in(N * C * H * W);
        std::vector<float> gamma(C);
        std::vector<float> beta(C);
        for (auto& v : in) v = dist(rng);
        for (auto& v : gamma) v = dist(rng);
        for (auto& v : beta) v = dist(rng);
        in_var = ag::Variable(make_tensor(in, ag::Shape{N, C, H, W}), true);
        gamma_var = ag::Variable(make_tensor(gamma, ag::Shape{C}), true);
        beta_var = ag::Variable(make_tensor(beta, ag::Shape{C}), true);
    };

    // d_input
    {
        ag::Variable xv, gv, bv;
        fresh(xv, gv, bv);
        auto f = [&] { return ag::group_norm(xv, gv, bv, G, eps); };
        auto fd = finite_difference(
            read_values(xv.value()), xv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(xv.grad()), fd, 5e-2f);
        report("ag::group_norm gradient: d_input matches FD");
    }
    // d_gamma
    {
        ag::Variable xv, gv, bv;
        fresh(xv, gv, bv);
        auto f = [&] { return ag::group_norm(xv, gv, bv, G, eps); };
        auto fd = finite_difference(
            read_values(gv.value()), gv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(gv.grad()), fd, 5e-2f);
        report("ag::group_norm gradient: d_gamma matches FD");
    }
    // d_beta
    {
        ag::Variable xv, gv, bv;
        fresh(xv, gv, bv);
        auto f = [&] { return ag::group_norm(xv, gv, bv, G, eps); };
        auto fd = finite_difference(
            read_values(bv.value()), bv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(bv.grad()), fd, 5e-2f);
        report("ag::group_norm gradient: d_beta matches FD");
    }
}

void test_group_norm_module_named_parameters() {
    ag::nn::GroupNorm gn(2, 4, 1e-5f);
    auto named = gn.named_parameters();
    CHECK(named.size() == 2);
    CHECK(named[0].name == "weight");
    CHECK(named[1].name == "bias");
    CHECK((named[0].parameter.value().shape() == ag::Shape{4}));
    CHECK((named[1].parameter.value().shape() == ag::Shape{4}));
    CHECK(named[0].parameter.requires_grad());
    CHECK(named[1].parameter.requires_grad());

    auto named2 = gn.named_parameters();
    CHECK(named2[0].name == "weight");
    CHECK(named2[1].name == "bias");

    auto params = gn.parameters();
    CHECK(params.size() == 2);
    report("ag::nn::GroupNorm parameter names and shapes (weight, bias)");
}

void test_group_norm_module_forward_backward() {
    const int N = 2, C = 4, H = 5, W = 5, G = 2;
    const float eps = 1e-5f;
    ag::nn::GroupNorm gn(G, C, eps);
    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.05f * static_cast<float>((i * 13) % 7) - 0.2f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = gn.forward(xv);
    CHECK((y.value().shape() == ag::Shape{N, C, H, W}));

    ag::sum(y).backward();
    CHECK(gn.weight().has_grad());
    CHECK(gn.bias().has_grad());
    CHECK((gn.weight().grad().shape() == ag::Shape{C}));
    CHECK((gn.bias().grad().shape() == ag::Shape{C}));

    // Alias-visible SGD step.
    std::vector<float> weight_before = read_values(gn.weight().value());
    ag::optim::SGD sgd(gn.parameters(), 0.01f);
    sgd.step();
    std::vector<float> weight_after = read_values(gn.weight().value());
    bool moved = false;
    for (std::size_t i = 0; i < weight_before.size(); ++i) {
        if (std::fabs(weight_after[i] - weight_before[i]) > 1e-7f) {
            moved = true;
            break;
        }
    }
    CHECK(moved);
    report("ag::nn::GroupNorm forward, backward, SGD alias-visible");
}

void test_group_norm_zero_grad_recurses() {
    ag::nn::GroupNorm gn(2, 4, 1e-5f);
    std::vector<float> in(2 * 4 * 3 * 3, 0.5f);
    ag::Variable xv(make_tensor(in, ag::Shape{2, 4, 3, 3}), true);
    ag::sum(gn.forward(xv)).backward();
    CHECK(gn.weight().has_grad());
    CHECK(gn.bias().has_grad());

    gn.zero_grad();
    CHECK(!gn.weight().has_grad());
    CHECK(!gn.bias().has_grad());
    report("ag::nn::GroupNorm zero_grad clears registered parameters");
}

void test_group_norm_backward_input_uses_saved_gamma_snapshot() {
    // Regression: ag::group_norm backward-input must use the
    // forward-time gamma snapshot, not the live gamma. If a caller
    // mutates gamma storage between forward and backward (e.g. via
    // an optimizer step on gamma), d_input must still be computed
    // against the original affine weights. This test:
    //   1. Builds a graph with a nontrivial gamma.
    //   2. Computes a reference d_input via an independent second
    //      graph whose gamma Tensor is a deep clone of the first
    //      gamma (so backward and reference are guaranteed to be
    //      computed against the same forward-time gamma).
    //   3. Mutates the live gamma storage of the first graph after
    //      forward.
    //   4. Calls backward on the first graph and asserts d_input
    //      matches the reference rather than the post-mutation
    //      gamma.
    const int N = 1, C = 4, H = 3, W = 3, G = 2;
    const float eps = 1e-5f;
    std::mt19937 rng(0x12c4'8f73u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> in(N * C * H * W);
    std::vector<float> gamma(C);
    std::vector<float> beta(C, 0.f);
    for (auto& v : in) v = dist(rng);
    for (auto& v : gamma) v = dist(rng);

    // Reference graph: clone gamma into a fresh Tensor so backward
    // operates on the same forward-time values as the test graph.
    ag::Variable in_ref(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable g_ref(make_tensor(gamma, ag::Shape{C}), true);
    ag::Variable b_ref(make_tensor(beta, ag::Shape{C}), true);
    ag::Variable y_ref = ag::group_norm(in_ref, g_ref, b_ref, G, eps);
    ag::sum(y_ref).backward();
    std::vector<float> expected_d_in = read_values(in_ref.grad());

    // Test graph: keep the live gamma variable. Mutate its storage
    // after forward, before backward, to a deliberately wrong value.
    ag::Variable in_var(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable g_var(make_tensor(gamma, ag::Shape{C}), true);
    ag::Variable b_var(make_tensor(beta, ag::Shape{C}), true);
    ag::Variable y_var = ag::group_norm(in_var, g_var, b_var, G, eps);

    // Confirm we are not accidentally sharing the bug: the reference
    // and test graphs started with identical gamma contents.
    {
        std::vector<float> ref_gamma = read_values(g_ref.value());
        std::vector<float> test_gamma = read_values(g_var.value());
        CHECK(ref_gamma == test_gamma);
    }

    // Scramble the live gamma storage. If backward-input incorrectly
    // reads g_var.value() at backward time, d_input will reflect
    // these scrambled weights rather than the original ones.
    std::vector<float> scrambled(C);
    for (int c = 0; c < C; ++c) scrambled[c] = 100.f + static_cast<float>(c);
    ag::Tensor live_gamma = g_var.value();
    live_gamma.copy_from_host(scrambled.data(), scrambled.size());

    // Backward must use the saved gamma snapshot, not the scrambled
    // live tensor.
    ag::sum(y_var).backward();
    std::vector<float> got_d_in = read_values(in_var.grad());
    check_near(got_d_in, expected_d_in, 1e-4f);

    // Sanity: the live gamma really was scrambled by the mutation.
    std::vector<float> post_gamma = read_values(g_var.value());
    CHECK(post_gamma == scrambled);

    report("ag::group_norm backward-input uses saved gamma snapshot");
}

void test_group_norm_invalid_arguments() {
    // Non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        ag::Variable gv(make_tensor({1.f, 1.f}, ag::Shape{2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 1e-5f),
                        std::invalid_argument);
    }
    // Rank-2 gamma rejected (only rank-1 (C,) is accepted).
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 4 * 3 * 3, 0.f),
                                    ag::Shape{1, 4, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{1, 4}),
                         true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f, 0.f}, ag::Shape{4}),
                         true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 1e-5f),
                        std::invalid_argument);
    }
    // Gamma length mismatch.
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 4 * 3 * 3, 0.f),
                                    ag::Shape{1, 4, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f}, ag::Shape{2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f, 0.f}, ag::Shape{4}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 1e-5f),
                        std::invalid_argument);
    }
    // Beta length mismatch.
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 4 * 3 * 3, 0.f),
                                    ag::Shape{1, 4, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{4}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 1e-5f),
                        std::invalid_argument);
    }
    // Channel not divisible by groups.
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 3 * 3 * 3, 0.f),
                                    ag::Shape{1, 3, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f, 1.f}, ag::Shape{3}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f}, ag::Shape{3}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 1e-5f),
                        std::invalid_argument);
    }
    // Non-positive groups.
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 4 * 3 * 3, 0.f),
                                    ag::Shape{1, 4, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{4}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f, 0.f}, ag::Shape{4}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 0, 1e-5f),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, -1, 1e-5f),
                        std::invalid_argument);
    }
    // Eps invalid (non-positive or non-finite).
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 4 * 3 * 3, 0.f),
                                    ag::Shape{1, 4, 3, 3}), true);
        ag::Variable gv(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{4}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f, 0.f}, ag::Shape{4}), true);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, 0.f),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, -1e-5f),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2,
                                        std::numeric_limits<float>::infinity()),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::group_norm(xv, gv, bv, 2, std::nanf("")),
                        std::invalid_argument);
    }
    // nn::GroupNorm constructor validation.
    CHECK_THROWS_AS(ag::nn::GroupNorm(0, 4, 1e-5f), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 0, 1e-5f), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 3, 1e-5f), std::invalid_argument); // not divisible
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 4, 0.f), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 4, -1e-5f), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 4,
                                     std::numeric_limits<float>::infinity()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::GroupNorm(2, 4, std::nanf("")),
                    std::invalid_argument);

    // nn::GroupNorm rejects channel mismatch on forward.
    {
        ag::nn::GroupNorm gn(2, 4, 1e-5f);
        ag::Variable xv(make_tensor(std::vector<float>(1 * 3 * 5 * 5, 0.f),
                                    ag::Shape{1, 3, 5, 5}), true);
        CHECK_THROWS_AS(gn.forward(xv), std::invalid_argument);
    }
    report("ag::group_norm / ag::nn::GroupNorm reject invalid arguments");
}

}  // namespace

int main() {
    test_group_norm_forward_matches_naive();
    test_group_norm_gradients();
    test_group_norm_module_named_parameters();
    test_group_norm_module_forward_backward();
    test_group_norm_zero_grad_recurses();
    test_group_norm_backward_input_uses_saved_gamma_snapshot();
    test_group_norm_invalid_arguments();

    std::printf("\nALL OOP GROUPNORM TESTS PASSED (%d)\n", passed);
    return 0;
}
