// Replacement consumer tests for the Tensor/Variable Conv2d and MaxPool2d
// APIs. The translation unit compiles against the new headers under
// autograd/core and the public umbrella autograd.h; it does not depend on
// the legacy Var/Mat Conv2d/MaxPool2d surface.
//
// Coverage:
//   * ag::conv2d / ag::max_pool2d free-function forward against an
//     independent naive nested-loop reference.
//   * Output shape metadata: (N, out_C, oH, oW) and (N, C, oH, oW).
//   * Numerical gradient checks against central finite differences for
//     the conv input, weight, and bias and for the pool input.
//   * ag::nn::Conv2d / ag::nn::MaxPool2d parameter names and order.
//   * Pool backward overlap accumulation and tie-determinism behaviour.
//   * Invalid argument paths: non-rank-4 input, channel mismatch, bad
//     kernel/stride/pad geometry, non-trainable conv parameters.
//   * Public-header hygiene: the consumer TU must compile without ever
//     pulling Eigen or CUDA macros through the new headers.

#include "autograd/core/module.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
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

// Naive NCHW Conv2d reference. Input (N, C, H, W), weight (OC, C, kH, kW),
// bias (OC,). Returns flat (N * OC * oH * oW) vector in the same
// first-axis-contiguous storage order the replacement Tensor API uses
// (stride[0]=1, stride[i]=stride[i-1]*shape[i-1]). The host buffer
// handed to Tensor::from_host is read with the same stride pattern, so
// the reference output can be compared element-wise against
// Tensor::copy_to_host.
std::vector<float> conv2d_reference(
        const std::vector<float>& in, int N, int C, int H, int W,
        const std::vector<float>& w, int OC, int kH, int kW,
        const std::vector<float>& b,
        int stride, int pad) {
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;
    const int in_stride_n = 1;
    const int in_stride_c = N;
    const int in_stride_h = N * C;
    const int in_stride_w = N * C * H;
    const int w_stride_oc = 1;
    const int w_stride_c  = OC;
    const int w_stride_kh = OC * C;
    const int w_stride_kw = OC * C * kH;
    const int out_stride_n = 1;
    const int out_stride_oc = N;
    const int out_stride_oh = N * OC;
    const int out_stride_ow = N * OC * oH;
    std::vector<float> out(static_cast<std::size_t>(N) * OC * oH * oW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float s = b[oc];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int ih = oh * stride + kh - pad;
                                const int iw = ow * stride + kw - pad;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    continue;
                                const int in_off = n * in_stride_n
                                                  + c * in_stride_c
                                                  + ih * in_stride_h
                                                  + iw * in_stride_w;
                                const int w_off = oc * w_stride_oc
                                                 + c * w_stride_c
                                                 + kh * w_stride_kh
                                                 + kw * w_stride_kw;
                                s += w[w_off] * in[in_off];
                            }
                        }
                    }
                    const int out_off = n * out_stride_n
                                      + oc * out_stride_oc
                                      + oh * out_stride_oh
                                      + ow * out_stride_ow;
                    out[out_off] = s;
                }
            }
        }
    }
    return out;
}

void test_conv2d_forward_matches_naive() {
    const int N = 2, C = 3, H = 5, W = 5;
    const int OC = 4, kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;

    std::mt19937 rng(0x9e37'79b9u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> in(N * C * H * W);
    std::vector<float> w(OC * C * kH * kW);
    std::vector<float> b(OC);
    for (auto& v : in) v = dist(rng);
    for (auto& v : w)  v = dist(rng);
    for (auto& v : b)  v = dist(rng);

    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable wv(make_tensor(w, ag::Shape{OC, C, kH, kW}), true);
    ag::Variable bv(make_tensor(b, ag::Shape{OC}), true);

    ag::Variable y = ag::conv2d(xv, wv, bv, stride, pad);
    CHECK((y.value().shape() == ag::Shape{N, OC, oH, oW}));

    std::vector<float> got = read_values(y.value());
    std::vector<float> ref = conv2d_reference(in, N, C, H, W,
                                              w, OC, kH, kW, b,
                                              stride, pad);
    check_close(got, ref, 1e-4f);
    report("ag::conv2d forward matches naive NCHW reference");
}

void test_conv2d_output_shape_metadata() {
    // stride=2, pad=0 — output spatial dims must shrink exactly.
    const int N = 2, C = 2, H = 7, W = 7, OC = 3, kH = 3, kW = 3;
    const int stride = 2, pad = 0;
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;
    ag::Variable xv(make_tensor(std::vector<float>(N * C * H * W, 0.5f),
                                ag::Shape{N, C, H, W}), true);
    ag::Variable wv(make_tensor(std::vector<float>(OC * C * kH * kW, 0.25f),
                                ag::Shape{OC, C, kH, kW}), true);
    ag::Variable bv(make_tensor(std::vector<float>(OC, 0.1f),
                                ag::Shape{OC}), true);
    ag::Variable y = ag::conv2d(xv, wv, bv, stride, pad);
    CHECK((y.value().shape() == ag::Shape{N, OC, oH, oW}));
    report("ag::conv2d output shape (N,OC,oH,oW)");
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

void test_conv2d_gradients() {
    const int N = 1, C = 2, H = 4, W = 4, OC = 2, kH = 3, kW = 3;
    const int stride = 1, pad = 0;
    std::mt19937 rng(0x12c4'8f73u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto fresh = [&](ag::Variable& in_var, ag::Variable& w_var,
                     ag::Variable& b_var) {
        std::vector<float> in(N * C * H * W);
        std::vector<float> w(OC * C * kH * kW);
        std::vector<float> b(OC);
        for (auto& v : in) v = dist(rng);
        for (auto& v : w)  v = dist(rng);
        for (auto& v : b)  v = dist(rng);
        in_var = ag::Variable(make_tensor(in, ag::Shape{N, C, H, W}), true);
        w_var  = ag::Variable(make_tensor(w, ag::Shape{OC, C, kH, kW}), true);
        b_var  = ag::Variable(make_tensor(b, ag::Shape{OC}), true);
    };

    // d_input
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(xv.value()), xv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(xv.grad()), fd, 5e-2f);
        report("ag::conv2d gradient: d_input matches FD");
    }

    // d_weight
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(wv.value()), wv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(wv.grad()), fd, 5e-2f);
        report("ag::conv2d gradient: d_weight matches FD");
    }

    // d_bias
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(bv.value()), bv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(bv.grad()), fd, 5e-2f);
        report("ag::conv2d gradient: d_bias matches FD");
    }
}

void test_conv2d_module_named_parameters() {
    ag::nn::Conv2d conv(2, 3, 3, 3, 1, 1);
    auto named = conv.named_parameters();
    CHECK(named.size() == 2);
    CHECK(named[0].name == "weight");
    CHECK(named[1].name == "bias");
    CHECK((named[0].parameter.value().shape() == ag::Shape{3, 2, 3, 3}));
    CHECK((named[1].parameter.value().shape() == ag::Shape{3}));
    CHECK(named[0].parameter.requires_grad());
    CHECK(named[1].parameter.requires_grad());

    // Repeated calls must produce identical order.
    auto named2 = conv.named_parameters();
    CHECK(named2.size() == 2);
    CHECK(named2[0].name == "weight");
    CHECK(named2[1].name == "bias");

    // parameters() must reflect registered leaves.
    auto params = conv.parameters();
    CHECK(params.size() == 2);
    CHECK((params[0].value().shape() == ag::Shape{3, 2, 3, 3}));
    CHECK((params[1].value().shape() == ag::Shape{3}));
    report("ag::nn::Conv2d parameter names and shapes (weight, bias)");
}

void test_conv2d_module_forward_backward() {
    const int N = 2, C = 2, H = 6, W = 6, OC = 3, kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    ag::nn::Conv2d conv(C, OC, kH, kW, stride, pad);
    std::vector<float> in(N * C * H * W, 0.1f);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.05f * static_cast<float>((i * 13) % 7) - 0.2f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = conv.forward(xv);
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;
    CHECK((y.value().shape() == ag::Shape{N, OC, oH, oW}));

    ag::sum(y).backward();
    CHECK(conv.weight().has_grad());
    CHECK(conv.bias().has_grad());
    CHECK((conv.weight().grad().shape() == ag::Shape{OC, C, kH, kW}));
    CHECK((conv.bias().grad().shape() == ag::Shape{OC}));

    // SGD step over the module's parameters must produce alias-visible
    // updates on the conv weight Tensor.
    std::vector<float> weight_before = read_values(conv.weight().value());
    ag::optim::SGD sgd(conv.parameters(), 0.01f);
    sgd.step();
    std::vector<float> weight_after = read_values(conv.weight().value());
    bool moved = false;
    for (std::size_t i = 0; i < weight_before.size(); ++i) {
        if (std::fabs(weight_after[i] - weight_before[i]) > 1e-7f) {
            moved = true;
            break;
        }
    }
    CHECK(moved);
    report("ag::nn::Conv2d forward, backward, SGD alias-visible");
}

void test_maxpool2d_forward_matches_naive() {
    const int N = 2, C = 3, H = 4, W = 4, kH = 2, kW = 2;
    const int stride = 2;
    const int oH = (H - kH) / stride + 1;
    const int oW = (W - kW) / stride + 1;

    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>((i * 7 + 3) % 11) - 5.f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::max_pool2d(xv, kH, kW, stride);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    std::vector<float> got = read_values(y.value());
    const int in_stride_n = 1;
    const int in_stride_c = N;
    const int in_stride_h = N * C;
    const int in_stride_w = N * C * H;
    const int out_stride_n = 1;
    const int out_stride_c = N;
    const int out_stride_oh = N * C;
    const int out_stride_ow = N * C * oH;
    std::vector<float> ref(static_cast<std::size_t>(N) * C * oH * oW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            const int ih = oh * stride + kh;
                            const int iw = ow * stride + kw;
                            const float v =
                                in[n * in_stride_n
                                  + c * in_stride_c
                                  + ih * in_stride_h
                                  + iw * in_stride_w];
                            if (v > m) m = v;
                        }
                    }
                    ref[n * out_stride_n
                       + c * out_stride_c
                       + oh * out_stride_oh
                       + ow * out_stride_ow] = m;
                }
            }
        }
    }
    check_near(got, ref, 1e-6f);
    report("ag::max_pool2d forward matches naive reference");
}

void test_maxpool2d_gradient_overlap() {
    // Overlapping pool (stride=1, kernel=2): each input pixel can be the
    // argmax of several windows; its gradient must accumulate.
    const int N = 1, C = 1, H = 3, W = 3, kH = 2, kW = 2;
    const int stride = 1;
    const int oH = H - kH + 1, oW = W - kW + 1;

    // Single batch single channel 3x3:
    //   0 0 0
    //   0 9 0    <- pixel (1,1) is the max in 4 windows (NW,NE,SW,SE)
    //   0 0 0
    std::vector<float> in{
        0.f, 0.f, 0.f,
        0.f, 9.f, 0.f,
        0.f, 0.f, 0.f,
    };
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::max_pool2d(xv, kH, kW, stride);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    ag::sum(y).backward();
    std::vector<float> g = read_values(xv.grad());
    CHECK(g.size() == in.size());
    // Pixel (1,1) is in 4 windows; expected gradient = 4.
    CHECK(std::fabs(g[4] - 4.f) <= 1e-5f);
    // All other pixels are zero; their gradients are zero.
    float other_sum = 0.f;
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (i == 4) continue;
        other_sum += std::fabs(g[i]);
    }
    CHECK(other_sum <= 1e-5f);
    report("ag::max_pool2d overlap accumulates at the argmax");
}

void test_maxpool2d_gradient_tie_determinism() {
    // 2x2 pool over a 2x2 input where every value is identical. The
    // output equals the input value; the backward must route the
    // gradient to exactly one input position (the deterministic
    // argmax), and the gradient magnitude at that position must equal
    // 1. Other positions must receive zero.
    const int N = 1, C = 1, H = 2, W = 2, kH = 2, kW = 2, stride = 2;
    std::vector<float> in{1.f, 1.f, 1.f, 1.f};
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::max_pool2d(xv, kH, kW, stride);
    ag::sum(y).backward();
    std::vector<float> g = read_values(xv.grad());
    CHECK(g.size() == 4);
    float total = 0.f;
    int nonzero = 0;
    for (float gi : g) {
        total += gi;
        if (std::fabs(gi) > 1e-7f) ++nonzero;
    }
    CHECK(std::fabs(total - 1.f) <= 1e-5f);
    CHECK(nonzero == 1);
    report("ag::max_pool2d tie routes gradient to one deterministic argmax");
}

void test_maxpool2d_module_zero_parameters() {
    ag::nn::MaxPool2d pool(2, 2, 2);
    CHECK(pool.parameters().empty());
    CHECK(pool.named_parameters().empty());

    const int N = 1, C = 2, H = 4, W = 4;
    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = pool.forward(xv);
    CHECK((y.value().shape() == ag::Shape{N, C, 2, 2}));
    ag::sum(y).backward();
    CHECK(xv.has_grad());
    report("ag::nn::MaxPool2d forward, backward, zero parameters");
}

void test_conv2d_invalid_arguments() {
    // Non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 2 * 2 * 2, 1.f),
                                    ag::Shape{2, 2, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, 0), std::invalid_argument);
    }
    // Channel mismatch: input has C=2 channels, weight expects in_C=3
    {
        ag::Variable xv(make_tensor(std::vector<float>(2 * 2 * 3 * 3, 0.f),
                                    ag::Shape{2, 2, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 3 * 2 * 2, 0.f),
                                    ag::Shape{2, 3, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, 0), std::invalid_argument);
    }
    // Bad kernel geometry: kernel larger than input + pad
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(1 * 1 * 5 * 5, 0.f),
                                    ag::Shape{1, 1, 5, 5}), true);
        ag::Variable bv(make_tensor({0.f}, ag::Shape{1}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, 0), std::invalid_argument);
    }
    // Non-positive stride
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(1 * 1 * 2 * 2, 0.f),
                                    ag::Shape{1, 1, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f}, ag::Shape{1}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 0, 0), std::invalid_argument);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, -1, 0), std::invalid_argument);
    }
    // Negative pad
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(1 * 1 * 2 * 2, 0.f),
                                    ag::Shape{1, 1, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f}, ag::Shape{1}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, -1), std::invalid_argument);
    }
    // Rank-2 bias rejected (only rank-1 (OC,) is accepted).
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(1 * 1 * 2 * 2, 0.f),
                                    ag::Shape{1, 1, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{1, 2}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, 0), std::invalid_argument);
    }
    // Bias length mismatch.
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 1 * 2 * 2, 0.f),
                                    ag::Shape{2, 1, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f}, ag::Shape{3}), true);
        CHECK_THROWS_AS(ag::conv2d(xv, wv, bv, 1, 0), std::invalid_argument);
    }
    // nn::Conv2d with non-positive channels
    CHECK_THROWS_AS(ag::nn::Conv2d(0, 4, 3, 3, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Conv2d(2, 0, 3, 3, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Conv2d(2, 4, 0, 3, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Conv2d(2, 4, 3, 0, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Conv2d(2, 4, 3, 3, 0, 0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Conv2d(2, 4, 3, 3, 1, -1), std::invalid_argument);

    // nn::Conv2d rejects channel mismatch on forward.
    {
        ag::nn::Conv2d conv(2, 4, 3, 3, 1, 0);
        ag::Variable xv(make_tensor(std::vector<float>(1 * 3 * 5 * 5, 0.f),
                                    ag::Shape{1, 3, 5, 5}), true);
        CHECK_THROWS_AS(conv.forward(xv), std::invalid_argument);
    }
    report("ag::conv2d / ag::nn::Conv2d reject invalid arguments");
}

void test_conv2d_rank1_bias_backward_shape() {
    // Backward must return a rank-1 gradient whose shape equals the
    // rank-1 bias parent (OC,). After removing rank-2 bias support
    // from the public contract, only rank-1 bias gradients are
    // produced and propagated through Variable::grad().
    const int N = 1, C = 1, H = 3, W = 3, OC = 2, kH = 2, kW = 2;
    const int stride = 1, pad = 0;
    ag::Variable xv(make_tensor(std::vector<float>(N * C * H * W, 0.5f),
                                ag::Shape{N, C, H, W}), true);
    ag::Variable wv(make_tensor(std::vector<float>(OC * C * kH * kW, 0.25f),
                                ag::Shape{OC, C, kH, kW}), true);
    ag::Variable bv(make_tensor({0.1f, -0.2f}, ag::Shape{OC}), true);
    ag::Variable y = ag::conv2d(xv, wv, bv, stride, pad);
    ag::sum(y).backward();
    CHECK(bv.has_grad());
    CHECK((bv.grad().shape() == ag::Shape{OC}));
    report("ag::conv2d rank-1 bias backward returns rank-1 gradient");
}

void test_maxpool2d_invalid_arguments() {
    // Non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 2, 2, 2), std::invalid_argument);
    }
    // Non-positive kernel
    {
        ag::Variable xv(make_tensor(std::vector<float>(16, 0.f),
                                    ag::Shape{1, 1, 4, 4}), true);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 0, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 2, 0, 2), std::invalid_argument);
    }
    // Non-positive stride
    {
        ag::Variable xv(make_tensor(std::vector<float>(16, 0.f),
                                    ag::Shape{1, 1, 4, 4}), true);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 2, 2, 0), std::invalid_argument);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 2, 2, -1), std::invalid_argument);
    }
    // Kernel larger than input
    {
        ag::Variable xv(make_tensor(std::vector<float>(9, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        CHECK_THROWS_AS(ag::max_pool2d(xv, 4, 4, 1), std::invalid_argument);
    }
    // nn::MaxPool2d constructors
    CHECK_THROWS_AS(ag::nn::MaxPool2d(0, 2, 2), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::MaxPool2d(2, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::MaxPool2d(2, 2, 0), std::invalid_argument);
    report("ag::max_pool2d / ag::nn::MaxPool2d reject invalid arguments");
}

// ── DepthwiseConv2d, AvgPool2d, NearestUpsample2d ───────────────────

// Naive NCHW DepthwiseConv2d reference. Input (N, C, H, W),
// weight (C, kH, kW), bias (C,). One per-channel filter shared across
// batches. Returns flat (N * C * oH * oW) in first-axis-contiguous
// layout matching the replacement Tensor API.
std::vector<float> depthwise_conv2d_reference(
        const std::vector<float>& in, int N, int C, int H, int W,
        const std::vector<float>& w, int kH, int kW,
        const std::vector<float>& b,
        int stride, int pad) {
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;
    const int in_stride_n = 1;
    const int in_stride_c = N;
    const int in_stride_h = N * C;
    const int in_stride_w = N * C * H;
    const int w_stride_c = 1;
    const int w_stride_kh = C;
    const int w_stride_kw = C * kH;
    const int out_stride_n = 1;
    const int out_stride_c = N;
    const int out_stride_oh = N * C;
    const int out_stride_ow = N * C * oH;
    std::vector<float> out(static_cast<std::size_t>(N) * C * oH * oW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float s = b[c];
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            const int ih = oh * stride + kh - pad;
                            const int iw = ow * stride + kw - pad;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                continue;
                            const int in_off = n * in_stride_n
                                              + c * in_stride_c
                                              + ih * in_stride_h
                                              + iw * in_stride_w;
                            const int w_off = c * w_stride_c
                                             + kh * w_stride_kh
                                             + kw * w_stride_kw;
                            s += w[w_off] * in[in_off];
                        }
                    }
                    const int out_off = n * out_stride_n
                                      + c * out_stride_c
                                      + oh * out_stride_oh
                                      + ow * out_stride_ow;
                    out[out_off] = s;
                }
            }
        }
    }
    return out;
}

void test_depthwise_conv2d_forward_matches_naive() {
    const int N = 2, C = 3, H = 5, W = 5;
    const int kH = 3, kW = 3, stride = 1, pad = 1;
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;

    std::mt19937 rng(0x4d2a'6f1bu);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> in(N * C * H * W);
    std::vector<float> w(C * kH * kW);
    std::vector<float> b(C);
    for (auto& v : in) v = dist(rng);
    for (auto& v : w)  v = dist(rng);
    for (auto& v : b)  v = dist(rng);

    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable wv(make_tensor(w, ag::Shape{C, kH, kW}), true);
    ag::Variable bv(make_tensor(b, ag::Shape{C}), true);
    ag::Variable y = ag::depthwise_conv2d(xv, wv, bv, stride, pad);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    std::vector<float> got = read_values(y.value());
    std::vector<float> ref = depthwise_conv2d_reference(
        in, N, C, H, W, w, kH, kW, b, stride, pad);
    check_close(got, ref, 1e-4f);
    report("ag::depthwise_conv2d forward matches naive NCHW reference");
}

void test_depthwise_conv2d_gradients() {
    const int N = 1, C = 2, H = 4, W = 4, kH = 3, kW = 3;
    const int stride = 1, pad = 0;
    std::mt19937 rng(0x77d3'4b91u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto fresh = [&](ag::Variable& in_var, ag::Variable& w_var,
                     ag::Variable& b_var) {
        std::vector<float> in(N * C * H * W);
        std::vector<float> w(C * kH * kW);
        std::vector<float> b(C);
        for (auto& v : in) v = dist(rng);
        for (auto& v : w)  v = dist(rng);
        for (auto& v : b)  v = dist(rng);
        in_var = ag::Variable(make_tensor(in, ag::Shape{N, C, H, W}), true);
        w_var  = ag::Variable(make_tensor(w, ag::Shape{C, kH, kW}), true);
        b_var  = ag::Variable(make_tensor(b, ag::Shape{C}), true);
    };

    // d_input
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::depthwise_conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(xv.value()), xv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(xv.grad()), fd, 5e-2f);
        report("ag::depthwise_conv2d gradient: d_input matches FD");
    }
    // d_weight
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::depthwise_conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(wv.value()), wv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(wv.grad()), fd, 5e-2f);
        report("ag::depthwise_conv2d gradient: d_weight matches FD");
    }
    // d_bias
    {
        ag::Variable xv, wv, bv;
        fresh(xv, wv, bv);
        auto f = [&] { return ag::depthwise_conv2d(xv, wv, bv, stride, pad); };
        auto fd = finite_difference(
            read_values(bv.value()), bv.value(), f);
        ag::sum(f()).backward();
        check_close(read_values(bv.grad()), fd, 5e-2f);
        report("ag::depthwise_conv2d gradient: d_bias matches FD");
    }
}

void test_depthwise_conv2d_module_named_parameters() {
    ag::nn::DepthwiseConv2d dw(3, 3, 3, 1, 1);
    auto named = dw.named_parameters();
    CHECK(named.size() == 2);
    CHECK(named[0].name == "weight");
    CHECK(named[1].name == "bias");
    CHECK((named[0].parameter.value().shape() == ag::Shape{3, 3, 3}));
    CHECK((named[1].parameter.value().shape() == ag::Shape{3}));
    CHECK(named[0].parameter.requires_grad());
    CHECK(named[1].parameter.requires_grad());

    // Repeated calls produce the same order.
    auto named2 = dw.named_parameters();
    CHECK(named2.size() == 2);
    CHECK(named2[0].name == "weight");
    CHECK(named2[1].name == "bias");

    auto params = dw.parameters();
    CHECK(params.size() == 2);
    report("ag::nn::DepthwiseConv2d parameter names and shapes");
}

void test_depthwise_conv2d_module_forward_backward() {
    const int N = 1, C = 2, H = 5, W = 5, kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int oH = (H + 2 * pad - kH) / stride + 1;
    const int oW = (W + 2 * pad - kW) / stride + 1;
    ag::nn::DepthwiseConv2d dw(C, kH, kW, stride, pad);
    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.05f * static_cast<float>((i * 17) % 11) - 0.3f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = dw.forward(xv);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    ag::sum(y).backward();
    CHECK(dw.weight().has_grad());
    CHECK(dw.bias().has_grad());
    CHECK((dw.weight().grad().shape() == ag::Shape{C, kH, kW}));
    CHECK((dw.bias().grad().shape() == ag::Shape{C}));
    report("ag::nn::DepthwiseConv2d forward, backward, parameter gradients");
}

void test_avgpool2d_forward_matches_naive() {
    const int N = 2, C = 2, H = 4, W = 4, kH = 2, kW = 2;
    const int stride = 2;
    const int oH = (H - kH) / stride + 1;
    const int oW = (W - kW) / stride + 1;

    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.1f * static_cast<float>(i) - 0.5f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::avg_pool2d(xv, kH, kW, stride);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    std::vector<float> got = read_values(y.value());
    const int ksz = kH * kW;
    const float inv_k = 1.f / static_cast<float>(ksz);
    const int in_stride_n = 1;
    const int in_stride_c = N;
    const int in_stride_h = N * C;
    const int in_stride_w = N * C * H;
    const int out_stride_n = 1;
    const int out_stride_c = N;
    const int out_stride_oh = N * C;
    const int out_stride_ow = N * C * oH;
    std::vector<float> ref(static_cast<std::size_t>(N) * C * oH * oW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float s = 0.f;
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            const int ih = oh * stride + kh;
                            const int iw = ow * stride + kw;
                            s += in[n * in_stride_n
                                   + c * in_stride_c
                                   + ih * in_stride_h
                                   + iw * in_stride_w];
                        }
                    }
                    ref[n * out_stride_n
                       + c * out_stride_c
                       + oh * out_stride_oh
                       + ow * out_stride_ow] = s * inv_k;
                }
            }
        }
    }
    check_near(got, ref, 1e-6f);
    report("ag::avg_pool2d forward matches naive reference");
}

void test_avgpool2d_gradient_overlap() {
    // Overlapping avg pool (stride=1, kernel=2): each input pixel
    // participates in multiple windows; the gradient must be the sum
    // of (1/kH*kW) over each window containing it.
    const int N = 1, C = 1, H = 3, W = 3, kH = 2, kW = 2;
    const int stride = 1;
    const int ksz = kH * kW;
    const float inv_k = 1.f / static_cast<float>(ksz);

    std::vector<float> in(9, 1.f);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::avg_pool2d(xv, kH, kW, stride);
    CHECK((y.value().shape() == ag::Shape{N, C, 2, 2}));
    ag::sum(y).backward();
    std::vector<float> g = read_values(xv.grad());
    // The 4 corners (0,0), (0,2), (2,0), (2,2) belong to exactly one
    // window each -> gradient = 1/4.
    // The 4 edges (0,1), (1,0), (1,2), (2,1) belong to two windows
    // each -> gradient = 2/4 = 0.5.
    // The center (1,1) belongs to four windows -> gradient = 4/4 = 1.
    // First-axis-contig flat index for (n, c, h, w) is n + c*N + h*N*C
    // + w*N*C*H. With N=1, C=1, H=3, W=3: idx = h + 3*w.
    std::vector<float> expected = {
        inv_k,
        2.f * inv_k,
        inv_k,
        2.f * inv_k,
        4.f * inv_k,
        2.f * inv_k,
        inv_k,
        2.f * inv_k,
        inv_k,
    };
    check_near(g, expected, 1e-6f);
    report("ag::avg_pool2d overlap accumulates window-mean gradients");
}

void test_avgpool2d_gradient_finite_difference() {
    const int N = 1, C = 1, H = 4, W = 4, kH = 2, kW = 2, stride = 2;
    std::mt19937 rng(0x6c1f'a8d2u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> in(N * C * H * W);
    for (auto& v : in) v = dist(rng);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    auto f = [&] { return ag::avg_pool2d(xv, kH, kW, stride); };
    auto fd = finite_difference(read_values(xv.value()), xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("ag::avg_pool2d gradient matches FD");
}

void test_avgpool2d_module_zero_parameters() {
    ag::nn::AvgPool2d pool(2, 2, 2);
    CHECK(pool.parameters().empty());
    CHECK(pool.named_parameters().empty());

    const int N = 1, C = 2, H = 4, W = 4;
    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = pool.forward(xv);
    CHECK((y.value().shape() == ag::Shape{N, C, 2, 2}));
    ag::sum(y).backward();
    CHECK(xv.has_grad());
    report("ag::nn::AvgPool2d forward, backward, zero parameters");
}

void test_nearest_upsample2d_forward_matches_naive() {
    const int N = 2, C = 2, H = 3, W = 3, scale = 2;
    const int oH = H * scale, oW = W * scale;

    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.5f * static_cast<float>(i) - 1.f;
    }
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::nearest_upsample2d(xv, scale);
    CHECK((y.value().shape() == ag::Shape{N, C, oH, oW}));

    std::vector<float> got = read_values(y.value());
    const int in_stride_n = 1;
    const int in_stride_c = N;
    const int in_stride_h = N * C;
    const int in_stride_w = N * C * H;
    const int out_stride_n = 1;
    const int out_stride_c = N;
    const int out_stride_oh = N * C;
    const int out_stride_ow = N * C * oH;
    std::vector<float> ref(static_cast<std::size_t>(N) * C * oH * oW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const float v = in[n * in_stride_n
                                     + c * in_stride_c
                                     + h * in_stride_h
                                     + w * in_stride_w];
                    for (int sh = 0; sh < scale; ++sh) {
                        for (int sw = 0; sw < scale; ++sw) {
                            const int oh = h * scale + sh;
                            const int ow = w * scale + sw;
                            ref[n * out_stride_n
                              + c * out_stride_c
                              + oh * out_stride_oh
                              + ow * out_stride_ow] = v;
                        }
                    }
                }
            }
        }
    }
    check_near(got, ref, 1e-6f);
    report("ag::nearest_upsample2d forward matches naive reference");
}

void test_nearest_upsample2d_gradient_multiplicity() {
    // Each input pixel must receive gradient equal to the sum of
    // upstream gradients over the scale*scale output cells that
    // sampled it. With a uniform upstream gradient of 1.0 and
    // scale=3, each input pixel must accumulate scale*scale = 9.
    const int N = 1, C = 1, H = 2, W = 2, scale = 3;
    std::vector<float> in(H * W, 1.f);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = ag::nearest_upsample2d(xv, scale);
    CHECK((y.value().shape() == ag::Shape{N, C, 6, 6}));
    ag::sum(y).backward();
    std::vector<float> g = read_values(xv.grad());
    CHECK(g.size() == 4);
    const float expected = static_cast<float>(scale * scale);
    for (float gi : g) {
        CHECK(std::fabs(gi - expected) <= 1e-5f);
    }
    report("ag::nearest_upsample2d gradient accumulates scale*scale per pixel");
}

void test_nearest_upsample2d_gradient_finite_difference() {
    const int N = 1, C = 1, H = 3, W = 3, scale = 2;
    std::mt19937 rng(0x4e7f'3c81u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> in(N * C * H * W);
    for (auto& v : in) v = dist(rng);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    auto f = [&] { return ag::nearest_upsample2d(xv, scale); };
    auto fd = finite_difference(read_values(xv.value()), xv.value(), f);
    ag::sum(f()).backward();
    check_close(read_values(xv.grad()), fd, 5e-2f);
    report("ag::nearest_upsample2d gradient matches FD");
}

void test_nearest_upsample2d_module_zero_parameters() {
    ag::nn::NearestUpsample2d up(2);
    CHECK(up.parameters().empty());
    CHECK(up.named_parameters().empty());

    const int N = 1, C = 2, H = 3, W = 3;
    std::vector<float> in(N * C * H * W);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = 0.1f * static_cast<float>(i);
    ag::Variable xv(make_tensor(in, ag::Shape{N, C, H, W}), true);
    ag::Variable y = up.forward(xv);
    CHECK((y.value().shape() == ag::Shape{N, C, 6, 6}));
    ag::sum(y).backward();
    CHECK(xv.has_grad());
    report("ag::nn::NearestUpsample2d forward, backward, zero parameters");
}

void test_spatial_extra_invalid_arguments() {
    // ag::depthwise_conv2d: non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 2 * 2, 1.f),
                                    ag::Shape{2, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 1, 0),
                        std::invalid_argument);
    }
    // ag::depthwise_conv2d: rank-2 weight rejected (must be rank-3)
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 2 * 3 * 3, 0.f),
                                    ag::Shape{1, 2, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 2, 0.f),
                                    ag::Shape{2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f}, ag::Shape{2}), true);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 1, 0),
                        std::invalid_argument);
    }
    // ag::depthwise_conv2d: channel mismatch (input C != weight C)
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 2 * 3 * 3, 0.f),
                                    ag::Shape{1, 2, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(3 * 2 * 2, 1.f),
                                    ag::Shape{3, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f}, ag::Shape{3}), true);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 1, 0),
                        std::invalid_argument);
    }
    // ag::depthwise_conv2d: bias length mismatch
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 2 * 3 * 3, 0.f),
                                    ag::Shape{1, 2, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(2 * 2 * 2, 1.f),
                                    ag::Shape{2, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f, 0.f, 0.f}, ag::Shape{3}), true);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 1, 0),
                        std::invalid_argument);
    }
    // ag::depthwise_conv2d: non-positive stride / negative pad
    {
        ag::Variable xv(make_tensor(std::vector<float>(1 * 1 * 3 * 3, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        ag::Variable wv(make_tensor(std::vector<float>(1 * 2 * 2, 1.f),
                                    ag::Shape{1, 2, 2}), true);
        ag::Variable bv(make_tensor({0.f}, ag::Shape{1}), true);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 0, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, -1, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::depthwise_conv2d(xv, wv, bv, 1, -1),
                        std::invalid_argument);
    }
    // nn::DepthwiseConv2d constructor validation
    CHECK_THROWS_AS(ag::nn::DepthwiseConv2d(0, 2, 2, 1, 0),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::DepthwiseConv2d(2, 0, 2, 1, 0),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::DepthwiseConv2d(2, 2, 0, 1, 0),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::DepthwiseConv2d(2, 2, 2, 0, 0),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::DepthwiseConv2d(2, 2, 2, 1, -1),
                    std::invalid_argument);

    // ag::avg_pool2d: non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        CHECK_THROWS_AS(ag::avg_pool2d(xv, 2, 2, 2),
                        std::invalid_argument);
    }
    // ag::avg_pool2d: non-positive kernel / stride
    {
        ag::Variable xv(make_tensor(std::vector<float>(16, 0.f),
                                    ag::Shape{1, 1, 4, 4}), true);
        CHECK_THROWS_AS(ag::avg_pool2d(xv, 0, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(ag::avg_pool2d(xv, 2, 0, 2), std::invalid_argument);
        CHECK_THROWS_AS(ag::avg_pool2d(xv, 2, 2, 0), std::invalid_argument);
    }
    // ag::avg_pool2d: kernel larger than input
    {
        ag::Variable xv(make_tensor(std::vector<float>(9, 0.f),
                                    ag::Shape{1, 1, 3, 3}), true);
        CHECK_THROWS_AS(ag::avg_pool2d(xv, 4, 4, 1), std::invalid_argument);
    }
    // nn::AvgPool2d constructor validation
    CHECK_THROWS_AS(ag::nn::AvgPool2d(0, 2, 2), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::AvgPool2d(2, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::AvgPool2d(2, 2, 0), std::invalid_argument);

    // ag::nearest_upsample2d: non-rank-4 input
    {
        ag::Variable xv(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{2, 2}),
                         true);
        CHECK_THROWS_AS(ag::nearest_upsample2d(xv, 2),
                        std::invalid_argument);
    }
    // ag::nearest_upsample2d: scale < 1
    {
        ag::Variable xv(make_tensor(std::vector<float>(16, 0.f),
                                    ag::Shape{1, 1, 4, 4}), true);
        CHECK_THROWS_AS(ag::nearest_upsample2d(xv, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(ag::nearest_upsample2d(xv, -2),
                        std::invalid_argument);
    }
    // nn::NearestUpsample2d constructor validation
    CHECK_THROWS_AS(ag::nn::NearestUpsample2d(0), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::NearestUpsample2d(-1), std::invalid_argument);

    report("spatial extras reject invalid arguments");
}

}  // namespace

int main() {
    test_conv2d_forward_matches_naive();
    test_conv2d_output_shape_metadata();
    test_conv2d_gradients();
    test_conv2d_module_named_parameters();
    test_conv2d_module_forward_backward();
    test_maxpool2d_forward_matches_naive();
    test_maxpool2d_gradient_overlap();
    test_maxpool2d_gradient_tie_determinism();
    test_maxpool2d_module_zero_parameters();
    test_conv2d_invalid_arguments();
    test_conv2d_rank1_bias_backward_shape();
    test_maxpool2d_invalid_arguments();

    test_depthwise_conv2d_forward_matches_naive();
    test_depthwise_conv2d_gradients();
    test_depthwise_conv2d_module_named_parameters();
    test_depthwise_conv2d_module_forward_backward();
    test_avgpool2d_forward_matches_naive();
    test_avgpool2d_gradient_overlap();
    test_avgpool2d_gradient_finite_difference();
    test_avgpool2d_module_zero_parameters();
    test_nearest_upsample2d_forward_matches_naive();
    test_nearest_upsample2d_gradient_multiplicity();
    test_nearest_upsample2d_gradient_finite_difference();
    test_nearest_upsample2d_module_zero_parameters();
    test_spatial_extra_invalid_arguments();

    std::printf("\nALL OOP CONV TESTS PASSED (%d)\n", passed);
    return 0;
}
