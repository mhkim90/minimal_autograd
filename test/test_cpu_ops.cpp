// test_cpu_ops.cpp — Tensor-based CPU operations.
//
// Eigen- and CUDA-free. Exercises the Tensor-based ag::Variable API for the
// full CPU vertical slice: matmul, activations, reductions, elementwise
// arithmetic, reshape/transpose, slice/concat, simple losses. Forward
// values use tight numerical checks; gradients are cross-checked against
// central finite differences for rank-2 inputs.

#include "autograd/core/variable.h"
#include "autograd/core/ops.h"
#include "autograd/core/loss.h"
#include "autograd/extension/custom_op.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Tensor-based autograd headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Tensor-based autograd headers must not include CUDA runtime headers"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>

using ag::Shape;
using ag::Tensor;
using ag::Variable;

namespace {

int passed = 0;

#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #__VA_ARGS__, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_THROWS(...) do { \
    bool threw = false; \
    try { (void)(__VA_ARGS__); } catch (...) { threw = true; } \
    CHECK(threw); \
} while (0)

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

Tensor make(const std::vector<float>& values, const Shape& shape) {
    return Tensor::from_host(values.empty() ? nullptr : values.data(), shape);
}

std::vector<float> to_vec(const Tensor& t) {
    std::vector<float> out(t.elements());
    t.copy_to_host(out.empty() ? nullptr : out.data(), out.size());
    return out;
}

void check_near(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tol = 1e-5f) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tol) {
            std::fprintf(stderr,
                         "value mismatch at %zu: actual=%g expected=%g\n",
                         i, actual[i], expected[i]);
            CHECK(false);
        }
    }
}

// Central finite-difference gradient for a rank-2 Variable -> Variable
// function. Always aggregates the function output through ag::sum so a
// scalar output is produced. The variable x is mutated in place; the
// returned analytic-gradient comparison lives in the caller.
std::vector<float> fd_grad_rank2(Variable x,
                                 std::function<Variable(const Variable&)> f,
                                 float eps = 1e-3f) {
    const int64_t R = x.value().shape()[0];
    const int64_t C = x.value().shape()[1];
    std::vector<float> data = to_vec(x.value());
    Tensor x_storage = x.value();  // alias — shares storage with x
    std::vector<float> g(R * C);

    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            const int flat = static_cast<int>(r + R * c);
            const float orig = data[flat];

            data[flat] = orig + eps;
            x_storage.copy_from_host(data.data(), data.size());
            const float v_plus =
                to_vec(ag::sum(f(x)).value())[0];

            data[flat] = orig - eps;
            x_storage.copy_from_host(data.data(), data.size());
            const float v_minus =
                to_vec(ag::sum(f(x)).value())[0];

            data[flat] = orig;
            x_storage.copy_from_host(data.data(), data.size());

            g[flat] = (v_plus - v_minus) / (2.f * eps);
        }
    }
    return g;
}

void compare_grads(const std::vector<float>& analytic,
                   const std::vector<float>& fd,
                   float tol = 5e-2f) {
    CHECK(analytic.size() == fd.size());
    float max_diff = 0.f;
    int bad_idx = -1;
    for (std::size_t i = 0; i < analytic.size(); ++i) {
        const float d = std::fabs(analytic[i] - fd[i]);
        if (d > max_diff) {
            max_diff = d;
            bad_idx = static_cast<int>(i);
        }
    }
    if (max_diff > tol && bad_idx >= 0) {
        std::fprintf(stderr,
                     "  grad diff at index %d: analytic=%g fd=%g\n",
                     bad_idx, analytic[bad_idx], fd[bad_idx]);
        for (std::size_t i = 0; i < analytic.size(); ++i) {
            std::fprintf(stderr, "    [%zu] ana=%g fd=%g\n",
                         i, analytic[i], fd[i]);
        }
    }
    CHECK(max_diff <= tol);
}

// ── matmul ──────────────────────────────────────────────────────────────

void test_matmul() {
    Variable a(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3}), true);
    Variable b(make({7.f, 8.f, 9.f, 10.f, 11.f, 12.f}, Shape{3, 2}), true);
    Variable out = ag::matmul(a, b);
    CHECK(out.value().shape() == Shape{2, 2});
    // A column-major: [[1,3,5],[2,4,6]]; B: [[7,10],[8,11],[9,12]].
    // C = A @ B: row 0 = [76, 103], row 1 = [100, 136]; flat = [76, 100, 103, 136].
    check_near(to_vec(out.value()), {76.f, 100.f, 103.f, 136.f});

    Variable wrong(make({1.f, 2.f, 3.f}, Shape{1, 3}));
    CHECK_THROWS(ag::matmul(a, wrong));

    ag::sum(out).backward();
    // da[i, k] = sum_c B[k, c] = col-sum of B at index k
    check_near(to_vec(a.grad()), {17.f, 17.f, 19.f, 19.f, 21.f, 21.f});
    // db[k, n] = sum_m A[m, k] (dC is ones); flat = col sums of A stacked
    //   col 0 = [3, 7, 11], col 1 = [3, 7, 11]
    check_near(to_vec(b.grad()), {3.f, 7.f, 11.f, 3.f, 7.f, 11.f});

    Variable x(make({0.1f, 0.2f, 0.3f, -0.4f}, Shape{2, 2}), true);
    Variable y(make({0.5f, -0.6f, 0.7f, 0.8f}, Shape{2, 2}), true);
    auto fd = fd_grad_rank2(x, [&](const Variable& v) {
        return ag::matmul(v, y);
    });
    x.zero_grad();
    ag::sum(ag::matmul(x, y)).backward();
    compare_grads(to_vec(x.grad()), fd);

    auto fd2 = fd_grad_rank2(y, [&](const Variable& v) {
        return ag::matmul(x, v);
    });
    y.zero_grad();
    ag::sum(ag::matmul(x, y)).backward();
    compare_grads(to_vec(y.grad()), fd2);

    report("matmul forward, validation, and gradient");
}

// ── relu, sigmoid, tanh_op ───────────────────────────────────────────────

void test_relu() {
    Variable x(make({-2.f, -1.f, 0.f, 1.f, 2.f}, Shape{1, 5}), true);
    Variable y = ag::relu(x);
    CHECK(y.value().shape() == Shape{1, 5});
    check_near(to_vec(y.value()), {0.f, 0.f, 0.f, 1.f, 2.f});
    ag::sum(y).backward();
    check_near(to_vec(x.grad()), {0.f, 0.f, 0.f, 1.f, 1.f});

    Variable x2(make({-0.4f, 0.5f, -2.f, 1.2f}, Shape{2, 2}), true);
    auto fd = fd_grad_rank2(x2, [](const Variable& v) { return ag::relu(v); });
    x2.zero_grad();
    ag::sum(ag::relu(x2)).backward();
    compare_grads(to_vec(x2.grad()), fd);
    report("relu");
}

void test_sigmoid() {
    Variable x(make({0.f, 1.f, -1.f, 2.f}, Shape{2, 2}), true);
    auto y = ag::sigmoid(x);
    std::vector<float> expected = {
        1.f / (1.f + std::exp(-0.f)),
        1.f / (1.f + std::exp(-1.f)),
        1.f / (1.f + std::exp(1.f)),
        1.f / (1.f + std::exp(-2.f)),
    };
    check_near(to_vec(y.value()), expected);

    auto fd = fd_grad_rank2(x, [](const Variable& v) {
        return ag::sigmoid(v);
    });
    x.zero_grad();
    ag::sum(ag::sigmoid(x)).backward();
    compare_grads(to_vec(x.grad()), fd);
    report("sigmoid");
}

void test_tanh_op() {
    Variable x(make({0.f, 1.f, -1.f, 2.f}, Shape{2, 2}), true);
    auto y = ag::tanh_op(x);
    std::vector<float> expected = {
        std::tanh(0.f), std::tanh(1.f), std::tanh(-1.f), std::tanh(2.f)
    };
    check_near(to_vec(y.value()), expected);

    auto fd = fd_grad_rank2(x, [](const Variable& v) {
        return ag::tanh_op(v);
    });
    x.zero_grad();
    ag::sum(ag::tanh_op(x)).backward();
    compare_grads(to_vec(x.grad()), fd);
    report("tanh_op");
}

// ── exp, log, sqrt, sin, cos ────────────────────────────────────────────

void test_exp_log_sqrt_sin_cos() {
    Variable x(make({0.5f, 1.f, 1.5f, 2.f}, Shape{2, 2}), true);

    // exp
    Variable xe = ag::exp_op(x);
    check_near(to_vec(xe.value()),
               {std::exp(0.5f), std::exp(1.f), std::exp(1.5f), std::exp(2.f)});
    auto fd = fd_grad_rank2(x, [](const Variable& v) { return ag::exp_op(v); });
    x.zero_grad();
    ag::sum(ag::exp_op(x)).backward();
    compare_grads(to_vec(x.grad()), fd);

    // log — domain positive
    Variable xl(make({0.5f, 1.f, 1.5f, 2.f}, Shape{2, 2}), true);
    Variable xlog = ag::log_op(xl);
    check_near(to_vec(xlog.value()),
               {std::log(0.5f), std::log(1.f),
                std::log(1.5f), std::log(2.f)});
    fd = fd_grad_rank2(xl, [](const Variable& v) { return ag::log_op(v); });
    xl.zero_grad();
    ag::sum(ag::log_op(xl)).backward();
    compare_grads(to_vec(xl.grad()), fd);

    // sqrt
    Variable xs = ag::sqrt_op(xl);
    check_near(to_vec(xs.value()),
               {std::sqrt(0.5f), std::sqrt(1.f),
                std::sqrt(1.5f), std::sqrt(2.f)});
    fd = fd_grad_rank2(xl, [](const Variable& v) { return ag::sqrt_op(v); });
    xl.zero_grad();
    ag::sum(ag::sqrt_op(xl)).backward();
    compare_grads(to_vec(xl.grad()), fd);

    // sin / cos
    Variable xsc(make({0.f, 0.5f, 1.f, 1.5f}, Shape{2, 2}), true);
    Variable ysin = ag::sin_op(xsc);
    Variable ycos = ag::cos_op(xsc);
    check_near(to_vec(ysin.value()),
               {std::sin(0.f), std::sin(0.5f),
                std::sin(1.f), std::sin(1.5f)});
    check_near(to_vec(ycos.value()),
               {std::cos(0.f), std::cos(0.5f),
                std::cos(1.f), std::cos(1.5f)});
    fd = fd_grad_rank2(xsc, [](const Variable& v) { return ag::sin_op(v); });
    xsc.zero_grad();
    ag::sum(ag::sin_op(xsc)).backward();
    compare_grads(to_vec(xsc.grad()), fd);
    fd = fd_grad_rank2(xsc, [](const Variable& v) { return ag::cos_op(v); });
    xsc.zero_grad();
    ag::sum(ag::cos_op(xsc)).backward();
    compare_grads(to_vec(xsc.grad()), fd);

    report("exp/log/sqrt/sin/cos forward and gradients");
}

// ── silu, softplus, clamp ───────────────────────────────────────────────

void test_silu_softplus_clamp() {
    Variable x(make({-1.f, 0.f, 1.f, 2.f}, Shape{2, 2}), true);

    // silu
    Variable ys = ag::silu(x);
    std::vector<float> es = {
        -1.f / (1.f + std::exp(1.f)),
        0.f,
        1.f / (1.f + std::exp(-1.f)),
        2.f / (1.f + std::exp(-2.f)),
    };
    check_near(to_vec(ys.value()), es);
    auto fd = fd_grad_rank2(x, [](const Variable& v) { return ag::silu(v); });
    x.zero_grad();
    ag::sum(ag::silu(x)).backward();
    compare_grads(to_vec(x.grad()), fd);

    // softplus
    Variable ysp = ag::softplus(x);
    std::vector<float> esp = {
        std::log(1.f + std::exp(-1.f)),
        std::log(2.f),
        std::log(1.f + std::exp(1.f)),
        std::log(1.f + std::exp(2.f)),
    };
    check_near(to_vec(ysp.value()), esp, 1e-4f);
    fd = fd_grad_rank2(x, [](const Variable& v) { return ag::softplus(v); });
    x.zero_grad();
    ag::sum(ag::softplus(x)).backward();
    compare_grads(to_vec(x.grad()), fd);

    // clamp
    Variable yc = ag::clamp(x, -0.5f, 1.5f);
    check_near(to_vec(yc.value()), {-0.5f, 0.f, 1.f, 1.5f});
    fd = fd_grad_rank2(x,
        [](const Variable& v) { return ag::clamp(v, -0.5f, 1.5f); });
    x.zero_grad();
    ag::sum(ag::clamp(x, -0.5f, 1.5f)).backward();
    compare_grads(to_vec(x.grad()), fd);

    report("silu/softplus/clamp forward and gradients");
}

// ── softmax, log_softmax ────────────────────────────────────────────────

void test_softmax_logsoftmax() {
    // Column-major storage for rows [1, 2, 3] and [-1, 0, 1].
    Variable x(make({1.f, -1.f, 2.f, 0.f, 3.f, 1.f}, Shape{2, 3}), true);

    Variable s = ag::softmax(x);
    // row 0: e^1, e^2, e^3 normalized
    {
        const float a = std::exp(1.f), b = std::exp(2.f), c = std::exp(3.f);
        const float z = a + b + c;
        const float z1 = std::exp(-1.f) + 1.f + std::exp(1.f);
        check_near(to_vec(s.value()),
                   {a / z, std::exp(-1.f) / z1,
                    b / z, 1.f / z1,
                    c / z, std::exp(1.f) / z1});
    }

    Variable weights(make({0.5f, -1.f, 2.f, 0.25f, -0.75f, 1.5f},
                          Shape{2, 3}));
    auto fd = fd_grad_rank2(x, [&](const Variable& v) {
        return ag::mul(ag::softmax(v), weights);
    });
    x.zero_grad();
    ag::sum(ag::mul(ag::softmax(x), weights)).backward();
    compare_grads(to_vec(x.grad()), fd);

    Variable lsm = ag::log_softmax(x);
    {
        const float m0 = 3.f;  // max(1,2,3)
        const float lse0 = std::log(std::exp(1.f - m0)
                                  + std::exp(2.f - m0)
                                  + std::exp(3.f - m0)) + m0;
        const float m1 = 1.f;  // max(-1, 0, 1)
        const float lse1 = std::log(std::exp(-1.f - m1)
                                  + std::exp(0.f - m1)
                                  + std::exp(1.f - m1)) + m1;
        check_near(to_vec(lsm.value()),
                   {1.f - lse0, -1.f - lse1,
                    2.f - lse0, 0.f - lse1,
                    3.f - lse0, 1.f - lse1}, 1e-4f);
    }
    fd = fd_grad_rank2(x,
        [](const Variable& v) { return ag::sum(ag::log_softmax(v)); });
    x.zero_grad();
    ag::sum(ag::log_softmax(x)).backward();
    compare_grads(to_vec(x.grad()), fd);

    report("softmax/log_softmax forward and gradients");
}

// ── transpose, reshape, mean ────────────────────────────────────────────

void test_transpose_reshape_mean() {
    Variable x(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3}), true);

    Variable t = ag::transpose(x);
    CHECK(t.value().shape() == Shape{3, 2});
    // transpose of [[1,3,5],[2,4,6]] (column-major) = [[1,2],[3,4],[5,6]]
    // in column-major flat layout.
    // x flat (column-major): col0=[1,2], col1=[3,4], col2=[5,6] -> [1,2,3,4,5,6]
    // t[r,c] = x[c, r]
    // t flat (col-major of shape 3x2): col0=[t00,t10], col1=[t01,t11], col2=[t02,t12]
    // t[0,0]=x[0,0]=1; t[1,0]=x[0,1]=3; t[2,0]=x[0,2]=5
    // t[0,1]=x[1,0]=2; t[1,1]=x[1,1]=4; t[2,1]=x[1,2]=6
    // t flat = [1,3,5,2,4,6]
    check_near(to_vec(t.value()), {1.f, 3.f, 5.f, 2.f, 4.f, 6.f});

    ag::sum(t).backward();
    // transpose is a permutation, grad passes through unchanged.
    check_near(to_vec(x.grad()), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});

    // reshape
    Variable r = ag::reshape(x, 3, 2);
    CHECK(r.value().shape() == Shape{3, 2});
    // Same flat storage: 1,2,3,4,5,6
    check_near(to_vec(r.value()), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    x.zero_grad();
    ag::sum(r).backward();
    check_near(to_vec(x.grad()), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});

    // mean = sum / numel
    Variable y(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}));
    Variable m = ag::mean(y);
    CHECK(m.value().shape() == Shape{});
    check_near(to_vec(m.value()), {2.5f});

    Variable y2(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}), true);
    ag::sum(ag::mean(y2)).backward();
    // mean's gradient is 1/N = 1/4 per element
    check_near(to_vec(y2.grad()), {0.25f, 0.25f, 0.25f, 0.25f});

    report("transpose/reshape/mean forward and gradients");
}

// ── broadcast_add, sub, div_op ──────────────────────────────────────────

void test_broadcast_add_sub_div() {
    Variable a(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3}), true);
    Variable bias(make({10.f, 20.f, 30.f}, Shape{1, 3}), true);

    Variable y = ag::broadcast_add(a, bias);
    CHECK(y.value().shape() == Shape{2, 3});
    // a rows are [1, 3, 5] and [2, 4, 6].
    check_near(to_vec(y.value()), {11.f, 12.f, 23.f, 24.f, 35.f, 36.f});

    ag::sum(y).backward();
    // db = sum over rows of g = ones(2,3) -> [2, 2, 2] (1, 3)
    check_near(to_vec(a.grad()), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    check_near(to_vec(bias.grad()), {2.f, 2.f, 2.f});

    // sub
    Variable x(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}), true);
    Variable z(make({1.f, 1.f, 1.f, 1.f}, Shape{2, 2}));
    Variable diff = ag::sub(x, z);
    check_near(to_vec(diff.value()), {0.f, 1.f, 2.f, 3.f});
    ag::sum(diff).backward();
    check_near(to_vec(x.grad()), {1.f, 1.f, 1.f, 1.f});

    CHECK_THROWS(ag::sub(x,
        Variable(make({1.f, 2.f}, Shape{1, 2}), false)));

    // div
    Variable d(make({2.f, 4.f, 5.f, 8.f}, Shape{2, 2}), true);
    Variable denom(make({1.f, 2.f, 1.f, 4.f}, Shape{2, 2}));
    Variable q = ag::div_op(d, denom);
    check_near(to_vec(q.value()), {2.f, 2.f, 5.f, 2.f});
    ag::sum(q).backward();
    // grad_d = 1/denom = [1, 0.5, 1, 0.25]
    check_near(to_vec(d.grad()), {1.f, 0.5f, 1.f, 0.25f});

    auto fd = fd_grad_rank2(d, [&](const Variable& v) {
        return ag::div_op(v, denom);
    });
    d.zero_grad();
    ag::sum(ag::div_op(d, denom)).backward();
    compare_grads(to_vec(d.grad()), fd);

    CHECK_THROWS(ag::div_op(d, Variable(make({1.f, 2.f}, Shape{1, 2}), false)));

    report("broadcast_add/sub/div_op forward and gradients");
}

// ── concat, hcat ────────────────────────────────────────────────────────

void test_concat_hcat() {
    Variable a(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3}), true);
    Variable b(make({10.f, 20.f, 30.f, 40.f}, Shape{2, 2}));

    CHECK_THROWS(ag::concat({a, b}));

    Variable a2(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}), true);
    Variable b2(make({5.f, 6.f, 7.f, 8.f}, Shape{2, 2}), true);
    Variable c2 = ag::concat({a2, b2});
    CHECK(c2.value().shape() == Shape{4, 2});
    check_near(to_vec(c2.value()), {1.f, 2.f, 5.f, 6.f,
                                    3.f, 4.f, 7.f, 8.f});

    ag::sum(c2).backward();
    check_near(to_vec(a2.grad()), {1.f, 1.f, 1.f, 1.f});
    check_near(to_vec(b2.grad()), {1.f, 1.f, 1.f, 1.f});

    auto fd = fd_grad_rank2(a2, [&](const Variable& v) {
        return ag::concat({v, b2});
    });
    a2.zero_grad();
    ag::sum(ag::concat({a2, b2})).backward();
    compare_grads(to_vec(a2.grad()), fd);

    a2.zero_grad();
    b2.zero_grad();
    Variable h = ag::hcat({a2, b2});
    CHECK(h.value().shape() == Shape{2, 4});
    // hcat stacks columns: [a2 | b2]. col0=[1,2], col1=[3,4], col2=[5,6], col3=[7,8]
    // flat (column-major): [1,2,3,4,5,6,7,8]
    check_near(to_vec(h.value()), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});

    ag::sum(h).backward();
    check_near(to_vec(a2.grad()), {1.f, 1.f, 1.f, 1.f});
    check_near(to_vec(b2.grad()), {1.f, 1.f, 1.f, 1.f});

    // mismatched rows for hcat
    CHECK_THROWS(ag::hcat({a2,
        Variable(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{3, 2}), false)}));

    report("concat/hcat forward and gradients");
}

// ── cumsum, flip, col_slice, row_slice, split ──────────────────────────

void test_cumsum_flip_slices_split() {
    Variable x(make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, Shape{2, 3}), true);

    // cumsum along axis 1 (cols)
    Variable y1 = ag::cumsum(x, 1);
    // Input rows are [1, 3, 5] and [2, 4, 6].
    check_near(to_vec(y1.value()), {1.f, 2.f, 4.f, 6.f, 9.f, 12.f});

    ag::sum(y1).backward();
    // suffix sum of ones: [3, 2, 1] per row
    check_near(to_vec(x.grad()), {3.f, 3.f, 2.f, 2.f, 1.f, 1.f});

    auto fd = fd_grad_rank2(x, [](const Variable& v) {
        return ag::cumsum(v, 1);
    });
    x.zero_grad();
    ag::sum(ag::cumsum(x, 1)).backward();
    compare_grads(to_vec(x.grad()), fd);

    // cumsum along axis 0 (rows)
    Variable y0 = ag::cumsum(x, 0);
    // col 0: [1, 3]; col 1: [2, 7]; col 2: [3, 15]; wait
    // col 0 r0=1, r1=1+2=3 -> ov col0 = [1, 3]
    // col 1 r0=3, r1=3+4=7 -> ov col1 = [3, 7]
    // col 2 r0=5, r1=5+6=11 -> ov col2 = [5, 11]
    check_near(to_vec(y0.value()), {1.f, 3.f, 3.f, 7.f, 5.f, 11.f});

    // flip along axis 1 (reverse each row's columns)
    Variable f1 = ag::flip(x, 1);
    // Rows [1,3,5] and [2,4,6] reverse independently.
    check_near(to_vec(f1.value()), {5.f, 6.f, 3.f, 4.f, 1.f, 2.f});

    // flip along axis 0 (reverse each col's rows)
    Variable f0 = ag::flip(x, 0);
    // col 0 reversed: [2, 1]; col 1: [4, 3]; col 2: [6, 5]
    check_near(to_vec(f0.value()), {2.f, 1.f, 4.f, 3.f, 6.f, 5.f});

    fd = fd_grad_rank2(x, [](const Variable& v) { return ag::flip(v, 1); });
    x.zero_grad();
    ag::sum(ag::flip(x, 1)).backward();
    compare_grads(to_vec(x.grad()), fd);

    // col_slice and row_slice
    Variable col = ag::col_slice(x, 1, 2);
    CHECK(col.value().shape() == Shape{2, 2});
    // cols 1..2 of x: col 1=[3,4], col 2=[5,6] -> shape (2,2)
    // flat: [3, 4, 5, 6]
    check_near(to_vec(col.value()), {3.f, 4.f, 5.f, 6.f});
    x.zero_grad();
    ag::sum(col).backward();
    // g has shape (2, 2); we scatter back: zeros, with cols 1..2 = ones
    check_near(to_vec(x.grad()), {0.f, 0.f, 1.f, 1.f, 1.f, 1.f});

    CHECK_THROWS(ag::col_slice(x, 1, 10));

    Variable row = ag::row_slice(x, 1, 1);
    CHECK(row.value().shape() == Shape{1, 3});
    // row 1 of x: [2, 4, 6]
    // flat (1, 3) col-major: [2, 4, 6]
    check_near(to_vec(row.value()), {2.f, 4.f, 6.f});
    x.zero_grad();
    ag::sum(row).backward();
    // grad: row 0 = zeros, row 1 = ones
    check_near(to_vec(x.grad()), {0.f, 1.f, 0.f, 1.f, 0.f, 1.f});

    CHECK_THROWS(ag::row_slice(x, 5, 1));

    // split
    Tensor s = make({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, Shape{2, 4});
    Variable s_var(s, false);
    auto halves = ag::split(s_var);
    CHECK(halves.first.value().shape() == Shape{2, 2});
    CHECK(halves.second.value().shape() == Shape{2, 2});
    check_near(to_vec(halves.first.value()), {1.f, 2.f, 3.f, 4.f});
    check_near(to_vec(halves.second.value()), {5.f, 6.f, 7.f, 8.f});

    report("cumsum/flip/col_slice/row_slice/split");
}

// ── mse_loss, cross_entropy ─────────────────────────────────────────────

void test_losses() {
    // mse_loss
    Variable pred(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}), true);
    Tensor target = make({0.f, 0.f, 0.f, 0.f}, Shape{2, 2});
    Variable l = ag::mse_loss(pred, target);
    CHECK(l.value().shape() == Shape{});
    check_near(to_vec(l.value()),
               {((1.f + 4.f + 9.f + 16.f) / 4.f)});

    Variable pred2(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}), true);
    Tensor t2 = make({0.f, 0.f, 0.f, 0.f}, Shape{2, 2});
    ag::sum(ag::mse_loss(pred2, t2)).backward();
    // d/dx mean((x - 0)^2) = 2*x/N.
    check_near(to_vec(pred2.grad()), {0.5f, 1.f, 1.5f, 2.f});

    auto fd = fd_grad_rank2(pred2, [&](const Variable& v) {
        return ag::mse_loss(v, t2);
    });
    pred2.zero_grad();
    ag::sum(ag::mse_loss(pred2, t2)).backward();
    compare_grads(to_vec(pred2.grad()), fd);

    // cross_entropy with one-hot target
    Variable p(make({1.f, -1.f, 2.f, 0.f, 3.f, 1.f},
                    Shape{2, 3}), true);
    Tensor tg = make({0.f, 1.f, 1.f, 0.f, 0.f, 0.f}, Shape{2, 3});
    Variable ce = ag::cross_entropy(p, tg);
    CHECK(ce.value().shape() == Shape{});
    {
        // row 0: lsm = [1-lse0, 2-lse0, 3-lse0]; target row 0 = [0,1,0]
        //   = 2-lse0
        // row 1: lsm = [-1-lse1, 0-lse1, 1-lse1]; target row 1 = [1,0,0]
        //   = -1 - lse1
        // mean = ((2-lse0) + (-1 - lse1)) / 2; loss = -mean
        const float m0 = 3.f;
        const float lse0 = std::log(std::exp(1.f - m0)
                                   + std::exp(2.f - m0)
                                   + std::exp(3.f - m0)) + m0;
        const float m1 = 1.f;
        const float lse1 = std::log(std::exp(-1.f - m1)
                                   + std::exp(0.f - m1)
                                   + std::exp(1.f - m1)) + m1;
        const float row0 = 2.f - lse0;
        const float row1 = -1.f - lse1;
        check_near(to_vec(ce.value()),
                   {-((row0 + row1) / 2.f)}, 1e-4f);
    }

    auto fd2 = fd_grad_rank2(p, [&](const Variable& v) {
        return ag::cross_entropy(v, tg);
    });
    p.zero_grad();
    ag::sum(ag::cross_entropy(p, tg)).backward();
    compare_grads(to_vec(p.grad()), fd2);

    report("mse_loss / cross_entropy forward and gradients");
}

// ── input validation: rank-2 enforcement ───────────────────────────────

void test_rank2_validation() {
    Variable rank1(make({1.f, 2.f, 3.f}, Shape{3}), true);
    Variable rank2_a(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}));
    CHECK_THROWS(ag::matmul(rank1, rank2_a));
    CHECK_THROWS(ag::transpose(rank1));
    CHECK_THROWS(ag::reshape(rank1, 1, 3));
    CHECK_THROWS(ag::softmax(rank1));
    CHECK_THROWS(ag::log_softmax(rank1));
    CHECK_THROWS(ag::cumsum(rank1, 0));
    CHECK_THROWS(ag::flip(rank1, 1));
    Variable rank2_b(make({1.f, 2.f, 3.f, 4.f}, Shape{2, 2}));
    CHECK_THROWS(ag::concat({rank1, rank2_b}));
    CHECK_THROWS(ag::hcat({rank1, rank2_b}));
    CHECK_THROWS(ag::clamp(rank2_a, 2.f, 1.f));
    report("rank-2 enforcement");
}

}  // namespace

int main() {
    test_matmul();
    test_relu();
    test_sigmoid();
    test_tanh_op();
    test_exp_log_sqrt_sin_cos();
    test_silu_softplus_clamp();
    test_softmax_logsoftmax();
    test_transpose_reshape_mean();
    test_broadcast_add_sub_div();
    test_concat_hcat();
    test_cumsum_flip_slices_split();
    test_losses();
    test_rank2_validation();

    std::printf("\nALL CPU OPS TESTS PASSED (%d)\n", passed);
    return 0;
}
