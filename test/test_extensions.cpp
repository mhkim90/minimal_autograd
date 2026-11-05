// test_extensions.cpp — grad-checks for all Phase 1-5 additions.
//
// grad_check expects the function to return a scalar (1x1). All non-scalar
// ops must be wrapped with ag::sum() before passing to grad_check.

#include "autograd/core/module.h"
#include "grad_check.h"
#include <cstdio>
#include <cmath>
#include <stdexcept>

using namespace ag;
using namespace ag_test;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++passed; }
    else     { std::printf("  FAIL  %s\n", name); ++failed; }
}

static Variable rand_var(int r, int c, float scale = 1.f) {
    return Variable(random_tensor(Shape{r, c}, scale), true);
}

static Variable pos_var(int r, int c) {
    return Variable(random_tensor(Shape{r, c}, 1.f, 0.1f), true);
}

static void test_sigmoid() {
    auto x = rand_var(3, 4);
    check("sigmoid", grad_check([](Variable v) { return sum(sigmoid(v)); }, x));
}

static void test_tanh_op() {
    auto x = rand_var(3, 4);
    check("tanh_op", grad_check([](Variable v) { return sum(tanh_op(v)); }, x));
}

static void test_exp_op() {
    auto x = rand_var(2, 3, 0.5f);
    check("exp_op", grad_check([](Variable v) { return sum(exp_op(v)); }, x));
}

static void test_log_op() {
    auto x = pos_var(2, 3);
    check("log_op", grad_check([](Variable v) { return sum(log_op(v)); }, x));
}

static void test_sqrt_op() {
    auto x = pos_var(3, 3);
    check("sqrt_op", grad_check([](Variable v) { return sum(sqrt_op(v)); }, x));
}

static void test_silu() {
    auto x = rand_var(3, 4);
    check("silu", grad_check([](Variable v) { return sum(silu(v)); }, x));
}

static void test_softplus() {
    auto x = rand_var(3, 4);
    check("softplus", grad_check([](Variable v) { return sum(softplus(v)); }, x));
}

static void test_sub() {
    auto a = rand_var(2, 3), b = rand_var(2, 3);
    check("sub", grad_check_multi(
        [](std::vector<Variable> vs) { return sum(sub(vs[0], vs[1])); }, {a, b}));
}

static void test_div_op() {
    auto a = rand_var(2, 3);
    auto b = Variable(random_tensor(Shape{2, 3}, 1.f, 0.5f), true);
    check("div_op", grad_check_multi(
        [](std::vector<Variable> vs) { return sum(div_op(vs[0], vs[1])); }, {a, b}));
}

static void test_cumsum() {
    auto x = rand_var(2, 5);
    check("cumsum axis=1", grad_check([](Variable v) { return sum(cumsum(v, 1)); }, x));
    check("cumsum axis=0", grad_check([](Variable v) { return sum(cumsum(v, 0)); }, x));
}

static void test_flip() {
    auto x = rand_var(3, 4);
    check("flip axis=1", grad_check([](Variable v) { return sum(flip(v, 1)); }, x));
    check("flip axis=0", grad_check([](Variable v) { return sum(flip(v, 0)); }, x));
}

static void test_avgpool2d() {
    int N=2, C=3, H=4, W=4, kH=2, kW=2, stride=2;
    auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
    check("avgpool2d", grad_check(
        [=](Variable v) { return sum(avg_pool2d(v, kH, kW, stride)); }, x));
}

static void test_nearest_upsample() {
    int N=2, C=2, H=3, W=3, scale=2;
    auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
    check("nearest_upsample2d", grad_check(
        [=](Variable v) { return sum(nearest_upsample2d(v, scale)); }, x));
}

static void test_depthwise_conv2d() {
    int N=2, C=3, H=5, W=5, kH=3, kW=3, stride=1, pad=1;
    auto x = Variable(random_tensor(Shape{N, C, H, W}), true);
    auto w = Variable(random_tensor(Shape{C, kH, kW}), true);
    auto b = Variable(Tensor::zeros(Shape{C}), true);
    check("depthwise_conv2d", grad_check_multi(
        [=](std::vector<Variable> vs) {
            return sum(depthwise_conv2d(
                vs[0], vs[1], vs[2], stride, pad));
        }, {x, w, b}));
}

static void test_groupnorm_forward() {
    int N=2, C=4, HW=9, G=2;
    auto x = Variable(random_tensor(Shape{N, C, 3, 3}), true);
    nn::GroupNorm gn(G, C);

    auto y = gn.forward(x);
    bool shape_ok = (y.value().shape() == Shape{N, C, 3, 3});

    float max_mean = 0.f, max_var_err = 0.f;
    int ch_per_g = C / G;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            int ch0 = g * ch_per_g;
            float m = 0.f;
            for (int c = ch0; c < ch0 + ch_per_g; ++c)
                for (int p = 0; p < HW; ++p) m += at(y.value(), n, c * HW + p);
            m /= static_cast<float>(ch_per_g * HW);
            float v = 0.f;
            for (int c = ch0; c < ch0 + ch_per_g; ++c)
                for (int p = 0; p < HW; ++p) {
                    float d = at(y.value(), n, c * HW + p) - m;
                    v += d * d;
                }
            v /= static_cast<float>(ch_per_g * HW);
            max_mean = std::max(max_mean, std::abs(m));
            max_var_err = std::max(max_var_err, std::abs(v - 1.f));
        }
    }
    bool stat_ok = (max_mean < 1e-4f) && (max_var_err < 1e-3f);
    check("groupnorm shape", shape_ok);
    check("groupnorm mean~0, var~1", stat_ok);
}

static void test_silu_module() {
    auto x = rand_var(3, 4);
    check("functional SiLU", grad_check([&](Variable v) { return sum(silu(v)); }, x));
}

static void test_sin_cos() {
    auto x = rand_var(3, 4);
    check("sin_op", grad_check([](Variable v) { return sum(sin_op(v)); }, x));
    check("cos_op", grad_check([](Variable v) { return sum(cos_op(v)); }, x));
}

static void test_clamp() {
    auto x = Variable(tensor(Shape{2, 4}, {-1.f, 0.3f, 0.7f, 1.5f,
         0.1f, 0.5f, 0.9f, 2.0f}), true);
    check("clamp", grad_check([](Variable v) { return sum(clamp(v, 0.f, 1.f)); }, x));
}

static void test_col_slice() {
    auto x = rand_var(3, 6);
    check("col_slice left",  grad_check([](Variable v) { return sum(col_slice(v, 0, 3)); }, x));
    check("col_slice right", grad_check([](Variable v) { return sum(col_slice(v, 3, 3)); }, x));

    bool threw = false;
    try {
        (void)col_slice(x, 5, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check("col_slice rejects out of range", threw);
}

static void test_row_slice() {
    auto x = rand_var(6, 3);
    check("row_slice top",    grad_check([](Variable v) { return sum(row_slice(v, 0, 3)); }, x));
    check("row_slice bottom", grad_check([](Variable v) { return sum(row_slice(v, 3, 3)); }, x));

    bool threw = false;
    try {
        (void)row_slice(x, 2, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check("row_slice rejects empty range", threw);
}

static void test_mean() {
    auto x = rand_var(3, 4);
    auto y = mean(x);
    check("mean value", std::fabs(at(y.value(), 0, 0) - host_mean(x.value())) < 1e-6f);
    check("mean grad", grad_check([](Variable v) { return mean(v); }, x));
}

static Variable finite_difference_regularizer(Variable v) {
    auto dx = sub(col_slice(v, 1, v.value().shape()[1] - 1),
                  col_slice(v, 0, v.value().shape()[1] - 1));
    auto dy = sub(row_slice(v, 1, v.value().shape()[0] - 1),
                  row_slice(v, 0, v.value().shape()[0] - 1));
    auto eps_x = Variable(filled(dx.value().shape(), 1e-3f), true);
    auto eps_y = Variable(filled(dy.value().shape(), 1e-3f), true);
    auto penalty_x = mean(sqrt_op(add(mul(dx, dx), eps_x)));
    auto penalty_y = mean(sqrt_op(add(mul(dy, dy), eps_y)));
    return add(penalty_x, penalty_y);
}

static void test_finite_difference_regularizer_from_slices() {
    auto x = rand_var(4, 5);
    check("finite difference regularizer from slices",
          grad_check([](Variable v) { return finite_difference_regularizer(v); }, x));
}

static void test_split() {
    auto x = rand_var(2, 4);
    check("split left",  grad_check([](Variable v) { return sum(split(v).first);  }, x));
    check("split right", grad_check([](Variable v) { return sum(split(v).second); }, x));
}

static void test_hcat() {
    auto a = rand_var(3, 4);
    auto b = rand_var(3, 2);
    auto c = rand_var(3, 5);
    check("hcat grad a", grad_check(
        [&](Variable v) { return sum(hcat({v, b, c})); }, a));
    check("hcat grad b", grad_check(
        [&](Variable v) { return sum(hcat({a, v, c})); }, b));
    check("hcat grad c", grad_check(
        [&](Variable v) { return sum(hcat({a, b, v})); }, c));
    auto out = hcat({a, b, c});
    check("hcat shape", out.value().shape()[0] == 3 && out.value().shape()[1] == 11);
}

int main() {
    std::printf("=== test_extensions ===\n");
    std::printf("-- Phase 1: activation + arithmetic ops --\n");
    test_sigmoid(); test_tanh_op(); test_exp_op();
    test_log_op();  test_sqrt_op(); test_silu();
    test_softplus(); test_sub();    test_div_op();
    test_cumsum();  test_flip();
    std::printf("-- Phase 2a: nn::AvgPool2d --\n");
    test_avgpool2d();
    std::printf("-- Phase 2b: nn::NearestUpsample2d --\n");
    test_nearest_upsample();
    std::printf("-- Phase 3: nn::DepthwiseConv2d --\n");
    test_depthwise_conv2d();
    std::printf("-- Phase 4: nn::GroupNorm --\n");
    test_groupnorm_forward();
    std::printf("-- Phase 5: functional SiLU --\n");
    test_silu_module();
    std::printf("-- Missing GUDM ops --\n");
    test_sin_cos();
    test_clamp();
    test_col_slice();
    test_row_slice();
    test_mean();
    test_finite_difference_regularizer_from_slices();
    test_split();
    std::printf("-- hcat --\n");
    test_hcat();
    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return (failed == 0) ? 0 : 1;
}
