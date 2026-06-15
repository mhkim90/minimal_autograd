// test_extensions.cpp — grad-checks for all Phase 1-5 additions.
//
// grad_check expects the function to return a scalar (1x1). All non-scalar
// ops must be wrapped with ag::sum() before passing to grad_check.

#include "autograd.h"
#include "grad_check.h"
#include <cstdio>
#include <cmath>

using namespace ag;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++passed; }
    else     { std::printf("  FAIL  %s\n", name); ++failed; }
}

static VarPtr rand_var(int r, int c, float scale = 1.f) {
    return Var::make(Mat::Random(r, c) * scale);
}

static VarPtr pos_var(int r, int c) {
    return Var::make((Mat::Random(r, c).array().abs() + 0.1f).matrix());
}

static void test_sigmoid() {
    auto x = rand_var(3, 4);
    check("sigmoid", grad_check([](VarPtr v) { return sum(sigmoid(v)); }, x));
}

static void test_tanh_op() {
    auto x = rand_var(3, 4);
    check("tanh_op", grad_check([](VarPtr v) { return sum(tanh_op(v)); }, x));
}

static void test_exp_op() {
    auto x = rand_var(2, 3, 0.5f);
    check("exp_op", grad_check([](VarPtr v) { return sum(exp_op(v)); }, x));
}

static void test_log_op() {
    auto x = pos_var(2, 3);
    check("log_op", grad_check([](VarPtr v) { return sum(log_op(v)); }, x));
}

static void test_sqrt_op() {
    auto x = pos_var(3, 3);
    check("sqrt_op", grad_check([](VarPtr v) { return sum(sqrt_op(v)); }, x));
}

static void test_silu() {
    auto x = rand_var(3, 4);
    check("silu", grad_check([](VarPtr v) { return sum(silu(v)); }, x));
}

static void test_softplus() {
    auto x = rand_var(3, 4);
    check("softplus", grad_check([](VarPtr v) { return sum(softplus(v)); }, x));
}

static void test_sub() {
    auto a = rand_var(2, 3), b = rand_var(2, 3);
    check("sub", grad_check_multi(
        [](std::vector<VarPtr> vs) { return sum(sub(vs[0], vs[1])); }, {a, b}));
}

static void test_div_op() {
    auto a = rand_var(2, 3);
    auto b = Var::make((Mat::Random(2, 3).array().abs() + 0.5f).matrix());
    check("div_op", grad_check_multi(
        [](std::vector<VarPtr> vs) { return sum(div_op(vs[0], vs[1])); }, {a, b}));
}

static void test_cumsum() {
    auto x = rand_var(2, 5);
    check("cumsum axis=1", grad_check([](VarPtr v) { return sum(cumsum(v, 1)); }, x));
    check("cumsum axis=0", grad_check([](VarPtr v) { return sum(cumsum(v, 0)); }, x));
}

static void test_flip() {
    auto x = rand_var(3, 4);
    check("flip axis=1", grad_check([](VarPtr v) { return sum(flip(v, 1)); }, x));
    check("flip axis=0", grad_check([](VarPtr v) { return sum(flip(v, 0)); }, x));
}

static void test_avgpool2d() {
    int N=2, C=3, H=4, W=4, kH=2, kW=2, stride=2;
    auto x = rand_var(N, C * H * W);
    check("avgpool2d", grad_check(
        [=](VarPtr v) { return sum(avgpool2d_op(v, N, C, H, W, kH, kW, stride)); }, x));
}

static void test_nearest_upsample() {
    int N=2, C=2, H=3, W=3, scale=2;
    auto x = rand_var(N, C * H * W);
    check("nearest_upsample2d", grad_check(
        [=](VarPtr v) { return sum(nearest_upsample2d_op(v, N, C, H, W, scale)); }, x));
}

static void test_depthwise_conv2d() {
    int N=2, C=3, H=5, W=5, kH=3, kW=3, stride=1, pad=1;
    auto x = rand_var(N, C * H * W);
    auto w = rand_var(C, kH * kW);
    auto b = Var::make(Mat::Zero(1, C));
    check("depthwise_conv2d", grad_check_multi(
        [=](std::vector<VarPtr> vs) {
            return sum(depthwise_conv2d_op(
                vs[0], vs[1], vs[2], N, C, H, W, kH, kW, stride, pad));
        }, {x, w, b}));
}

static void test_groupnorm_forward() {
    int N=2, C=4, HW=9, G=2;
    auto x = rand_var(N, C * HW);
    GroupNorm gn(G, C);
    gn.gamma->data.setOnes();
    gn.beta->data.setZero();

    auto y = gn.forward(x, C, HW);
    bool shape_ok = (y->data.rows() == N && y->data.cols() == C * HW);

    float max_mean = 0.f, max_var_err = 0.f;
    int ch_per_g = C / G;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            int ch0 = g * ch_per_g;
            float m = 0.f;
            for (int c = ch0; c < ch0 + ch_per_g; ++c)
                for (int p = 0; p < HW; ++p) m += y->data(n, c * HW + p);
            m /= static_cast<float>(ch_per_g * HW);
            float v = 0.f;
            for (int c = ch0; c < ch0 + ch_per_g; ++c)
                for (int p = 0; p < HW; ++p) {
                    float d = y->data(n, c * HW + p) - m;
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
    SiLUModule m;
    auto x = rand_var(3, 4);
    check("SiLUModule", grad_check([&](VarPtr v) { return sum(m.forward(v)); }, x));
}

static void test_sin_cos() {
    auto x = rand_var(3, 4);
    check("sin_op", grad_check([](VarPtr v) { return sum(sin_op(v)); }, x));
    check("cos_op", grad_check([](VarPtr v) { return sum(cos_op(v)); }, x));
}

static void test_clamp() {
    auto x = Var::make((Mat(2, 4) <<
        -1.f, 0.3f, 0.7f, 1.5f,
         0.1f, 0.5f, 0.9f, 2.0f).finished());
    check("clamp", grad_check([](VarPtr v) { return sum(clamp(v, 0.f, 1.f)); }, x));
}

static void test_col_slice() {
    auto x = rand_var(3, 6);
    check("col_slice left",  grad_check([](VarPtr v) { return sum(col_slice(v, 0, 3)); }, x));
    check("col_slice right", grad_check([](VarPtr v) { return sum(col_slice(v, 3, 3)); }, x));
}

static void test_split() {
    auto x = rand_var(2, 4);
    check("split left",  grad_check([](VarPtr v) { return sum(split(v).first);  }, x));
    check("split right", grad_check([](VarPtr v) { return sum(split(v).second); }, x));
}

int main() {
    std::printf("=== test_extensions ===\n");
    std::printf("-- Phase 1: activation + arithmetic ops --\n");
    test_sigmoid(); test_tanh_op(); test_exp_op();
    test_log_op();  test_sqrt_op(); test_silu();
    test_softplus(); test_sub();    test_div_op();
    test_cumsum();  test_flip();
    std::printf("-- Phase 2a: AvgPool2d --\n");
    test_avgpool2d();
    std::printf("-- Phase 2b: NearestUpsample2d --\n");
    test_nearest_upsample();
    std::printf("-- Phase 3: DepthwiseConv2d --\n");
    test_depthwise_conv2d();
    std::printf("-- Phase 4: GroupNorm --\n");
    test_groupnorm_forward();
    std::printf("-- Phase 5: SiLUModule --\n");
    test_silu_module();
    std::printf("-- Missing GUDM ops --\n");
    test_sin_cos();
    test_clamp();
    test_col_slice();
    test_split();
    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return (failed == 0) ? 0 : 1;
}
