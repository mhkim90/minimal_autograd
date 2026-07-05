#include "autograd.h"
#include "autograd/cuda_core.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ag;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (std::fabs(_a - _b) > (tol)) { \
        std::cerr << "FAIL: " << #a << " (" << _a << ") vs " << #b << " (" << _b \
                  << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

static void check_mat_near(const Mat& a, const Mat& b, float tol, const char* what) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        std::cerr << "FAIL: " << what << " shape mismatch\n";
        std::exit(1);
    }
    float diff = (a - b).cwiseAbs().maxCoeff();
    if (diff > tol) {
        std::cerr << "FAIL: " << what << " max diff " << diff << "\n";
        std::exit(1);
    }
}

static void check_unary_op(const char* name, const Mat& input,
                           VarPtr (*op)(VarPtr), float tol = 1e-4f) {
    auto cpu_x = Var::make(input);
    auto cpu_y = op(cpu_x);
    sum(cpu_y)->backward();

    auto cuda_x = Var::make(input)->cuda();
    auto cuda_y = op(cuda_x);
    CHECK(cuda_y->is_cuda());
    sum(cuda_y)->backward();

    check_mat_near(cuda_y->cpu()->data, cpu_y->data, tol, name);
    check_mat_near(cuda_x->cpu()->grad, cpu_x->grad, tol, name);
}

static void check_binary_op(const char* name, const Mat& lhs, const Mat& rhs,
                            VarPtr (*op)(VarPtr, VarPtr), float tol = 1e-4f) {
    auto cpu_a = Var::make(lhs);
    auto cpu_b = Var::make(rhs);
    auto cpu_y = op(cpu_a, cpu_b);
    sum(cpu_y)->backward();

    auto cuda_a = Var::make(lhs)->cuda();
    auto cuda_b = Var::make(rhs)->cuda();
    auto cuda_y = op(cuda_a, cuda_b);
    CHECK(cuda_y->is_cuda());
    sum(cuda_y)->backward();

    check_mat_near(cuda_y->cpu()->data, cpu_y->data, tol, name);
    check_mat_near(cuda_a->cpu()->grad, cpu_a->grad, tol, name);
    check_mat_near(cuda_b->cpu()->grad, cpu_b->grad, tol, name);
}

int main() {
    CudaRuntimeInfo cuda_info;
    try {
        cuda_info = cuda_runtime_info();
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA runtime probe failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << cuda_runtime_summary(cuda_info) << "\n";
    if (!cuda_info.has_device()) {
        std::cout << "SKIP CUDA CORE TESTS: " << cuda_info.status << "\n";
        return 0;
    }
    CHECK(cuda_info.device_count > 0);
    CHECK(cuda_info.current_device >= 0);
    CHECK(!cuda_info.device.name.empty());
    CHECK(cuda_info.device.compute_major > 0);

    Mat m(2, 3);
    m << 1.f, -2.f, 3.f,
         4.f, -5.f, 6.f;

    auto x = Var::make(m)->cuda();
    CHECK(x->is_cuda());

    auto y = sum(relu(add(x, x)));
    CHECK(y->is_cuda());
    y->backward();

    auto y_cpu = y->cpu();
    CHECK_NEAR(y_cpu->data(0, 0), 28.f, 1e-4f);

    auto x_cpu = x->cpu();
    CHECK_NEAR(x_cpu->grad(0, 0), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(0, 1), 0.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(0, 2), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 0), 2.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 1), 0.f, 1e-4f);
    CHECK_NEAR(x_cpu->grad(1, 2), 2.f, 1e-4f);

    {
        Mat shaped_data(2, 6);
        shaped_data << 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f, 10.f, 11.f, 12.f;
        auto shaped_cpu = Var::make4d(shaped_data, 2, 3, 1, 2);
        auto shaped_cuda = shaped_cpu->cuda();
        CHECK(shaped_cuda->is_cuda());
        CHECK(shaped_cuda->is4d());
        CHECK(shaped_cuda->dim(0) == 2 && shaped_cuda->dim(1) == 3);
        CHECK(shaped_cuda->dim(2) == 1 && shaped_cuda->dim(3) == 2);
        auto shaped_roundtrip = shaped_cuda->cpu();
        CHECK(!shaped_roundtrip->is_cuda());
        CHECK(shaped_roundtrip->is4d());
        CHECK(shaped_roundtrip->dim(0) == 2 && shaped_roundtrip->dim(1) == 3);
        CHECK(shaped_roundtrip->dim(2) == 1 && shaped_roundtrip->dim(3) == 2);
        check_mat_near(shaped_roundtrip->data, shaped_data, 1e-4f, "cuda/cpu shape roundtrip data");

        shaped_cuda->data = Mat::Constant(2, 6, 9.f);
        shaped_cuda->sync_data_to_cuda();
        CHECK(shaped_cuda->is4d());
        shaped_cuda->data.setZero();
        shaped_cuda->sync_data_from_cuda();
        CHECK(shaped_cuda->is4d());
        CHECK_NEAR(shaped_cuda->data(1, 5), 9.f, 1e-4f);

        shaped_cuda->grad = Mat::Constant(2, 6, 4.f);
        shaped_cuda->sync_grad_to_cuda();
        CHECK(shaped_cuda->is4d());
        shaped_cuda->grad.setZero();
        shaped_cuda->sync_grad_from_cuda();
        CHECK(shaped_cuda->is4d());
        CHECK_NEAR(shaped_cuda->grad(0, 3), 4.f, 1e-4f);

        shaped_cuda->clear_grad();
        CHECK_NEAR(shaped_cuda->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);
        CHECK_NEAR(shaped_cuda->cpu()->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);

        auto sync_loss = sum(scale(shaped_cuda, 3.f));
        sync_loss->backward();
        auto shaped_after_backward = shaped_cuda->cpu();
        CHECK(shaped_after_backward->is4d());
        CHECK_NEAR(shaped_after_backward->grad(1, 4), 3.f, 1e-4f);

        shaped_cuda->clear_grad();
        CHECK_NEAR(shaped_cuda->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);
        CHECK_NEAR(shaped_cuda->cpu()->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);
    }

    auto a = Var::make(Mat::Constant(2, 2, 3.f))->cuda();
    auto b = Var::make(Mat::Constant(2, 2, 4.f))->cuda();
    auto z = sum(scale(mul(a, b), 0.5f));
    z->backward();
    CHECK_NEAR(z->cpu()->data(0, 0), 24.f, 1e-4f);
    CHECK_NEAR(a->cpu()->grad(0, 0), 2.f, 1e-4f);
    CHECK_NEAR(b->cpu()->grad(0, 0), 1.5f, 1e-4f);

    bool threw = false;
    try {
        (void)transpose(a);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    bool shape_threw = false;
    try {
        (void)add(a, Var::make(Mat::Constant(1, 2, 1.f))->cuda());
    } catch (const std::runtime_error&) {
        shape_threw = true;
    }
    CHECK(shape_threw);

    Mat A(2, 3);
    A << 1.f, 2.f, 3.f,
         4.f, 5.f, 6.f;
    Mat B(3, 2);
    B << 1.f, 2.f,
         3.f, 4.f,
         5.f, 6.f;
    auto ma = Var::make(A)->cuda();
    auto mb = Var::make(B)->cuda();
    auto mm = sum(matmul(ma, mb));
    mm->backward();
    CHECK_NEAR(mm->cpu()->data(0, 0), (A * B).sum(), 1e-4f);
    Mat expected_ma_grad = Mat::Ones(2, 2) * B.transpose();
    Mat expected_mb_grad = A.transpose() * Mat::Ones(2, 2);
    CHECK_NEAR(ma->cpu()->grad(1, 2), expected_ma_grad(1, 2), 1e-4f);
    CHECK_NEAR(mb->cpu()->grad(2, 1), expected_mb_grad(2, 1), 1e-4f);

    auto bx = Var::make(Mat::Constant(4, 3, 1.f))->cuda();
    auto bb = Var::make(Mat::Constant(1, 3, 2.f))->cuda();
    auto by = sum(broadcast_add(bx, bb));
    by->backward();
    CHECK_NEAR(by->cpu()->data(0, 0), 36.f, 1e-4f);
    CHECK_NEAR(bx->cpu()->grad(3, 2), 1.f, 1e-4f);
    CHECK_NEAR(bb->cpu()->grad(0, 2), 4.f, 1e-4f);

    auto p = Var::make(Mat::Constant(2, 2, 1.f))->cuda();
    auto loss = sum(scale(p, 2.f));
    loss->backward();
    SGD opt({p}, 0.1f);
    opt.step();
    CHECK_NEAR(p->cpu()->data(0, 0), 0.8f, 1e-4f);
    opt.zero_grad();
    CHECK_NEAR(p->cpu()->grad(0, 0), 0.f, 1e-4f);
    auto loss2 = sum(scale(p, 2.f));
    loss2->backward();
    CHECK_NEAR(p->cpu()->grad(0, 0), 2.f, 1e-4f);

    Linear lin(3, 2);
    lin.W = lin.W->cuda();
    lin.b = lin.b->cuda();
    auto lx = Var::make(Mat::Constant(4, 3, 1.f))->cuda();
    auto lloss = sum(lin.forward(lx));
    lloss->backward();
    SGD lin_opt(lin.parameters(), 0.01f);
    Mat w_before = lin.W->cpu()->data;
    lin_opt.step();
    CHECK(lloss->is_cuda());
    CHECK(lin.W->is_cuda());
    CHECK(lin.W->cpu()->grad.cwiseAbs().maxCoeff() > 0.f);
    CHECK((lin.W->cpu()->data - w_before).cwiseAbs().maxCoeff() > 0.f);
    lin.zero_grad();
    CHECK_NEAR(lin.W->cpu()->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);

    Mat logits(2, 3);
    logits << 1.f, 2.f, 3.f,
              1.f, 0.f, -1.f;
    Mat target = Mat::Zero(2, 3);
    target(0, 2) = 1.f;
    target(1, 0) = 1.f;
    auto lg = Var::make(logits)->cuda();
    auto sm = softmax(lg)->cpu();
    CHECK_NEAR(sm->data.row(0).sum(), 1.f, 1e-4f);
    CHECK_NEAR(sm->data.row(1).sum(), 1.f, 1e-4f);

    auto ce = cross_entropy(lg, target);
    CHECK(ce->is_cuda());
    ce->backward();
    auto ce_cpu = ce->cpu();
    auto lg_cpu = lg->cpu();
    Mat expected_sm = softmax(Var::make(logits))->data;
    float expected_ce = -std::log(expected_sm(0, 2)) / 2.f
                        -std::log(expected_sm(1, 0)) / 2.f;
    CHECK_NEAR(ce_cpu->data(0, 0), expected_ce, 1e-4f);
    CHECK_NEAR(lg_cpu->grad(0, 2), (expected_sm(0, 2) - 1.f) / 2.f, 1e-4f);
    CHECK_NEAR(lg_cpu->grad(1, 0), (expected_sm(1, 0) - 1.f) / 2.f, 1e-4f);

    {
        Mat act(2, 5);
        act << -8.f, -2.f, -0.25f, 0.5f, 8.f,
                3.f, -4.f, 0.f, 1.5f, -1.f;
        Mat small = act * 0.25f;
        Mat pos(2, 5);
        pos << 0.1f, 0.5f, 1.f, 4.f, 16.f,
               2.f, 0.25f, 9.f, 0.75f, 6.f;
        check_unary_op("cuda sigmoid", act, sigmoid);
        check_unary_op("cuda tanh_op", act, tanh_op);
        check_unary_op("cuda exp_op", small, exp_op);
        check_unary_op("cuda log_op", pos, log_op);
        check_unary_op("cuda sqrt_op", pos, sqrt_op);
        check_unary_op("cuda silu", act, silu);
        check_unary_op("cuda softplus", act, softplus);

        Mat lhs(2, 5);
        lhs << 1.f, -2.f, 3.f, -4.f, 5.f,
               -1.5f, 2.5f, -3.5f, 4.5f, -5.5f;
        Mat rhs(2, 5);
        rhs << 0.5f, 2.f, -1.5f, 4.f, -2.f,
               3.f, -2.5f, 1.25f, -4.5f, 5.5f;
        Mat div_rhs(2, 5);
        div_rhs << 0.75f, 1.5f, 2.5f, 4.f, 8.f,
                   1.25f, 3.f, 5.f, 6.f, 9.f;
        check_binary_op("cuda sub", lhs, rhs, sub);
        check_binary_op("cuda div_op", lhs, div_rhs, div_op);

        auto shaped_a = Var::make4d(Mat::Constant(2, 12, 1.f), 2, 3, 2, 2)->cuda();
        auto shaped_b = Var::make4d(Mat::Constant(2, 12, 0.5f), 2, 3, 2, 2)->cuda();
        auto shaped_unary = sigmoid(shaped_a);
        auto shaped_binary = sub(shaped_a, shaped_b);
        CHECK(shaped_unary->is4d());
        CHECK(shaped_binary->is4d());
        CHECK(shaped_unary->dim(0) == 2 && shaped_unary->dim(1) == 3);
        CHECK(shaped_unary->dim(2) == 2 && shaped_unary->dim(3) == 2);
        CHECK(shaped_binary->dim(0) == 2 && shaped_binary->dim(1) == 3);
        CHECK(shaped_binary->dim(2) == 2 && shaped_binary->dim(3) == 2);

        bool mixed_threw = false;
        try {
            (void)sub(Var::make(lhs)->cuda(), Var::make(rhs));
        } catch (const std::runtime_error&) {
            mixed_threw = true;
        }
        CHECK(mixed_threw);

        mixed_threw = false;
        try {
            (void)div_op(Var::make(lhs), Var::make(div_rhs)->cuda());
        } catch (const std::runtime_error&) {
            mixed_threw = true;
        }
        CHECK(mixed_threw);
    }

    {
        Mat param0(2, 3);
        param0 << 1.f, -2.f, 3.f,
                  -4.f, 5.f, -6.f;
        Mat grad1(2, 3);
        grad1 << 0.1f, -0.2f, 0.3f,
                 -0.4f, 0.5f, -0.6f;
        Mat grad2(2, 3);
        grad2 << -0.3f, 0.25f, -0.2f,
                  0.15f, -0.1f, 0.05f;
        Mat grad3(2, 3);
        grad3 << 0.01f, 0.02f, -0.03f,
                 -0.04f, 0.05f, 0.06f;
        std::vector<Mat> grads = {grad1, grad2, grad3};

        auto cpu_p = Var::make(param0);
        auto cuda_p = Var::make(param0)->cuda();
        Adam cpu_adam({cpu_p}, 0.01f);
        Adam cuda_adam({cuda_p}, 0.01f);

        for (const auto& g : grads) {
            cpu_p->grad = g;
            cuda_p->grad = g;
            cuda_p->sync_grad_to_cuda();
            cpu_adam.step();
            cuda_adam.step();
            check_mat_near(cuda_p->cpu()->data, cpu_p->data, 1e-4f, "cuda Adam data");
        }
        cuda_adam.zero_grad();
        CHECK_NEAR(cuda_p->cpu()->grad.cwiseAbs().maxCoeff(), 0.f, 1e-4f);

        auto mixed_cpu = Var::make(param0);
        auto mixed_cuda = Var::make(param0)->cuda();
        auto ref_cpu = Var::make(param0);
        auto ref_cuda = Var::make(param0)->cuda();
        Adam mixed_adam({mixed_cpu, mixed_cuda}, 0.01f);
        Adam ref_cpu_adam({ref_cpu}, 0.01f);
        Adam ref_cuda_adam({ref_cuda}, 0.01f);
        mixed_cpu->grad = grad1;
        ref_cpu->grad = grad1;
        mixed_cuda->grad = grad2;
        mixed_cuda->sync_grad_to_cuda();
        ref_cuda->grad = grad2;
        ref_cuda->sync_grad_to_cuda();
        mixed_adam.step();
        ref_cpu_adam.step();
        ref_cuda_adam.step();
        check_mat_near(mixed_cpu->data, ref_cpu->data, 1e-4f, "mixed Adam CPU param");
        check_mat_near(mixed_cuda->cpu()->data, ref_cuda->cpu()->data, 1e-4f,
                       "mixed Adam CUDA param");
    }

    {
        const int N = 2, C = 2, H = 4, W = 5;
        const int out_ch = 3, kH = 3, kW = 2, stride = 1, pad = 1;
        Mat conv_x(N, C * H * W);
        Mat conv_w(out_ch, C * kH * kW);
        Mat conv_b(1, out_ch);
        for (int i = 0; i < conv_x.size(); ++i) {
            conv_x.data()[i] = 0.03f * static_cast<float>((i % 19) - 9);
        }
        for (int i = 0; i < conv_w.size(); ++i) {
            conv_w.data()[i] = 0.02f * static_cast<float>((i % 17) - 8);
        }
        for (int i = 0; i < conv_b.size(); ++i) {
            conv_b.data()[i] = 0.05f * static_cast<float>(i - 1);
        }

        Conv2d conv_cpu(C, out_ch, kH, kW, stride, pad);
        conv_cpu.W = Var::make(conv_w);
        conv_cpu.b = Var::make(conv_b);
        auto cx = Var::make4d(conv_x, N, C, H, W);
        auto cy = conv_cpu.forward(cx);
        sum(cy)->backward();

        Conv2d conv_cuda(C, out_ch, kH, kW, stride, pad);
        conv_cuda.W = Var::make(conv_w)->cuda();
        conv_cuda.b = Var::make(conv_b)->cuda();
        auto gx = Var::make4d(conv_x, N, C, H, W)->cuda();
        auto gy = conv_cuda.forward(gx);
        CHECK(gy->is_cuda());
        CHECK(gy->is4d());
        sum(gy)->backward();

        check_mat_near(gy->cpu()->data, cy->data, 1e-4f, "cuda Conv2d forward");
        check_mat_near(gx->cpu()->grad, cx->grad, 1e-4f, "cuda Conv2d grad input");
        check_mat_near(conv_cuda.W->cpu()->grad, conv_cpu.W->grad, 1e-4f,
                       "cuda Conv2d grad weight");
        check_mat_near(conv_cuda.b->cpu()->grad, conv_cpu.b->grad, 1e-4f,
                       "cuda Conv2d grad bias");
    }

    {
        const int N = 1, C = 2, H = 5, W = 7;
        const int out_ch = 2, kH = 3, kW = 3, stride = 2, pad = 1;
        Mat conv_x(N, C * H * W);
        Mat conv_w(out_ch, C * kH * kW);
        Mat conv_b(1, out_ch);
        for (int i = 0; i < conv_x.size(); ++i) {
            conv_x.data()[i] = 0.01f * static_cast<float>((i % 23) - 11);
        }
        for (int i = 0; i < conv_w.size(); ++i) {
            conv_w.data()[i] = 0.015f * static_cast<float>((i % 13) - 6);
        }
        conv_b << -0.03f, 0.04f;

        Conv2d conv_cpu(C, out_ch, kH, kW, stride, pad);
        conv_cpu.W = Var::make(conv_w);
        conv_cpu.b = Var::make(conv_b);
        auto cx = Var::make4d(conv_x, N, C, H, W);
        auto cy = conv_cpu.forward(cx);
        CHECK(cy->dim(2) == 3 && cy->dim(3) == 4);
        sum(add(cy, cy))->backward();

        Conv2d conv_cuda(C, out_ch, kH, kW, stride, pad);
        conv_cuda.W = Var::make(conv_w)->cuda();
        conv_cuda.b = Var::make(conv_b)->cuda();
        auto gx = Var::make4d(conv_x, N, C, H, W)->cuda();
        auto gy = conv_cuda.forward(gx);
        CHECK(gy->is_cuda());
        CHECK(gy->is4d());
        CHECK(gy->dim(0) == N && gy->dim(1) == out_ch);
        CHECK(gy->dim(2) == 3 && gy->dim(3) == 4);
        sum(add(gy, gy))->backward();

        check_mat_near(gy->cpu()->data, cy->data, 1e-4f,
                       "cuda Conv2d stride/pad forward");
        check_mat_near(gx->cpu()->grad, cx->grad, 1e-4f,
                       "cuda Conv2d stride/pad repeated grad input");
        check_mat_near(conv_cuda.W->cpu()->grad, conv_cpu.W->grad, 1e-4f,
                       "cuda Conv2d stride/pad repeated grad weight");
        check_mat_near(conv_cuda.b->cpu()->grad, conv_cpu.b->grad, 1e-4f,
                       "cuda Conv2d stride/pad repeated grad bias");
    }

    {
        const int N = 2, C = 2, H = 4, W = 4;
        Mat pool_x(N, C * H * W);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        pool_x(n, c * H * W + h * W + w) =
                            static_cast<float>(c * H * W + h * W + w) +
                            0.01f * static_cast<float>(n);
                    }
                }
            }
        }

        MaxPool2d pool(2, 2, 1);
        auto px = Var::make4d(pool_x, N, C, H, W);
        auto py = pool.forward(px);
        sum(py)->backward();

        auto pgx = Var::make4d(pool_x, N, C, H, W)->cuda();
        auto pgy = pool.forward(pgx);
        CHECK(pgy->is_cuda());
        CHECK(pgy->is4d());
        sum(pgy)->backward();

        check_mat_near(pgy->cpu()->data, py->data, 1e-4f, "cuda MaxPool2d forward");
        check_mat_near(pgx->cpu()->grad, px->grad, 1e-4f, "cuda MaxPool2d grad input");
    }

    {
        const int N = 2, C = 2, H = 6, W = 7;
        Mat pool_x(N, C * H * W);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        pool_x(n, c * H * W + h * W + w) =
                            static_cast<float>(c * H * W + h * W + w) +
                            0.01f * static_cast<float>(n);
                    }
                }
            }
        }

        MaxPool2d pool(2, 3, 2);
        auto px = Var::make4d(pool_x, N, C, H, W);
        auto py = pool.forward(px);
        CHECK(py->dim(2) == 3 && py->dim(3) == 3);
        sum(add(py, py))->backward();

        auto pgx = Var::make4d(pool_x, N, C, H, W)->cuda();
        auto pgy = pool.forward(pgx);
        CHECK(pgy->is_cuda());
        CHECK(pgy->is4d());
        CHECK(pgy->dim(0) == N && pgy->dim(1) == C);
        CHECK(pgy->dim(2) == 3 && pgy->dim(3) == 3);
        sum(add(pgy, pgy))->backward();

        check_mat_near(pgy->cpu()->data, py->data, 1e-4f,
                       "cuda MaxPool2d stride forward");
        check_mat_near(pgx->cpu()->grad, px->grad, 1e-4f,
                       "cuda MaxPool2d repeated grad input");
    }

    {
        auto unsupported_x = Var::make4d(Mat::Constant(1, 2 * 4 * 4, 1.f),
                                         1, 2, 4, 4)->cuda();

        bool unsupported_threw = false;
        try {
            AvgPool2d avg(2, 2);
            (void)avg.forward(unsupported_x);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);

        unsupported_threw = false;
        try {
            DepthwiseConv2d depthwise(2, 3, 3, 1, 1);
            depthwise.W = depthwise.W->cuda();
            depthwise.b = depthwise.b->cuda();
            (void)depthwise.forward(unsupported_x);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);

        unsupported_threw = false;
        try {
            NearestUpsample2d upsample(2);
            (void)upsample.forward(unsupported_x);
        } catch (const std::runtime_error&) {
            unsupported_threw = true;
        }
        CHECK(unsupported_threw);
    }

    std::cout << "ALL CUDA CORE TESTS PASSED\n";
    return 0;
}
