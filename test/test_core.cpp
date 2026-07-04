// test_core.cpp — autograd core: shared-node correctness, all built-in ops.

#include "autograd.h"
#include "grad_check.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <functional>

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

#define CHECK_EQ_MAT(a, b) do { \
    auto _A = (a); auto _B = (b); \
    if (_A.rows() != _B.rows() || _A.cols() != _B.cols() || \
        (_A - _B).cwiseAbs().maxCoeff() > 1e-4f) { \
        std::cerr << "FAIL: mat mismatch at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// --- Phase 2: shared-node correctness ---

void test_shared_x_plus_x() {
    auto x = Var::make(Mat::Constant(2, 2, 3.f));
    auto y = sum(add(x, x));     // y = sum(2x), scalar
    y->backward();
    // d/dx (2x) = 2, NOT 4 (no double-counting).
    CHECK_NEAR(x->grad(0, 0), 2.f, 1e-4f);
    CHECK_NEAR(x->grad(1, 1), 2.f, 1e-4f);
    std::cout << "[ok] x + x -> grad = 2 (no double-count)\n";
}

void test_shared_x_cubed() {
    auto x  = Var::make(Mat::Constant(1, 1, 2.5f));
    auto x2 = mul(x, x);
    auto y  = mul(x2, x);   // y = x^3
    y->backward();
    // d/dx (x^3) = 3x^2 = 3 * 6.25 = 18.75
    CHECK_NEAR(x->grad(0, 0), 18.75f, 1e-3f);
    std::cout << "[ok] x*x*x -> grad = 3x^2 (no double-count)\n";
}

void test_grad_check_ops() {
    // AddFn
    {
        auto x = Var::make(Mat::Random(3, 3));
        auto f = [](VarPtr x) { return sum(add(x, x)); };
        CHECK(grad_check(f, x));
    }
    // MulFn
    {
        auto x = Var::make(Mat::Random(2, 2));
        auto f = [](VarPtr x) { return sum(mul(x, x)); };
        CHECK(grad_check(f, x));
    }
    // MatMulFn
    {
        auto x  = Var::make(Mat::Random(3, 4));
        auto w  = Var::make(Mat::Random(4, 2));
        auto f  = [w](VarPtr x) { return sum(matmul(x, w)); };
        CHECK(grad_check(f, x));
    }
    // ReLUFn
    {
        auto x = Var::make(Mat::Random(2, 2));
        auto f = [](VarPtr x) { return sum(relu(x)); };
        CHECK(grad_check(f, x));
    }
    // SumFn
    {
        auto x = Var::make(Mat::Random(2, 2));
        auto f = [](VarPtr x) { return sum(sum(x)); };
        CHECK(grad_check(f, x));
    }
    std::cout << "[ok] grad_check passes for Add/Mul/MatMul/ReLU/Sum\n";
}

// --- Phase 2c: broadcast bias add ---

void test_broadcast_add() {
    auto a = Var::make(Mat::Constant(4, 3, 1.f));   // 4-row input
    auto b = Var::make(Mat::Constant(1, 3, 2.f));   // row bias
    auto y = sum(broadcast_add(a, b));              // reduce to scalar
    y->backward();

    // grad w.r.t. a is all ones
    CHECK_NEAR(a->grad(0, 0), 1.f, 1e-4f);
    CHECK_NEAR(a->grad(3, 2), 1.f, 1e-4f);
    // grad w.r.t. b sums over the batch axis → 4 in each entry
    CHECK_NEAR(b->grad(0, 0), 4.f, 1e-4f);
    CHECK_NEAR(b->grad(0, 2), 4.f, 1e-4f);
    std::cout << "[ok] BroadcastAddFn: bias grad sums over batch\n";
}

// --- Phase 3: shape ops + softmax ---

void test_softmax() {
    auto x = Var::make(Mat::Random(2, 3));
    auto y = softmax(x);
    CHECK_NEAR(y->data.row(0).sum(), 1.f, 1e-4f);
    CHECK_NEAR(y->data.row(1).sum(), 1.f, 1e-4f);

    auto f = [](VarPtr x) { return sum(softmax(x)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] softmax: forward sums to 1, grad_check ok\n";
}

void test_log_softmax() {
    auto x = Var::make(Mat::Random(2, 3));
    auto f = [](VarPtr x) { return sum(log_softmax(x)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] log_softmax: grad_check ok\n";
}

void test_transpose() {
    auto x = Var::make((Mat(2, 3) << 1, 2, 3, 4, 5, 6).finished());
    auto y = transpose(x);
    CHECK_EQ_MAT(y->data, Mat((Mat(3, 2) << 1, 4, 2, 5, 3, 6).finished()));
    auto f = [](VarPtr x) { return sum(transpose(x)); };
    CHECK(grad_check(f, x));

    auto square4d = Var::make4d(Mat::Random(2, 2), 2, 1, 1, 2);
    auto transposed = transpose(square4d);
    CHECK(transposed->ndim() == 2);
    CHECK(transposed->dim(0) == 2 && transposed->dim(1) == 2);
    std::cout << "[ok] transpose: forward + grad_check ok\n";
}

void test_reshape() {
    auto x = Var::make(Mat::Random(2, 6));
    auto y = reshape(x, 3, 4);
    CHECK(y->data.rows() == 3 && y->data.cols() == 4);
    CHECK_NEAR(y->data(0, 0), x->data(0, 0), 1e-5f);
    CHECK_NEAR(y->data(2, 3), x->data(1, 5), 1e-5f);

    auto f = [](VarPtr x) { return sum(reshape(x, 3, 4)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] reshape: forward + grad_check ok\n";
}

void test_logical_4d_shape() {
    auto x = Var::make4d(Mat::Random(2, 3 * 4 * 5), 2, 3, 4, 5);
    CHECK(x->is4d());
    CHECK(x->ndim() == 4);
    CHECK(x->numel() == 2 * 3 * 4 * 5);
    CHECK(x->dim(0) == 2);
    CHECK(x->dim(1) == 3);
    CHECK(x->dim(2) == 4);
    CHECK(x->dim(3) == 5);

    auto y = relu(x);
    CHECK(y->is4d());
    CHECK(y->dim(0) == 2 && y->dim(1) == 3 && y->dim(2) == 4 && y->dim(3) == 5);

    y->view({2, 3 * 4 * 5});
    CHECK(y->ndim() == 2);
    CHECK(y->dim(0) == 2 && y->dim(1) == 60);

    auto reshaped = reshape(x, 4, 30);
    CHECK(reshaped->ndim() == 2);
    CHECK(reshaped->dim(0) == 4 && reshaped->dim(1) == 30);
    std::cout << "[ok] logical 4D shape metadata + same-footprint op propagation\n";
}

void test_concat() {
    auto a = Var::make(Mat::Random(2, 3));
    auto b = Var::make(Mat::Random(4, 3));
    auto y = concat({a, b});
    CHECK(y->data.rows() == 6 && y->data.cols() == 3);
    CHECK_NEAR(y->data(0, 0), a->data(0, 0), 1e-5f);
    CHECK_NEAR(y->data(5, 2), b->data(3, 2), 1e-5f);

    auto f = [b](VarPtr a) { return sum(concat({a, b})); };
    CHECK(grad_check(f, a));
    std::cout << "[ok] concat: forward + grad_check ok\n";
}

void test_scale() {
    auto x = Var::make(Mat::Random(2, 2));
    auto y = sum(scale(x, 3.f));
    y->backward();
    CHECK_NEAR(x->grad(0, 0), 3.f, 1e-4f);
    x->zero_grad();
    auto f = [](VarPtr x) { return sum(scale(x, 2.5f)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] scale: grad_check ok\n";
}

int main() {
    test_shared_x_plus_x();
    test_shared_x_cubed();
    test_grad_check_ops();
    test_broadcast_add();
    test_softmax();
    test_log_softmax();
    test_transpose();
    test_reshape();
    test_logical_4d_shape();
    test_concat();
    test_scale();
    std::cout << "\nALL CORE TESTS PASSED\n";
    return 0;
}
