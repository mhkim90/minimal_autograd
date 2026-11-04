// test_core.cpp — autograd core: shared-node correctness, all built-in ops.

#include "autograd/core/ops.h"
#include "autograd/extension/custom_op.h"
#include "grad_check.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>

using namespace ag;
using namespace ag_test;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (!std::isfinite(_a) || !std::isfinite(_b) || std::fabs(_a - _b) > (tol)) { \
        std::cerr << "FAIL: " << #a << " (" << _a << ") vs " << #b << " (" << _b \
                  << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CHECK_EQ_TENSOR(a, b) do { \
    auto _A = (a); auto _B = (b); \
    if (_A.shape() != _B.shape() || \
        max_diff(_A, _B) > 1e-4f) { \
        std::cerr << "FAIL: tensor mismatch at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// --- Phase 2: shared-node correctness ---

void test_shared_x_plus_x() {
    auto x = Variable(filled(Shape{2, 2}, 3.f), true);
    auto y = sum(add(x, x));     // y = sum(2x), scalar
    y.backward();
    // d/dx (2x) = 2, NOT 4 (no double-counting).
    CHECK_NEAR(at(x.grad(), 0, 0), 2.f, 1e-4f);
    CHECK_NEAR(at(x.grad(), 1, 1), 2.f, 1e-4f);
    std::cout << "[ok] x + x -> grad = 2 (no double-count)\n";
}

void test_shared_x_cubed() {
    auto x  = Variable(filled(Shape{1, 1}, 2.5f), true);
    auto x2 = mul(x, x);
    auto y  = mul(x2, x);   // y = x^3
    y.backward();
    // d/dx (x^3) = 3x^2 = 3 * 6.25 = 18.75
    CHECK_NEAR(at(x.grad(), 0, 0), 18.75f, 1e-3f);
    std::cout << "[ok] x*x*x -> grad = 3x^2 (no double-count)\n";
}

void test_grad_check_ops() {
    // AddFn
    {
        auto x = Variable(random_tensor(Shape{3, 3}), true);
        auto f = [](Variable x) { return sum(add(x, x)); };
        CHECK(grad_check(f, x));
    }
    // MulFn
    {
        auto x = Variable(random_tensor(Shape{2, 2}), true);
        auto f = [](Variable x) { return sum(mul(x, x)); };
        CHECK(grad_check(f, x));
    }
    // MatMulFn
    {
        auto x  = Variable(random_tensor(Shape{3, 4}), true);
        auto w  = Variable(random_tensor(Shape{4, 2}), true);
        auto f  = [w](Variable x) { return sum(matmul(x, w)); };
        CHECK(grad_check(f, x));
    }
    // ReLUFn
    {
        auto x = Variable(random_tensor(Shape{2, 2}), true);
        auto f = [](Variable x) { return sum(relu(x)); };
        CHECK(grad_check(f, x));
    }
    // SumFn
    {
        auto x = Variable(random_tensor(Shape{2, 2}), true);
        auto f = [](Variable x) { return sum(sum(x)); };
        CHECK(grad_check(f, x));
    }
    std::cout << "[ok] grad_check passes for Add/Mul/MatMul/ReLU/Sum\n";
}

// --- Phase 2c: broadcast bias add ---

void test_broadcast_add() {
    auto a = Variable(filled(Shape{4, 3}, 1.f), true);   // 4-row input
    auto b = Variable(filled(Shape{1, 3}, 2.f), true);   // row bias
    auto y = sum(broadcast_add(a, b));              // reduce to scalar
    y.backward();

    // grad w.r.t. a is all ones
    CHECK_NEAR(at(a.grad(), 0, 0), 1.f, 1e-4f);
    CHECK_NEAR(at(a.grad(), 3, 2), 1.f, 1e-4f);
    // grad w.r.t. b sums over the batch axis → 4 in each entry
    CHECK_NEAR(at(b.grad(), 0, 0), 4.f, 1e-4f);
    CHECK_NEAR(at(b.grad(), 0, 2), 4.f, 1e-4f);
    std::cout << "[ok] BroadcastAddFn: bias grad sums over batch\n";
}

// --- Phase 3: shape ops + softmax ---

void test_softmax() {
    auto x = Variable(random_tensor(Shape{2, 3}), true);
    auto y = softmax(x);
    CHECK_NEAR(row_sum(y.value(), 0), 1.f, 1e-4f);
    CHECK_NEAR(row_sum(y.value(), 1), 1.f, 1e-4f);

    auto f = [](Variable x) { return sum(softmax(x)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] softmax: forward sums to 1, grad_check ok\n";
}

void test_log_softmax() {
    auto x = Variable(random_tensor(Shape{2, 3}), true);
    auto f = [](Variable x) { return sum(log_softmax(x)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] log_softmax: grad_check ok\n";
}

void test_transpose() {
    auto x = Variable(tensor(Shape{2, 3}, {1, 2, 3, 4, 5, 6}), true);
    auto y = transpose(x);
    CHECK_EQ_TENSOR(y.value(), Tensor(tensor(Shape{3, 2}, {1, 4, 2, 5, 3, 6})));
    auto f = [](Variable x) { return sum(transpose(x)); };
    CHECK(grad_check(f, x));

    auto square4d = Variable(random_tensor(Shape{2, 1, 1, 2}), true);
    auto transposed = transpose(square4d);
    CHECK(transposed.value().shape().ndim() == 4);
    CHECK((transposed.value().shape() == Shape{2, 1, 2, 1}));
    std::cout << "[ok] transpose: forward + grad_check ok\n";
}

void test_reshape() {
    auto x = Variable(random_tensor(Shape{2, 6}), true);
    auto y = reshape(x, 3, 4);
    CHECK(y.value().shape()[0] == 3 && y.value().shape()[1] == 4);
    CHECK_NEAR(at(y.value(), 0, 0), at(x.value(), 0, 0), 1e-5f);
    CHECK_NEAR(at(y.value(), 2, 3), at(x.value(), 1, 5), 1e-5f);

    auto f = [](Variable x) { return sum(reshape(x, 3, 4)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] reshape: forward + grad_check ok\n";
}

void test_logical_4d_shape() {
    auto x = Variable(random_tensor(Shape{2, 3, 4, 5}), true);
    CHECK(x.value().shape().rank() == 4);
    CHECK(x.value().shape().ndim() == 4);
    CHECK(x.value().elements() == 2 * 3 * 4 * 5);
    CHECK(x.value().shape()[0] == 2);
    CHECK(x.value().shape()[1] == 3);
    CHECK(x.value().shape()[2] == 4);
    CHECK(x.value().shape()[3] == 5);
    CHECK(contiguous_stride(x.value().shape()) == Stride({60, 20, 5, 1}));

    auto y = relu(x);
    CHECK(y.value().shape().rank() == 4);
    CHECK(y.value().shape()[0] == 2 && y.value().shape()[1] == 3 && y.value().shape()[2] == 4 && y.value().shape()[3] == 5);

    y = reshape(y, 2, 3 * 4 * 5);
    CHECK(y.value().shape().ndim() == 2);
    CHECK(y.value().shape()[0] == 2 && y.value().shape()[1] == 60);
    CHECK(contiguous_stride(y.value().shape()) == Stride({60, 1}));

    auto reshaped = reshape(x, 4, 30);
    CHECK(reshaped.value().shape().ndim() == 2);
    CHECK(reshaped.value().shape()[0] == 4 && reshaped.value().shape()[1] == 30);
    std::cout << "[ok] logical 4D shape metadata + same-footprint op propagation\n";
}

void test_concat() {
    auto a = Variable(random_tensor(Shape{2, 3}), true);
    auto b = Variable(random_tensor(Shape{4, 3}), true);
    auto y = concat({a, b});
    CHECK(y.value().shape()[0] == 6 && y.value().shape()[1] == 3);
    CHECK_NEAR(at(y.value(), 0, 0), at(a.value(), 0, 0), 1e-5f);
    CHECK_NEAR(at(y.value(), 5, 2), at(b.value(), 3, 2), 1e-5f);

    auto f = [b](Variable a) { return sum(concat({a, b})); };
    CHECK(grad_check(f, a));
    std::cout << "[ok] concat: forward + grad_check ok\n";
}

void test_scale() {
    auto x = Variable(random_tensor(Shape{2, 2}), true);
    auto y = sum(scale(x, 3.f));
    y.backward();
    CHECK_NEAR(at(x.grad(), 0, 0), 3.f, 1e-4f);
    x.zero_grad();
    auto f = [](Variable x) { return sum(scale(x, 2.5f)); };
    CHECK(grad_check(f, x));
    std::cout << "[ok] scale: grad_check ok\n";
}

void test_sub_div_shape_mismatch() {
    auto a = Variable(filled(Shape{2, 3}, 1.f), true);   // (2, 3)
    auto b = Variable(filled(Shape{4, 3}, 1.f), true);   // incompatible (4, 3)

    bool sub_threw = false;
    try {
        sub(a, b);
    } catch (const std::invalid_argument&) {
        sub_threw = true;
    }
    CHECK(sub_threw);

    bool div_threw = false;
    try {
        div_op(a, b);
    } catch (const std::invalid_argument&) {
        div_threw = true;
    }
    CHECK(div_threw);

    auto broadcastable = Variable(filled(Shape{1, 3}, 1.f), true);
    bool broadcast_sub_threw = false;
    try {
        sub(a, broadcastable);
    } catch (const std::invalid_argument&) {
        broadcast_sub_threw = true;
    }
    CHECK(broadcast_sub_threw);

    bool broadcast_div_threw = false;
    try {
        div_op(a, broadcastable);
    } catch (const std::invalid_argument&) {
        broadcast_div_threw = true;
    }
    CHECK(broadcast_div_threw);

    std::cout << "[ok] sub/div_op reject (2,3) vs (4,3) and (2,3) vs (1,3) with invalid_argument\n";
}

void test_backward_exception_rolls_back_grads() {
    auto x = Variable(filled(Shape{2, 2}, 1.f), true);
    auto y = scale(sum(x), 0.f);
    seed_grad(x, filled(Shape{2, 2}, 7.f));
    seed_grad(y, filled(Shape{}, 3.f));
    const Tensor x_before = x.grad().clone();
    const Tensor y_before = y.grad().clone();
    auto throwing = make_custom_variable(filled(Shape{}, 0.f), {x, y},
        [](const Tensor&) -> std::vector<Tensor> {
            throw std::runtime_error("intentional backward failure");
        });

    bool threw = false;
    try {
        throwing.backward();
    } catch (const std::runtime_error&) {
        threw = true;
    }

    CHECK(threw);
    CHECK_EQ_TENSOR(x.grad(), x_before);
    CHECK_EQ_TENSOR(y.grad(), y_before);
    // A strict ancestor chain forces a successful callback before the throw;
    // no sibling traversal order can bypass its pending contributions.
    bool callback_ran = false;
    bool ancestor_ran = false;
    auto fresh = Variable(filled(Shape{2, 2}, 2.f), true);
    auto ancestor = make_custom_variable(filled(Shape{}, 0.f), {x, y},
        [&](const Tensor& output_grad) -> std::vector<Tensor> {
            ancestor_ran = true;
            CHECK(callback_ran);
            CHECK_EQ_TENSOR(output_grad, filled(Shape{}, 5.f));
            throw std::runtime_error("intentional ancestor backward failure");
        });
    auto successful = make_custom_variable(filled(Shape{}, 0.f), {ancestor, x, fresh},
        [&](const Tensor& output_grad) -> std::vector<Tensor> {
            CHECK(!ancestor_ran);
            CHECK_EQ_TENSOR(output_grad, filled(Shape{}, 1.f));
            callback_ran = true;
            return {filled(Shape{}, 5.f), filled(Shape{2, 2}, 11.f),
                    filled(Shape{2, 2}, 13.f)};
        });
    threw = false;
    try {
        successful.backward();
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "intentional ancestor backward failure");
        threw = true;
    }
    CHECK(threw);
    CHECK(callback_ran);
    CHECK(ancestor_ran);
    CHECK_EQ_TENSOR(x.grad(), x_before);
    CHECK_EQ_TENSOR(y.grad(), y_before);
    CHECK(!fresh.has_grad());
    CHECK(!ancestor.has_grad());
    CHECK(!successful.has_grad());
    std::cout << "[ok] backward exception restores prior grads\n";
    std::cout << "[ok] rollback after successful callback: pending contributions discarded, new nodes uncommitted\n";
}

void test_grad_check_helpers_restore_values() {
    auto x = Variable(tensor(Shape{1,2}, {.25f, -.5f}), true);
    auto other = Variable(tensor(Shape{1,2}, {.5f, .75f}), true);
    const Tensor before = x.value().clone(), other_before = other.value().clone();
    CHECK(grad_check([](Variable v) { return sum(mul(v,v)); }, x));
    CHECK_EQ_TENSOR(x.value(), before);
    CHECK(grad_check_multi([](std::vector<Variable> v) { return sum(mul(v[0],v[1])); }, {x,other}));
    CHECK_EQ_TENSOR(x.value(), before);
    CHECK_EQ_TENSOR(other.value(), other_before);
    int calls = 0;
    bool threw = false;
    try {
        (void)grad_check([&](Variable v) {
            if (++calls == 2) throw std::runtime_error("probe failure");
            return sum(v);
        }, x);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK_EQ_TENSOR(x.value(), before);
    calls = 0; threw = false;
    try {
        (void)grad_check_multi([&](std::vector<Variable> v) {
            if (++calls == 2) throw std::runtime_error("multi probe failure");
            return sum(add(v[0],v[1]));
        }, {x,other});
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK_EQ_TENSOR(x.value(), before);
    CHECK_EQ_TENSOR(other.value(), other_before);
    std::cout << "[ok] finite-difference helpers restore owned values, including failed probes\n";
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
    test_sub_div_shape_mismatch();
    test_backward_exception_rolls_back_grads();
    test_grad_check_helpers_restore_values();
    std::cout << "\nALL CORE TESTS PASSED\n";
    return 0;
}
