#include "autograd/core/variable.h"
#include "autograd/core/ops.h"
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
#include <stdexcept>
#include <vector>

using ag::Shape;
using ag::Tensor;
using ag::Variable;

namespace {

int passed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #cond, __FILE__, __LINE__); \
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

Tensor tensor(const std::vector<float>& values, const Shape& shape) {
    CHECK(static_cast<std::size_t>(shape.numel()) == values.size());
    return Tensor::from_host(values.empty() ? nullptr : values.data(), shape);
}

Tensor scalar(float value) {
    return Tensor::from_host(&value, Shape{});
}

std::vector<float> values(const Tensor& value) {
    std::vector<float> out(value.elements());
    value.copy_to_host(out.empty() ? nullptr : out.data(), out.size());
    return out;
}

void check_near(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tolerance = 1e-5f) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::fabs(actual[i] - expected[i]) <= tolerance);
    }
}

Tensor scaled_tensor(const Tensor& input, float scale) {
    auto data = values(input);
    for (float& value : data) value *= scale;
    return tensor(data, input.shape());
}

float combined_forward(const std::vector<float>& x,
                       const std::vector<float>& c) {
    Variable vx(tensor(x, Shape{static_cast<int64_t>(x.size())}));
    Variable vc(tensor(c, Shape{static_cast<int64_t>(c.size())}));
    Variable out = ag::sum(ag::add(ag::mul(vx, vc), ag::scale(vx, 2.f)));
    return values(out.value())[0];
}

void test_variable_basics() {
    Variable empty;
    CHECK(empty.value().empty());
    CHECK(!empty.requires_grad());
    CHECK(!empty.has_grad());
    CHECK_THROWS(empty.grad());

    Variable x(tensor({1.f, 2.f}, Shape{2}), true);
    Variable alias = x;
    ag::sum(x).backward();
    CHECK(x.has_grad());
    check_near(values(alias.grad()), {1.f, 1.f});

    x.zero_grad();
    CHECK(!x.has_grad());
    CHECK(!alias.has_grad());
    report("Variable accessors, aliases, grad(), and zero_grad()");
}

void test_forward_ops() {
    Variable a(tensor({1.f, 2.f, 3.f}, Shape{3}));
    Variable b(tensor({4.f, 5.f, 6.f}, Shape{3}));

    check_near(values(ag::add(a, b).value()), {5.f, 7.f, 9.f});
    check_near(values(ag::mul(a, b).value()), {4.f, 10.f, 18.f});
    check_near(values(ag::scale(a, -2.f).value()), {-2.f, -4.f, -6.f});

    Variable total = ag::sum(a);
    CHECK(total.value().shape() == Shape{});
    check_near(values(total.value()), {6.f});
    CHECK(!total.requires_grad());

    Variable wrong(tensor({1.f, 2.f}, Shape{2}));
    CHECK_THROWS(ag::add(a, wrong));
    CHECK_THROWS(ag::mul(a, wrong));
    report("add/mul/scale/sum forward and shape validation");
}

void test_analytic_gradients() {
    {
        Variable x(tensor({1.f, -2.f}, Shape{2}), true);
        Variable c(tensor({3.f, 4.f}, Shape{2}));
        ag::sum(ag::add(x, c)).backward();
        check_near(values(x.grad()), {1.f, 1.f});
    }
    {
        Variable x(tensor({1.f, -2.f}, Shape{2}), true);
        Variable c(tensor({3.f, 4.f}, Shape{2}));
        ag::sum(ag::mul(x, c)).backward();
        check_near(values(x.grad()), {3.f, 4.f});
    }
    {
        Variable x(tensor({1.f, -2.f}, Shape{2}), true);
        ag::sum(ag::scale(x, 2.5f)).backward();
        check_near(values(x.grad()), {2.5f, 2.5f});
    }
    {
        Variable x(tensor({1.f, -2.f}, Shape{2}), true);
        ag::sum(x).backward();
        check_near(values(x.grad()), {1.f, 1.f});
    }
    {
        Variable x(tensor({2.f, 3.f}, Shape{2}), true);
        Variable y(tensor({4.f, 5.f}, Shape{2}), true);
        Variable product = ag::mul(x, y);

        Tensor x_storage = x.value();
        Tensor y_storage = y.value();
        const std::vector<float> changed{10.f, 10.f};
        x_storage.copy_from_host(changed.data(), changed.size());
        y_storage.copy_from_host(changed.data(), changed.size());

        ag::sum(product).backward();
        check_near(values(x.grad()), {4.f, 5.f});
        check_near(values(y.grad()), {2.f, 3.f});
    }
    report("analytic gradients for vertical-slice operations");
}

void test_finite_difference() {
    const std::vector<float> x{0.4f, -1.2f, 2.f};
    const std::vector<float> c{3.f, -0.5f, 1.5f};
    Variable vx(tensor(x, Shape{3}), true);
    Variable vc(tensor(c, Shape{3}));
    ag::sum(ag::add(ag::mul(vx, vc), ag::scale(vx, 2.f))).backward();
    const auto analytic = values(vx.grad());

    constexpr float eps = 1e-3f;
    std::vector<float> numeric(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto plus = x;
        auto minus = x;
        plus[i] += eps;
        minus[i] -= eps;
        numeric[i] =
            (combined_forward(plus, c) - combined_forward(minus, c)) /
            (2.f * eps);
    }
    check_near(analytic, numeric, 2e-3f);
    report("finite-difference gradient through add/mul/scale/sum");
}

void test_shared_graph_and_repeated_backward() {
    Variable x(tensor({2.f, -3.f}, Shape{2}), true);
    Variable square = ag::mul(x, x);
    Variable diamond = ag::add(square, square);
    Variable loss = ag::sum(diamond);

    loss.backward();
    check_near(values(x.grad()), {8.f, -12.f});
    loss.backward();
    check_near(values(x.grad()), {16.f, -24.f});
    report("shared-node topology and repeated gradient accumulation");
}

void test_upstream_detach_and_zero_grad() {
    Variable x(tensor({1.f, 2.f}, Shape{2}), true);
    Variable y = ag::scale(x, 3.f);
    y.backward(tensor({0.5f, -2.f}, Shape{2}));
    check_near(values(x.grad()), {1.5f, -6.f});
    CHECK_THROWS(y.backward());

    const auto before = values(x.grad());
    CHECK_THROWS(y.backward(tensor({1.f}, Shape{1})));
    check_near(values(x.grad()), before);

    CHECK(y.has_grad());
    x.zero_grad();
    CHECK(!x.has_grad());
    CHECK(y.has_grad());

    Variable detached = y.detach();
    CHECK(!detached.requires_grad());
    CHECK(!detached.has_grad());
    Tensor shared = detached.value();
    const std::vector<float> changed{7.f, 8.f};
    shared.copy_from_host(changed.data(), changed.size());
    check_near(values(y.value()), {7.f, 8.f});
    CHECK_THROWS(ag::sum(detached).backward());
    report("explicit upstream, detach storage sharing, and local zero_grad");
}

void test_custom_op_and_transaction() {
    Variable x(tensor({1.f, 2.f}, Shape{2}), true);
    Variable y(tensor({3.f, 4.f}, Shape{2}), true);
    Tensor output = tensor({7.f, 10.f}, Shape{2});  // x + 2*y
    Variable custom = ag::make_custom_variable(
        output, {x, y}, [](const Tensor& grad) {
            return std::vector<Tensor>{grad.clone(), scaled_tensor(grad, 2.f)};
        });
    ag::sum(custom).backward();
    check_near(values(x.grad()), {1.f, 1.f});
    check_near(values(y.grad()), {2.f, 2.f});

    const auto x_before = values(x.grad());
    Variable wrong_count = ag::make_custom_variable(
        scalar(0.f), {x}, [](const Tensor&) {
            return std::vector<Tensor>{};
        });
    CHECK_THROWS(wrong_count.backward());
    check_near(values(x.grad()), x_before);

    Variable wrong_shape = ag::make_custom_variable(
        scalar(0.f), {x}, [](const Tensor&) {
            return std::vector<Tensor>{scalar(1.f)};
        });
    CHECK_THROWS(wrong_shape.backward());
    check_near(values(x.grad()), x_before);

    Variable throwing = ag::make_custom_variable(
        scalar(0.f), {x}, [](const Tensor&) -> std::vector<Tensor> {
            throw std::runtime_error("custom failure");
        });
    CHECK_THROWS(throwing.backward());
    check_near(values(x.grad()), x_before);

    Variable constant(tensor({5.f, 6.f}, Shape{2}));
    Variable validate_non_grad = ag::make_custom_variable(
        scalar(0.f), {x, constant}, [](const Tensor&) {
            return std::vector<Tensor>{
                tensor({1.f, 1.f}, Shape{2}),
                scalar(1.f),
            };
        });
    CHECK_THROWS(validate_non_grad.backward());
    check_near(values(x.grad()), x_before);
    report("custom-op ordering, validation, and transactional rollback");
}

}  // namespace

int main() {
    test_variable_basics();
    test_forward_ops();
    test_analytic_gradients();
    test_finite_difference();
    test_shared_graph_and_repeated_backward();
    test_upstream_detach_and_zero_grad();
    test_custom_op_and_transaction();

    std::printf("\nALL AUTOGRAD CORE TESTS PASSED (%d)\n", passed);
    return 0;
}
