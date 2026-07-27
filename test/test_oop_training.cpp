// test/test_oop_training.cpp — Phase 5a training gate.
//
// Public-consumer TU. The new module/optim/loss/ops headers
// (Eigen-free, CUDA-header free) are included at the top; any leakage
// of EIGEN_* / CUDA_* macros into this translation unit fires the
// preprocessor #error below.
//
// Phase 5a contract coverage:
//   * nn::Linear forward shape and operator() dispatch.
//   * Deterministic {weight, bias} parameter/name order.
//   * register_parameter rejects empty / duplicate / non-trainable.
//   * nn::Module::zero_grad and optim::SGD::zero_grad clear leaves.
//   * optim::SGD exact scalar and vector in-place updates under a
//     finite positive learning rate.
//   * Tensor alias taken before step() observes the post-step values
//     (shared-mutable contract on parameter storage).
//   * optim::SGD skips parameters without requires_grad / has_grad.
//   * Two manual nn::Linear layers + relu + mse_loss + backward + SGD
//     achieve a measurable loss decrease.
//   * Invalid Linear constructor inputs and invalid SGD learning rates
//     raise std::invalid_argument.

#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "autograd/core/loss.h"
#include "autograd/core/ops.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Phase 5a new module/optim/loss/ops headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Phase 5a new module/optim/loss/ops headers must not include CUDA runtime"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int passed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #cond, __FILE__, __LINE__); \
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

float read_scalar(const ag::Tensor& t) {
    std::vector<float> one(1);
    t.copy_to_host(one.data(), 1);
    return one[0];
}

void check_near(const std::vector<float>& a, const std::vector<float>& b,
                float tol = 1e-5f) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(std::fabs(a[i] - b[i]) <= tol);
    }
}

// Test-only derived Module so the test can exercise the protected
// register_parameter through the public surface area. It does not
// depend on any internal headers.
struct TestModule : public ag::nn::Module {
    ag::Variable forward(const ag::Variable& input) override { return input; }
    void do_register(const std::string& name, const ag::Variable& v) {
        register_parameter(name, v);
    }
};

static_assert(!std::is_copy_constructible<ag::nn::Linear>::value, "");
static_assert(!std::is_copy_assignable<ag::nn::Linear>::value, "");
static_assert(std::is_move_constructible<ag::nn::Linear>::value, "");

void test_linear_forward_and_parens() {
    ag::nn::Linear lin(4, 3);
    ag::Variable x(make_tensor({1.f, 2.f, 3.f, 4.f,
                                5.f, 6.f, 7.f, 8.f,
                                9.f, 10.f, 11.f, 12.f,
                                13.f, 14.f, 15.f, 16.f,
                                17.f, 18.f, 19.f, 20.f},
                               ag::Shape{5, 4}));
    ag::Variable y = lin.forward(x);
    CHECK((y.value().shape() == ag::Shape{5, 3}));
    ag::Variable via_op = lin(x);
    check_near(read_values(via_op.value()), read_values(y.value()));
    report("nn::Linear forward shape and operator()");
}

void test_parameters_order_and_named() {
    ag::nn::Linear lin(3, 2);
    auto params1 = lin.parameters();
    auto params2 = lin.parameters();
    CHECK(params1.size() == 2);
    CHECK(params2.size() == 2);
    CHECK((params1[0].value().shape() == ag::Shape{3, 2}));
    CHECK((params1[1].value().shape() == ag::Shape{1, 2}));
    auto named = lin.named_parameters();
    auto named2 = lin.named_parameters();
    CHECK(named.size() == 2);
    CHECK(named[0].name == "weight");
    CHECK(named[1].name == "bias");
    CHECK(named2[0].name == "weight");
    CHECK(named2[1].name == "bias");
    report("nn::Linear parameters/named_parameters deterministic order");
}

void test_register_parameter_validation() {
    ag::Tensor t = ag::Tensor::zeros(ag::Shape{2});
    ag::Variable trainable(t, true);
    ag::Variable non_trainable(t, false);

    // Empty name rejected.
    {
        TestModule m;
        CHECK_THROWS_AS(m.do_register("", trainable), std::invalid_argument);
    }
    // Duplicate name rejected.
    {
        TestModule m;
        m.do_register("w", trainable);
        CHECK_THROWS_AS(m.do_register("w", trainable), std::invalid_argument);
    }
    // Non-trainable parameter rejected.
    {
        TestModule m;
        CHECK_THROWS_AS(m.do_register("w", non_trainable),
                         std::invalid_argument);
    }
    report("register_parameter rejects empty, duplicate, non-trainable");
}

void test_parameter_alias_visibility() {
    ag::nn::Linear lin(3, 2);
    ag::Tensor alias_weight = lin.weight().value();
    ag::Tensor alias_bias  = lin.bias().value();
    auto params = lin.parameters();
    CHECK((alias_weight.shape() == params[0].value().shape()));
    CHECK((alias_bias.shape()  == params[1].value().shape()));
    std::vector<float> ones(alias_weight.elements(), 1.f);
    alias_weight.copy_from_host(ones.data(), ones.size());
    check_near(read_values(lin.weight().value()),
               std::vector<float>(lin.weight().value().elements(), 1.f));
    report("nn::Linear parameter Tensor alias shares storage");
}

void test_zero_grad_clears_leaves() {
    ag::nn::Linear lin(2, 2);
    ag::Variable x(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{2, 2}));
    ag::sum(lin(x)).backward();
    CHECK(lin.weight().has_grad());
    CHECK(lin.bias().has_grad());

    lin.zero_grad();
    CHECK(!lin.weight().has_grad());
    CHECK(!lin.bias().has_grad());

    // Restore grads and clear through SGD::zero_grad.
    ag::sum(lin(x)).backward();
    CHECK(lin.weight().has_grad());
    ag::optim::SGD sgd(lin.parameters(), 0.01f);
    sgd.zero_grad();
    CHECK(!lin.weight().has_grad());
    CHECK(!lin.bias().has_grad());
    report("nn::Module::zero_grad and optim::SGD::zero_grad clear leaves");
}

void test_sgd_exact_scalar_update() {
    ag::Variable p(make_tensor({2.0f}, ag::Shape{}), true);
    p.backward(make_tensor({0.5f}, ag::Shape{}));

    ag::optim::SGD sgd({p}, 0.1f);
    sgd.step();
    CHECK(std::fabs(read_scalar(p.value()) - 1.95f) <= 1e-6f);

    sgd.zero_grad();
    p.backward(make_tensor({0.5f}, ag::Shape{}));
    sgd.step();
    CHECK(std::fabs(read_scalar(p.value()) - 1.90f) <= 1e-6f);
    report("optim::SGD exact scalar in-place update");
}

void test_sgd_exact_vector_update() {
    ag::Variable p(make_tensor({1.f, 2.f, 3.f, 4.f}, ag::Shape{4}), true);
    p.backward(make_tensor({0.1f, 0.2f, 0.3f, 0.4f}, ag::Shape{4}));

    ag::optim::SGD sgd({p}, 0.5f);
    sgd.step();
    check_near(read_values(p.value()), {0.95f, 1.90f, 2.85f, 3.80f});
    report("optim::SGD exact vector in-place update");
}

void test_sgd_tensor_alias_sees_inplace_update() {
    ag::nn::Linear lin(2, 2);
    ag::Tensor alias_weight = lin.weight().value();
    ag::Tensor alias_bias  = lin.bias().value();
    std::vector<float> weight_before = read_values(alias_weight);
    std::vector<float> bias_before   = read_values(alias_bias);

    ag::Variable x(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{2, 2}));
    ag::Variable y = lin(x);
    ag::Variable loss = ag::mse_loss(
        y, make_tensor({0.f, 0.f, 0.f, 0.f}, ag::Shape{2, 2}));
    loss.backward();

    ag::optim::SGD sgd(lin.parameters(), 0.01f);
    sgd.step();

    // Pre-step alias must reflect post-step values.
    check_near(read_values(alias_weight), read_values(lin.weight().value()));
    check_near(read_values(alias_bias),  read_values(lin.bias().value()));
    bool weight_changed = false;
    for (std::size_t i = 0; i < weight_before.size(); ++i) {
        if (std::fabs(read_values(alias_weight)[i] - weight_before[i]) >
            1e-7f) {
            weight_changed = true;
            break;
        }
    }
    bool bias_changed = false;
    for (std::size_t i = 0; i < bias_before.size(); ++i) {
        if (std::fabs(read_values(alias_bias)[i] - bias_before[i]) > 1e-7f) {
            bias_changed = true;
            break;
        }
    }
    CHECK(weight_changed);
    CHECK(bias_changed);
    report("optim::SGD in-place update is alias-visible");
}

void test_sgd_skips_parameters_without_grad() {
    ag::Variable frozen(make_tensor({7.f, 8.f, 9.f}, ag::Shape{3}), false);
    ag::Variable trainable(make_tensor({1.f, 2.f, 3.f}, ag::Shape{3}), true);
    trainable.backward(make_tensor({0.1f, 0.1f, 0.1f}, ag::Shape{3}));

    std::vector<float> frozen_before = read_values(frozen.value());
    ag::optim::SGD sgd({frozen, trainable}, 0.5f);
    sgd.step();
    // frozen: no requires_grad, untouched.
    check_near(read_values(frozen.value()), frozen_before);
    // trainable: 1.0 - 0.5*0.1 = 0.95, etc.
    check_near(read_values(trainable.value()), {0.95f, 1.95f, 2.95f});

    sgd.zero_grad();
    CHECK(!trainable.has_grad());
    sgd.step();
    check_near(read_values(trainable.value()), {0.95f, 1.95f, 2.95f});
    report("optim::SGD skips parameters without requires_grad or grad");
}

void test_invalid_linear_constructors() {
    CHECK_THROWS_AS(ag::nn::Linear(-1, 4), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Linear(4, -1), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Linear(0, 4), std::invalid_argument);
    CHECK_THROWS_AS(ag::nn::Linear(4, 0), std::invalid_argument);
    report("nn::Linear rejects non-positive in/out features");
}

void test_invalid_sgd_learning_rate() {
    ag::optim::SGD zero_lr({}, 0.f);
    CHECK(zero_lr.learning_rate() == 0.f);
    CHECK_THROWS_AS(ag::optim::SGD({}, -1e-3f), std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::SGD({}, std::numeric_limits<float>::infinity()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::SGD({},
                                    -std::numeric_limits<float>::infinity()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::SGD({}, std::nanf("")),
                    std::invalid_argument);
    report("optim::SGD accepts zero and rejects negative/non-finite rates");
}

void test_training_gate_loss_decrease() {
    ag::nn::Linear l1(4, 8);
    ag::nn::Linear l2(8, 2);

    const std::vector<float> x_data = {
        0.5f, -0.2f, 0.1f, 0.7f,
        -0.4f, 0.3f, -0.1f, 0.2f,
        0.6f, -0.5f, 0.4f, 0.0f,
        -0.3f, 0.8f, -0.6f, 0.1f,
    };
    const std::vector<float> y_data = {
        1.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        0.f, 1.f,
    };
    ag::Variable x(make_tensor(x_data, ag::Shape{4, 4}));
    ag::Tensor y_t = make_tensor(y_data, ag::Shape{4, 2});

    auto compute_loss = [&]() {
        return ag::mse_loss(l2.forward(ag::relu(l1.forward(x))), y_t);
    };

    float initial_loss = read_scalar(compute_loss().value());
    CHECK(std::isfinite(initial_loss));
    CHECK(initial_loss > 0.f);

    std::vector<ag::Variable> all_params;
    for (auto& v : l1.parameters()) all_params.push_back(v);
    for (auto& v : l2.parameters()) all_params.push_back(v);
    ag::optim::SGD sgd(all_params, 0.05f);

    constexpr int kSteps = 200;
    float last = initial_loss;
    for (int step = 0; step < kSteps; ++step) {
        sgd.zero_grad();
        ag::Variable loss = compute_loss();
        if (step == kSteps - 1) last = read_scalar(loss.value());
        loss.backward();
        sgd.step();
    }
    CHECK(std::isfinite(last));
    CHECK(last < initial_loss * 0.95f);
    report("two-layer Linear + relu + mse_loss + SGD loss decreases");
}

}  // namespace

int main() {
    test_linear_forward_and_parens();
    test_parameters_order_and_named();
    test_register_parameter_validation();
    test_parameter_alias_visibility();
    test_zero_grad_clears_leaves();
    test_sgd_exact_scalar_update();
    test_sgd_exact_vector_update();
    test_sgd_tensor_alias_sees_inplace_update();
    test_sgd_skips_parameters_without_grad();
    test_invalid_linear_constructors();
    test_invalid_sgd_learning_rate();
    test_training_gate_loss_decrease();

    std::printf("\nALL OOP TRAINING TESTS PASSED (%d)\n", passed);
    return 0;
}
