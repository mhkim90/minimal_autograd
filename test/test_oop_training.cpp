// test/test_oop_training.cpp — OOP training-stack tests.
//
// Public-consumer TU. The new module/optim/loss/ops headers
// (Eigen-free, CUDA-header free) are included at the top; any leakage
// of EIGEN_* / CUDA_* macros into this translation unit fires the
// preprocessor #error below.
//
// Linear and SGD contract coverage:
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
//
// Composition and Adam contract coverage:
//   * nn::Sequential forward, numeric naming, nested depth-first
//     named_parameters, recursive zero_grad.
//   * nn::ReLU has no parameters and routes through public ag::relu.
//   * register_module validates empty/duplicate/null/cross-kind.
//   * optim::Adam first-step exact arithmetic; multi-step trajectory
//     matches an independent scalar reference.
//   * state() deep-clones moments; load_state validates the entire
//     snapshot before mutating any live state, and a partial failure
//     leaves the optimizer and parameters unchanged.
//   * zero_grad preserves moments and step count.

#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "autograd/core/loss.h"
#include "autograd/core/ops.h"

#if defined(EIGEN_WORLD_VERSION) || defined(EIGEN_MAJOR_VERSION) || \
    defined(EIGEN_MINOR_VERSION)
#error "Public module/optim/loss/ops headers must not include Eigen"
#endif
#if defined(CUDART_VERSION) || defined(__CUDART_API_VERSION) || \
    defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "Public module/optim/loss/ops headers must not include CUDA runtime"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
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
// register_parameter / register_module through the public surface
// area. It does not depend on any internal headers.
struct TestModule : public ag::nn::Module {
    ag::Variable forward(const ag::Variable& input) override { return input; }
    void do_register(const std::string& name, const ag::Variable& v) {
        register_parameter(name, v);
    }
    void do_register_module(const std::string& name,
                            std::shared_ptr<ag::nn::Module> m) {
        register_module(name, std::move(m));
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

// ── Composition and Adam tests ───────────────────────────────────────

void test_relu_module() {
    ag::nn::ReLU r;
    ag::Variable x(make_tensor({-1.f, 0.f, 1.f, 2.f}, ag::Shape{4}));
    check_near(read_values(r.forward(x).value()), {0.f, 0.f, 1.f, 2.f});
    CHECK(r.parameters().empty());
    CHECK(r.named_parameters().empty());
    report("nn::ReLU forward and zero parameters");
}

void test_sequential_forward() {
    ag::nn::Sequential net;
    net.add(std::make_shared<ag::nn::Linear>(3, 4));
    net.add(std::make_shared<ag::nn::Linear>(4, 2));
    ag::Variable x(make_tensor({0.1f, 0.2f, 0.3f,
                                0.4f, 0.5f, 0.6f},
                               ag::Shape{2, 3}));
    ag::Variable y = net.forward(x);
    CHECK((y.value().shape() == ag::Shape{2, 2}));
    check_near(read_values(net(x).value()), read_values(y.value()));
    report("nn::Sequential forward shape and operator()");
}

void test_sequential_named_parameters_depth_first() {
    ag::nn::Sequential outer;
    auto inner = std::make_shared<ag::nn::Sequential>();
    inner->add(std::make_shared<ag::nn::Linear>(2, 3));
    inner->add(std::make_shared<ag::nn::ReLU>());
    outer.add(inner);
    outer.add(std::make_shared<ag::nn::Linear>(3, 2));

    auto named_a = outer.named_parameters();
    auto named_b = outer.named_parameters();
    CHECK(named_a.size() == 4);
    CHECK(named_a[0].name == "0.0.weight");
    CHECK(named_a[1].name == "0.0.bias");
    CHECK(named_a[2].name == "1.weight");
    CHECK(named_a[3].name == "1.bias");
    for (std::size_t i = 0; i < named_a.size(); ++i) {
        CHECK(named_a[i].name == named_b[i].name);
    }
    report("Sequential nested named_parameters depth-first and stable");
}

void test_sequential_zero_grad_recurses() {
    ag::nn::Sequential net;
    net.add(std::make_shared<ag::nn::Linear>(2, 2));
    net.add(std::make_shared<ag::nn::Linear>(2, 2));
    ag::Variable x(make_tensor({1.f, 1.f, 1.f, 1.f}, ag::Shape{2, 2}));
    ag::sum(net.forward(x)).backward();
    for (const auto& p : net.named_parameters()) CHECK(p.parameter.has_grad());
    net.zero_grad();
    for (const auto& p : net.named_parameters()) CHECK(!p.parameter.has_grad());
    report("nn::Sequential::zero_grad recurses into children");
}

void test_register_module_validation() {
    ag::Tensor t = ag::Tensor::zeros(ag::Shape{2});
    ag::Variable v(t, true);
    auto lin = std::make_shared<ag::nn::Linear>(2, 2);

    // Empty name rejected.
    {
        TestModule m;
        CHECK_THROWS_AS(m.do_register_module("", lin), std::invalid_argument);
    }
    // Null child rejected.
    {
        TestModule m;
        CHECK_THROWS_AS(m.do_register_module("child", nullptr),
                        std::invalid_argument);
    }
    // Duplicate child name rejected.
    {
        TestModule m;
        m.do_register_module("child", lin);
        CHECK_THROWS_AS(m.do_register_module("child", lin),
                        std::invalid_argument);
    }
    // Cross-kind collision: parameter "p" already exists, then register module "p".
    {
        TestModule m;
        m.do_register("p", v);
        CHECK_THROWS_AS(m.do_register_module("p", lin),
                        std::invalid_argument);
    }
    // And the reverse: module "m" then parameter "m".
    {
        TestModule m;
        m.do_register_module("m", lin);
        CHECK_THROWS_AS(m.do_register("m", v), std::invalid_argument);
    }
    // Direct and indirect cycles rejected.
    {
        auto parent = std::make_shared<TestModule>();
        CHECK_THROWS_AS(parent->do_register_module("self", parent),
                        std::invalid_argument);
        auto child = std::make_shared<TestModule>();
        parent->do_register_module("child", child);
        CHECK_THROWS_AS(child->do_register_module("parent", parent),
                        std::invalid_argument);
    }
    report("register_module validates names, children, collisions, cycles");
}

void test_recursive_parameters_through_module() {
    TestModule parent;
    parent.do_register("alpha",
                       ag::Variable(ag::Tensor::zeros(ag::Shape{2}), true));
    auto child_lin = std::make_shared<ag::nn::Linear>(2, 3);
    parent.do_register_module("child", child_lin);
    auto named = parent.named_parameters();
    CHECK(named.size() == 3);
    CHECK(named[0].name == "alpha");
    CHECK(named[1].name == "child.weight");
    CHECK(named[2].name == "child.bias");

    // Drive a real backward through the child Linear directly (the
    // test-only TestModule is a pass-through) and step SGD over the
    // recursive parameter list.
    ag::Tensor alias_child_w = child_lin->weight().value();
    std::vector<ag::Variable> all;
    for (auto& pv : parent.parameters()) all.push_back(pv);
    ag::Variable x(make_tensor({1.f, 1.f}, ag::Shape{1, 2}));
    ag::sum(child_lin->forward(x)).backward();
    ag::optim::SGD sgd(all, 0.01f);
    sgd.step();
    check_near(read_values(alias_child_w),
               read_values(child_lin->weight().value()));
    report("Module recursion + dotted names + alias-visible SGD step");
}

void test_sequential_mlp_loss_decrease() {
    ag::nn::Sequential net;
    net.add(std::make_shared<ag::nn::Linear>(4, 8));
    net.add(std::make_shared<ag::nn::ReLU>());
    net.add(std::make_shared<ag::nn::Linear>(8, 2));
    const std::vector<float> x_data = {
        0.5f, -0.2f, 0.1f, 0.7f,
        -0.4f, 0.3f, -0.1f, 0.2f,
        0.6f, -0.5f, 0.4f, 0.0f,
        -0.3f, 0.8f, -0.6f, 0.1f,
    };
    const std::vector<float> y_data = {
        1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f,
    };
    ag::Variable x(make_tensor(x_data, ag::Shape{4, 4}));
    ag::Tensor y_t = make_tensor(y_data, ag::Shape{4, 2});

    float initial_loss = read_scalar(
        ag::mse_loss(net.forward(x), y_t).value());
    CHECK(std::isfinite(initial_loss));
    CHECK(initial_loss > 0.f);

    std::vector<ag::Variable> all;
    for (auto& v : net.parameters()) all.push_back(v);
    ag::optim::SGD sgd(all, 0.05f);

    constexpr int kSteps = 200;
    float last = initial_loss;
    for (int i = 0; i < kSteps; ++i) {
        sgd.zero_grad();
        ag::Variable loss = ag::mse_loss(net.forward(x), y_t);
        if (i == kSteps - 1) last = read_scalar(loss.value());
        loss.backward();
        sgd.step();
    }
    CHECK(std::isfinite(last));
    CHECK(last < initial_loss * 0.95f);
    report("Sequential + ReLU MLP training loss decreases");
}

namespace {
float adam_reference_scalar(float p0,
                            const std::vector<float>& grads,
                            float lr, float b1, float b2, float eps) {
    float p = p0;
    float m = 0.f, v = 0.f;
    for (std::size_t t = 1; t <= grads.size(); ++t) {
        const float g = grads[t - 1];
        m = b1 * m + (1.f - b1) * g;
        v = b2 * v + (1.f - b2) * g * g;
        const float bc1 = 1.f - std::pow(b1, static_cast<float>(t));
        const float bc2 = 1.f - std::pow(b2, static_cast<float>(t));
        const float denom = std::sqrt(v / bc2) + eps;
        p -= lr * (m / bc1) / denom;
    }
    return p;
}
}  // namespace

void test_adam_first_step_exact() {
    ag::Variable p(make_tensor({2.0f}, ag::Shape{}), true);
    p.backward(make_tensor({0.5f}, ag::Shape{}));
    ag::optim::Adam adam({p}, 0.1f);
    adam.step();
    const float expected = adam_reference_scalar(
        2.0f, {0.5f}, 0.1f, 0.9f, 0.999f, 1e-8f);
    CHECK(std::fabs(read_scalar(p.value()) - expected) <= 1e-5f);
    report("optim::Adam first-step exact reference");
}

void test_adam_multi_step_trajectory() {
    const std::vector<float> grads = {0.1f, 0.2f, -0.05f, 0.3f, -0.1f};
    ag::Variable p(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam adam({p}, 0.01f);
    for (float g : grads) {
        adam.zero_grad();
        p.backward(make_tensor({g}, ag::Shape{}));
        adam.step();
    }
    const float expected = adam_reference_scalar(
        1.0f, grads, 0.01f, 0.9f, 0.999f, 1e-8f);
    CHECK(std::fabs(read_scalar(p.value()) - expected) <= 1e-5f);
    report("optim::Adam multi-step trajectory matches reference");
}

void test_adam_state_snapshot_continue() {
    const std::vector<float> warmup = {0.1f, 0.2f, 0.3f};
    const std::vector<float> post   = {0.05f, 0.15f};

    auto replay = [](ag::Variable& p, ag::optim::Adam& opt,
                     const std::vector<float>& grads) {
        for (float g : grads) {
            opt.zero_grad();
            p.backward(make_tensor({g}, ag::Shape{}));
            opt.step();
        }
    };

    ag::Variable p_a(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam opt_a({p_a}, 0.01f);
    replay(p_a, opt_a, warmup);
    auto snap = opt_a.state();
    replay(p_a, opt_a, post);
    float a_final = read_scalar(p_a.value());

    ag::Variable p_c(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam warmup_c({p_c}, 0.01f);
    replay(p_c, warmup_c, warmup);
    ag::optim::Adam opt_c({p_c}, 0.2f, 0.5f, 0.8f, 0.1f);
    opt_c.load_state(snap);
    CHECK(opt_c.learning_rate() == snap.lr);
    CHECK(opt_c.beta1() == snap.beta1);
    CHECK(opt_c.beta2() == snap.beta2);
    CHECK(opt_c.eps() == snap.eps);
    replay(p_c, opt_c, post);
    float c_final = read_scalar(p_c.value());

    CHECK(std::fabs(a_final - c_final) <= 1e-5f);
    report("optim::Adam snapshot -> continue matches reference");
}

void test_adam_state_snapshot_disturb_then_load() {
    const std::vector<float> warmup = {0.1f, 0.2f, 0.3f};
    const std::vector<float> post   = {0.05f, 0.15f};

    auto replay = [](ag::Variable& p, ag::optim::Adam& opt,
                     const std::vector<float>& grads) {
        for (float g : grads) {
            opt.zero_grad();
            p.backward(make_tensor({g}, ag::Shape{}));
            opt.step();
        }
    };

    ag::Variable p_a(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam opt_a({p_a}, 0.01f);
    replay(p_a, opt_a, warmup);
    auto snap = opt_a.state();
    float p_a_at_snap = read_scalar(p_a.value());
    replay(p_a, opt_a, post);
    float a_final = read_scalar(p_a.value());

    // Build a fresh C, replay the same warmup, then disturb it with a
    // different gradient before loading snap. The disturb advances the
    // optimizer state and updates p_c.value; load_state restores the
    // optimizer state but not the parameter value, so we explicitly
    // restore p_c.value to p_a's snap-point value through the public
    // Tensor copy_from_host API before replaying the post grads.
    ag::Variable p_c(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam opt_c({p_c}, 0.01f);
    replay(p_c, opt_c, warmup);
    opt_c.zero_grad();
    p_c.backward(make_tensor({0.7f}, ag::Shape{}));
    opt_c.step();
    opt_c.load_state(snap);
    std::vector<float> restore = {p_a_at_snap};
    ag::Tensor parameter_alias = p_c.value();
    parameter_alias.copy_from_host(restore.data(), 1);
    replay(p_c, opt_c, post);
    float c_final = read_scalar(p_c.value());
    auto c_state = opt_c.state();

    CHECK(std::fabs(a_final - c_final) <= 1e-5f);
    auto a_state = opt_a.state();
    CHECK(a_state.t == c_state.t);
    check_near(read_values(a_state.first_moments[0]),
               read_values(c_state.first_moments[0]));
    check_near(read_values(a_state.second_moments[0]),
               read_values(c_state.second_moments[0]));
    report("optim::Adam disturb-then-load -> continuation matches reference");
}

void test_adam_load_state_validates_before_mutate() {
    ag::Variable p(make_tensor({1.f, 2.f}, ag::Shape{2}), true);
    ag::optim::Adam opt({p}, 0.01f);
    p.backward(make_tensor({0.1f, 0.2f}, ag::Shape{2}));
    opt.step();
    auto pre = opt.state();

    // A snapshot whose metadata fields would each mutate the live
    // optimizer (t, lr, beta1, beta2, eps), but whose moment shape
    // mismatches the parameter. A correct implementation must reject
    // the load without touching any of the metadata fields.
    ag::optim::AdamState bad;
    bad.t = 999;
    bad.lr = 0.5f;
    bad.beta1 = 0.5f;
    bad.beta2 = 0.5f;
    bad.eps = 0.5f;
    bad.first_moments.push_back(make_tensor({0.1f}, ag::Shape{1}));
    bad.second_moments.push_back(make_tensor({0.01f}, ag::Shape{1}));

    CHECK_THROWS_AS(opt.load_state(bad), std::invalid_argument);
    auto post = opt.state();
    CHECK(pre.t == post.t);
    CHECK(pre.lr == post.lr);
    CHECK(pre.beta1 == post.beta1);
    CHECK(pre.beta2 == post.beta2);
    CHECK(pre.eps == post.eps);
    check_near(read_values(pre.first_moments[0]),
               read_values(post.first_moments[0]));
    check_near(read_values(pre.second_moments[0]),
               read_values(post.second_moments[0]));
    report("optim::Adam load_state validates-before-mutate");
}

void test_adam_load_state_invalid_cases() {
    ag::Variable p(make_tensor({1.f, 2.f}, ag::Shape{2}), true);
    ag::optim::Adam opt({p}, 0.01f);
    p.backward(make_tensor({0.1f, 0.2f}, ag::Shape{2}));
    opt.step();
    auto p_before = read_values(p.value());
    auto pre = opt.state();

    auto build = [&](int64_t t, float lr, float b1, float b2, float eps,
                     std::vector<ag::Tensor> m,
                     std::vector<ag::Tensor> v) {
        ag::optim::AdamState s;
        s.t = t; s.lr = lr; s.beta1 = b1; s.beta2 = b2; s.eps = eps;
        s.first_moments = std::move(m);
        s.second_moments = std::move(v);
        return s;
    };
    auto m_ok = std::vector<ag::Tensor>{
        make_tensor({0.1f, 0.2f}, ag::Shape{2})};
    auto v_ok = std::vector<ag::Tensor>{
        make_tensor({0.01f, 0.04f}, ag::Shape{2})};

    // Negative step count.
    CHECK_THROWS_AS(opt.load_state(build(-1, 0.01f, 0.9f, 0.999f, 1e-8f,
                                          m_ok, v_ok)),
                    std::invalid_argument);
    // Wrong count.
    CHECK_THROWS_AS(opt.load_state(build(1, 0.01f, 0.9f, 0.999f, 1e-8f,
                                         std::vector<ag::Tensor>{},
                                         std::vector<ag::Tensor>{})),
                    std::invalid_argument);
    // Wrong shape.
    auto m_bad = std::vector<ag::Tensor>{make_tensor({0.1f}, ag::Shape{1})};
    CHECK_THROWS_AS(opt.load_state(build(1, 0.01f, 0.9f, 0.999f, 1e-8f,
                                         m_bad, v_ok)),
                    std::invalid_argument);
    // Invalid hyperparameters.
    CHECK_THROWS_AS(opt.load_state(build(1, -0.01f, 0.9f, 0.999f, 1e-8f,
                                         m_ok, v_ok)),
                    std::invalid_argument);
    CHECK_THROWS_AS(opt.load_state(build(1, 0.01f, 1.0f, 0.999f, 1e-8f,
                                         m_ok, v_ok)),
                    std::invalid_argument);
    CHECK_THROWS_AS(opt.load_state(build(1, 0.01f, 0.9f, 0.999f, 0.f,
                                         m_ok, v_ok)),
                    std::invalid_argument);

    auto post = opt.state();
    CHECK(pre.t == post.t);
    CHECK(pre.lr == post.lr);
    check_near(read_values(p.value()), p_before);
    report("optim::Adam load_state rejects invalid snapshots, state unchanged");
}

void test_adam_state_deep_copy_isolation() {
    ag::Variable p(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam opt({p}, 0.01f);
    p.backward(make_tensor({0.1f}, ag::Shape{}));
    opt.step();
    auto snap = opt.state();
    auto m_before = read_values(snap.first_moments[0]);

    // Continue the live optimizer with a different gradient.
    opt.zero_grad();
    p.backward(make_tensor({0.5f}, ag::Shape{}));
    opt.step();
    auto m_after_live = read_values(opt.state().first_moments[0]);

    // snap.first_moments must be untouched by the live update because
    // state() returns deep clones.
    auto m_snap_now = read_values(snap.first_moments[0]);
    check_near(m_snap_now, m_before);
    // And the live moments really did move.
    bool moved = false;
    for (std::size_t i = 0; i < m_after_live.size(); ++i) {
        if (std::fabs(m_after_live[i] - m_before[i]) > 1e-7f) {
            moved = true;
            break;
        }
    }
    CHECK(moved);
    report("optim::Adam state() returns deep-cloned moments");
}

void test_adam_zero_grad_preserves_state() {
    ag::Variable p(make_tensor({1.0f}, ag::Shape{}), true);
    ag::optim::Adam opt({p}, 0.01f);
    p.backward(make_tensor({0.1f}, ag::Shape{}));
    opt.step();
    auto snap1 = opt.state();
    int64_t t_before = snap1.t;
    auto m_before = read_values(snap1.first_moments[0]);
    auto v_before = read_values(snap1.second_moments[0]);

    opt.zero_grad();
    CHECK(!p.has_grad());

    auto snap2 = opt.state();
    CHECK(snap2.t == t_before);
    check_near(read_values(snap2.first_moments[0]), m_before);
    check_near(read_values(snap2.second_moments[0]), v_before);
    report("optim::Adam::zero_grad preserves moments and step count");
}

void test_adam_invalid_constructor() {
    ag::Variable p(make_tensor({1.f}, ag::Shape{}), true);
    CHECK_THROWS_AS(ag::optim::Adam({p}, -0.01f), std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, std::nanf("")), std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p},
                                     std::numeric_limits<float>::infinity()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, -0.1f), std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 0.9f, 1.0f),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 0.9f, -0.1f),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 0.9f, 0.999f, 0.f),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 0.9f, 0.999f, -1e-8f),
                    std::invalid_argument);
    CHECK_THROWS_AS(ag::optim::Adam({p}, 0.01f, 0.9f, 0.999f,
                                    std::numeric_limits<float>::infinity()),
                    std::invalid_argument);
    report("optim::Adam rejects invalid hyperparameters");
}

void test_adam_empty_params() {
    ag::optim::Adam opt({}, 0.01f);
    opt.step();
    opt.zero_grad();
    auto snap = opt.state();
    CHECK(snap.t == 0);
    CHECK(snap.first_moments.empty());
    CHECK(snap.second_moments.empty());
    report("optim::Adam accepts empty parameter list");
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

    test_relu_module();
    test_sequential_forward();
    test_sequential_named_parameters_depth_first();
    test_sequential_zero_grad_recurses();
    test_register_module_validation();
    test_recursive_parameters_through_module();
    test_sequential_mlp_loss_decrease();
    test_adam_first_step_exact();
    test_adam_multi_step_trajectory();
    test_adam_state_snapshot_continue();
    test_adam_state_snapshot_disturb_then_load();
    test_adam_load_state_validates_before_mutate();
    test_adam_load_state_invalid_cases();
    test_adam_state_deep_copy_isolation();
    test_adam_zero_grad_preserves_state();
    test_adam_invalid_constructor();
    test_adam_empty_params();

    std::printf("\nALL OOP TRAINING TESTS PASSED (%d)\n", passed);
    return 0;
}
