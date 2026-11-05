// test_characterization.cpp — Phase 0 characterization tests.
//
// These tests pin observable behavior that the architecture refactor must
// preserve. They are intentionally narrow: each test exercises one
// contract gap that the existing core/NN/conv/extensions/diffusion/smoke
// suites do not already cover. They do NOT duplicate grad_check coverage
// (see test_core, test_extensions, test_fft, test_conv).
//
// Coverage mapping lives in docs/PHASE0_BEHAVIOR_CONTRACT.md.

#include "autograd/core/module.h"
#include "autograd/core/optim.h"
#include "grad_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace ag;
using namespace ag_test;

namespace {

int passed = 0;
int failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s at %s:%d\n", \
                     #cond, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (!std::isfinite(_a) || !std::isfinite(_b) || std::fabs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "FAIL: %s (%g) vs %s (%g) at %s:%d\n", \
                     #a, _a, #b, _b, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_EQ_TENSOR(a, b) do { \
    auto _A = (a); auto _B = (b); \
    if (_A.shape() != _B.shape() || \
        max_diff(_A, _B) > 1e-5f) { \
        std::fprintf(stderr, "FAIL: tensor mismatch at %s:%d\n", __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

// ── §2 Variable: fresh-seed backward accumulation and node-local zero_grad ─────────────────────

void test_backward_accumulates_without_zero() {
    auto x = Variable(filled(Shape{2, 2}, 3.f), true);
    auto y = sum(x);
    y.backward();
    CHECK_NEAR(at(x.grad(), 0, 0), 1.f, 1e-5f);
    y.backward();
    CHECK_NEAR(at(x.grad(), 0, 0), 2.f, 1e-5f);
    CHECK_NEAR(at(x.grad(), 1, 1), 2.f, 1e-5f);
    y.zero_grad();
    CHECK(!y.has_grad());
    CHECK_NEAR(at(x.grad(), 0, 0), 2.f, 1e-7f); // node-local, not recursive
    x.zero_grad();
    CHECK(!x.has_grad());
    y.backward();
    CHECK_NEAR(at(x.grad(), 0, 0), 1.f, 1e-5f);
    report("backward accumulates fresh seeds; explicitly cleared leaves reset");
}

void test_var_zero_grad_reaches_all_reachable() {
    auto x = Variable(random_tensor(Shape{2, 3}), true);
    auto h = relu(x);
    auto y = sum(h);
    y.backward();
    CHECK(max_abs(x.grad()) > 0.f);
    CHECK(max_abs(h.grad()) > 0.f);
    CHECK_NEAR(at(y.grad()), 1.f, 1e-7f);
    const Tensor x_before = x.grad().clone(), h_before = h.grad().clone();
    y.zero_grad();
    CHECK(!y.has_grad());
    CHECK_EQ_TENSOR(x.grad(), x_before);
    CHECK_EQ_TENSOR(h.grad(), h_before);
    h.zero_grad();
    CHECK(!h.has_grad());
    CHECK_EQ_TENSOR(x.grad(), x_before);
    x.zero_grad();
    CHECK(!x.has_grad());
    report("Variable::zero_grad clears only its receiver");
}

void test_module_zero_grad_isolates_intermediates() {
    // Module::zero_grad zeros only the registered leaf parameters.
    // Intermediate activation Variables retain their gradients untouched
    // (they will be re-created by the next forward pass anyway).
    auto net = std::make_shared<nn::Sequential>();
    net->add(std::make_shared<nn::Linear>(3, 2));

    auto x = Variable(random_tensor(Shape{4, 3}), true);
    auto y = net->forward(x);
    auto loss = sum(mul(y, y));
    loss.backward();

    // Sanity: every parameter got a non-zero gradient.
    for (auto& p : net->parameters()) {
        CHECK(max_abs(p.grad()) > 0.f);
    }
    // Snapshot the activation's grad before module::zero_grad.
    Tensor activation_grad_before = y.grad().clone();
    Tensor input_grad_before = x.grad().clone();

    net->zero_grad();

    for (auto& p : net->parameters()) {
        CHECK(!p.has_grad());
    }
    // Activation's grad must be untouched by Module::zero_grad.
    CHECK_EQ_TENSOR(y.grad(), activation_grad_before);
    CHECK_EQ_TENSOR(x.grad(), input_grad_before);
    report("Module::zero_grad zeros only registered parameter leaves");
}

void test_optimizer_zero_grad_semantics() {
    // optim::SGD::zero_grad zeros only `grad` on every parameter; it must not
    // touch `data`. optim::Adam::zero_grad zeros only `grad`; the per-parameter
    // first/second moments (`m`, `v`) and the step counter `t` must be
    // preserved in the public AdamState snapshot.
    auto p1 = Variable(random_tensor(Shape{2, 2}), true);
    auto p2 = Variable(random_tensor(Shape{2, 2}), true);
    auto loss = sum(mul(p1, p2));
    loss.backward();
    for (auto& p : {p1, p2}) CHECK(max_abs(p.grad()) > 0.f);

    Tensor p1_data_before = p1.value().clone();
    Tensor p2_data_before = p2.value().clone();
    optim::SGD sgd({p1, p2}, 0.01f);
    sgd.zero_grad();
    for (auto& p : {p1, p2}) {
        CHECK(!p.has_grad());
    }
    CHECK_EQ_TENSOR(p1.value(), p1_data_before);
    CHECK_EQ_TENSOR(p2.value(), p2_data_before);

    // optim::Adam: snapshot the public optimizer state, run a step to populate
    // m/v/t, call zero_grad(), and verify m/v/t are bit-for-bit preserved
    // while grad is cleared.
    auto q = Variable(filled(Shape{1, 1}, 1.f), true);
    optim::Adam adam({q}, 0.1f);
    seed_grad(q, filled(Shape{1, 1}, 0.5f));
    adam.step();                                // t=1, populates m[0], v[0]
    CHECK(adam.step_count() == 1);
    Tensor m_snapshot = adam.state().first_moments[0];
    Tensor v_snapshot = adam.state().second_moments[0];
    int t_snapshot = adam.step_count();

    adam.zero_grad();
    CHECK_EQ_TENSOR(adam.state().first_moments[0],  m_snapshot);      // m preserved
    CHECK_EQ_TENSOR(adam.state().second_moments[0],  v_snapshot);      // v preserved
    CHECK(adam.step_count() == t_snapshot);               // t preserved
    CHECK(!q.has_grad());

    // Discriminating step with a different gradient: m/v must reflect
    // carry-over from the previous step (β1·m + (1-β1)·g_new), not a
    // reset state (which would be just (1-β1)·g_new). The expected
    // values pin the recurrence exactly.
    seed_grad(q, filled(Shape{1, 1}, -0.3f));
    adam.step();                                // t=2
    // m_new = 0.9·m_prev + 0.1·(-0.3) = 0.9·0.05 + (-0.03) = 0.015
    CHECK_NEAR(at(adam.state().first_moments[0], 0, 0), 0.015f, 1e-5f);
    // v_new = 0.999·v_prev + 0.001·(-0.3)² = 0.999·0.00025 + 0.00009
    //       = 0.00024975 + 0.00009 = 0.00033975
    CHECK_NEAR(at(adam.state().second_moments[0], 0, 0), 0.00033975f, 1e-7f);
    CHECK(adam.step_count() == 2);

    report("Optimizer::zero_grad zeros only grad; optim::Adam m/v/t are preserved");
}

// ── §4 Module parameter order ─────────────────────────────────────────────

bool observable_alias(Variable a, Variable b) {
    if (a.value().shape() != b.value().shape()) return false;
    const float original = at(a.value());
    put(a.value(), 0, original + 3.f);
    const bool shared_value = at(b.value()) == original + 3.f;
    put(a.value(), 0, original);
    a.zero_grad(); b.zero_grad();
    sum(a).backward();
    const bool shared_node = b.has_grad() && max_diff(a.grad(), b.grad()) == 0.f;
    b.zero_grad();
    return shared_value && shared_node && !a.has_grad();
}

void test_module_parameter_order() {
    // nn::Linear: [W, b].
    nn::Linear lin(3, 5);
    auto lp = lin.parameters();
    CHECK(lp.size() == 2);
    CHECK(observable_alias(lp[0], lin.weight()));
    CHECK(observable_alias(lp[1], lin.bias()));
    CHECK(lp[0].value().shape()[0] == 3 && lp[0].value().shape()[1] == 5);
    CHECK(lp[1].value().shape()[0] == 1 && lp[1].value().shape()[1] == 5);

    // Same ordering across repeated calls (identity, not just equality).
    CHECK(lin.named_parameters()[0].name == "weight");
    CHECK(lin.named_parameters()[1].name == "bias");
    auto lp2 = lin.parameters();
    CHECK(lp2.size() == lp.size());
    for (size_t i = 0; i < lp.size(); ++i) {
        CHECK(observable_alias(lp[i], lp2[i]));
    }

    // nn::Sequential: children in add() insertion order, each with its own
    // per-child parameter order. Public fields like `layers` are NOT
    // pinned (Phase 11 removes them) — only the public `add()` API and
    // the resulting `parameters()` traversal are.
    auto net = std::make_shared<nn::Sequential>();
    auto lin_a = std::make_shared<nn::Linear>(4, 6);
    auto relu  = std::make_shared<nn::ReLU>();
    auto lin_b = std::make_shared<nn::Linear>(6, 2);

    net->add(lin_a);
    net->add(relu);
    net->add(lin_b);

    auto params = net->parameters();
    CHECK(params.size() == 4);                // nn::ReLU has none.
    CHECK(observable_alias(params[0], lin_a->weight()));
    CHECK(observable_alias(params[1], lin_a->bias()));
    CHECK(observable_alias(params[2], lin_b->weight()));
    CHECK(observable_alias(params[3], lin_b->bias()));

    // Repeated traversal returns the same parameter sequence (identity).
    const auto names = net->named_parameters();
    CHECK(names[0].name == "0.weight" && names[1].name == "0.bias");
    CHECK(names[2].name == "2.weight" && names[3].name == "2.bias");
    auto params2 = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        CHECK(observable_alias(params[i], params2[i]));
    }

    report("Module::parameters() order is deterministic and stable across calls");
}

void test_module_parameter_order_with_conv() {
    // nn::Conv2d params are also [W, b] in that order.
    auto conv = std::make_shared<nn::Conv2d>(2, 4, 3, 3, 1, 0);
    auto lin  = std::make_shared<nn::Linear>(16, 3);
    auto net  = std::make_shared<nn::Sequential>();
    net->add(conv);
    net->add(lin);

    auto params = net->parameters();
    CHECK(params.size() == 4);
    CHECK(observable_alias(params[0], conv->weight()));
    CHECK(observable_alias(params[1], conv->bias()));
    CHECK(observable_alias(params[2], lin->weight()));
    CHECK(observable_alias(params[3], lin->bias()));

    // Check shape matchers so the order is clearly tied to data shapes.
    CHECK(params[0].value().shape()[0] == 4);   // out_ch
    CHECK((params[0].value().shape() == Shape{4, 2, 3, 3}));
    CHECK((params[1].value().shape() == Shape{4}));
    report("Conv2d parameter order and node identity: weight (OC,C,kH,kW), bias (OC)");
}

// ── §5 Optimizer: exact trajectories and state isolation ──────────────────

void test_sgd_trajectory() {
    auto p = Variable(tensor(Shape{2, 2}, {1.f, 2.f,
        3.f, 4.f}), true);
    optim::SGD sgd({p}, 0.1f);

    Tensor g1 = tensor(Shape{2, 2}, {0.5f, -1.0f,
        2.0f, -0.5f});

    // Step 1: p <- p - lr * g1.
    seed_grad(p, g1);
    sgd.step();
    CHECK_NEAR(at(p.value(), 0, 0), 1.f - 0.05f, 1e-5f);
    CHECK_NEAR(at(p.value(), 0, 1), 2.f + 0.1f,  1e-5f);
    CHECK_NEAR(at(p.value(), 1, 0), 3.f - 0.2f,  1e-5f);
    CHECK_NEAR(at(p.value(), 1, 1), 4.f + 0.05f, 1e-5f);

    // Step 2 (no zero_grad, same grad): another full step.
    seed_grad(p, g1);
    sgd.step();
    CHECK_NEAR(at(p.value(), 0, 0), 1.f - 0.10f, 1e-5f);
    CHECK_NEAR(at(p.value(), 0, 1), 2.f + 0.20f, 1e-5f);

    // Step 3 with a different grad (only the (1,0) element nonzero).
    Tensor g2 = tensor(Shape{2, 2}, {0.f, 0.f,
        1.f, 0.f});
    sgd.zero_grad();
    seed_grad(p, g2);
    sgd.step();
    // After steps 1+2 with grad1 = [[0.5,-1.0],[2.0,-0.5]], then step 3
    // with g2 = [[0,0],[1,0]]: (1,0) gets one more -0.1, the others
    // settle at their step-2 values.
    CHECK_NEAR(at(p.value(), 0, 0), 0.9f,  1e-5f);
    CHECK_NEAR(at(p.value(), 0, 1), 2.2f,  1e-5f);
    CHECK_NEAR(at(p.value(), 1, 0), 2.5f,  1e-5f);
    CHECK_NEAR(at(p.value(), 1, 1), 4.10f, 1e-5f);

    report("optim::SGD::step() applies p -= lr * grad exactly across multiple steps");
}

void test_adam_trajectory() {
    // optim::Adam trajectory with varying signed/magnitude gradients per step.
    //
    // A reference optim::Adam recurrence (different code structure than the
    // library: out-of-place per-element temporaries vs. in-place
    // Eigen expressions, computed in scalars vs. matrices) is run in
    // parallel. After every step the library's AdamState moments, counter and
    // the parameter are compared element-wise against the reference.
    //
    // Varying gradients are essential: a constant-gradient trajectory
    // cancels the bias-correction terms and yields a degenerate test
    // (update ≈ lr·sign(g) per step regardless of β1, β2). With varying
    // gradients, the bias correction actually matters and small
    // implementation drift in m̂/v̂ would surface as a measurable
    // parameter mismatch.
    auto p = Variable(tensor(Shape{1, 3}, {2.f, -1.f, 0.5f}), true);
    optim::Adam adam({p}, /*lr=*/0.05f);

    std::vector<Tensor> grads = {
        tensor(Shape{1, 3}, {0.4f, -0.8f,  0.2f}),
        tensor(Shape{1, 3}, {-0.5f,  0.3f, -0.1f}),
        tensor(Shape{1, 3}, {0.2f,  0.6f, -0.3f}),
    };

    // Independent reference recurrence (scalar temporaries, out-of-place).
    auto p_ref = host(p.value());
    std::vector<float> m_ref(3, 0.f);
    std::vector<float> v_ref(3, 0.f);
    int  t_ref = 0;
    const float lr  = 0.05f;
    const float b1  = 0.9f;
    const float b2  = 0.999f;
    const float eps = 1e-8f;

    for (const auto& g : grads) {
        // Library step.
        seed_grad(p, g);
        adam.step();

        // Reference step: independent code path, same optim::Adam math.
        t_ref += 1;
        float bc1 = 1.f - std::pow(b1, static_cast<float>(t_ref));
        float bc2 = 1.f - std::pow(b2, static_cast<float>(t_ref));
        for (int j = 0; j < 3; ++j) {
            float gj = at(g, 0, j);
            m_ref[j] = b1 * m_ref[j] + (1.f - b1) * gj;
            v_ref[j] = b2 * v_ref[j] + (1.f - b2) * gj * gj;
            float m_hat = m_ref[j] / bc1;
            float v_hat = v_ref[j] / bc2;
            float denom = std::sqrt(v_hat) + eps;
            p_ref[j] -= lr * m_hat / denom;
        }

        // Per-element comparison: m, v, t, parameter.
        for (int j = 0; j < 3; ++j) {
            CHECK_NEAR(at(adam.state().first_moments[0], 0, j), m_ref[j], 1e-5f);
            CHECK_NEAR(at(adam.state().second_moments[0], 0, j), v_ref[j], 1e-5f);
            CHECK_NEAR(at(p.value(), 0, j),    p_ref[j], 1e-5f);
        }
        CHECK(adam.step_count() == t_ref);
    }
    report("optim::Adam trajectory with varying signed grads matches reference m/v/t/p");
}

void test_optimizer_state_isolation() {
    // Two optim::Adam instances on disjoint params never see each other's state.
    auto a = Variable(filled(Shape{1, 1}, 1.f), true);
    auto b = Variable(filled(Shape{1, 1}, 1.f), true);
    optim::Adam adam_a({a}, 0.1f);
    optim::Adam adam_b({b}, 0.1f);
    seed_grad(a, filled(Shape{1, 1}, 1.f));
    adam_a.step();
    // b's data and grad must be untouched.
    CHECK_NEAR(at(b.value(), 0, 0), 1.f, 1e-6f);
    CHECK(!b.has_grad());
    CHECK(adam_b.state().t == 0);
    CHECK(max_abs(adam_b.state().first_moments[0]) == 0.f);
    CHECK(max_abs(adam_b.state().second_moments[0]) == 0.f);
    CHECK_NEAR(at(a.value(), 0, 0), 0.9f, 1e-3f);

    // Rebuilding Adam on the same parameter starts independent zero moments/counter.
    auto c = Variable(filled(Shape{1, 1}, 1.f), true);
    {
        optim::Adam adam_c_old({c}, 0.1f);
        seed_grad(c, filled(Shape{1, 1}, 1.f));
        adam_c_old.step();
    }  // adam_c_old destroyed; c.value() is ~0.9 now.
    // New optim::Adam on the same c: m, v, t start at zero again.
    optim::Adam adam_c_new({c}, 0.1f);
    CHECK(adam_c_new.state().t == 0);
    CHECK(max_abs(adam_c_new.state().first_moments[0]) == 0.f);
    CHECK(max_abs(adam_c_new.state().second_moments[0]) == 0.f);
    seed_grad(c, filled(Shape{1, 1}, 1.f));
    adam_c_new.step();
    // Step #1 again (t=1) on a fresh optim::Adam → same first-step update as the
    // single-step test in test_nn::test_adam_step.
    CHECK_NEAR(at(c.value(), 0, 0), 0.9f - 0.1f, 1e-3f);
    report("Optimizer state per-instance; rebuilding resets m/v/t");
}

// ── §3 Conv: repeated-backward accumulation ────────────────────────────────

void test_conv_repeated_backward_accumulates() {
    nn::Conv2d conv(1, 1, 2, 2, 1, 0);
    auto w = conv.weight(), b = conv.bias();
    Tensor weights = w.value(), biases = b.value();
    auto wd = host(filled(weights.shape(), .25f)); weights.copy_from_host(wd.data(), wd.size());
    auto bd = host(Tensor::zeros(biases.shape())); biases.copy_from_host(bd.data(), bd.size());
    auto x = Variable(Tensor::ones(Shape{1, 1, 4, 4}), true);
    auto y = conv.forward(x);
    auto loss = sum(y);
    loss.backward();
    const Tensor w_first = w.grad().clone(), x_first = x.grad().clone();
    CHECK(max_abs(w_first) > 0.f);
    CHECK(max_abs(x_first) > 0.f);
    loss.backward();
    auto twice_w = host(w_first), twice_x = host(x_first);
    for (auto& v : twice_w) v *= 2.f;
    for (auto& v : twice_x) v *= 2.f;
    CHECK_NEAR(max_diff(w.grad(), tensor(w_first.shape(), twice_w)), 0.f, 1e-4f);
    CHECK_NEAR(max_diff(x.grad(), tensor(x_first.shape(), twice_x)), 0.f, 1e-4f);
    loss.zero_grad(); w.zero_grad(); b.zero_grad(); x.zero_grad();
    CHECK(!w.has_grad()); CHECK(!b.has_grad()); CHECK(!x.has_grad());
    loss.backward();
    CHECK_EQ_TENSOR(w.grad(), w_first);
    CHECK_EQ_TENSOR(x.grad(), x_first);
    loss.backward();
    const auto input_before = x.grad().clone();
    conv.zero_grad();
    CHECK(!w.has_grad()); CHECK(!b.has_grad());
    CHECK(max_abs(x.grad()) > 0.f);
    CHECK_EQ_TENSOR(x.grad(), input_before);
    report("Conv2d fresh-seed 2x accumulation, explicit leaves and parameter-only clearing");
}

// ── §2/§4 backward leaves data untouched ──────────────────────────────────

void test_backward_does_not_mutate_data() {
    auto p = Variable(tensor(Shape{1, 1}, {7.f}), true);
    Tensor before = p.value().clone();
    auto y = sum(mul(p, p));
    y.backward();
    p.zero_grad();
    // Data must be unchanged after backward + zero_grad.
    CHECK_EQ_TENSOR(p.value(), before);

    optim::SGD sgd({p}, 0.1f);
    seed_grad(p, filled(Shape{1, 1}, 1.f));
    sgd.step();
    // optim::SGD step mutates data; that's expected. The contract is that
    // backward() and zero_grad() do NOT.
    CHECK(std::fabs(at(p.value(), 0, 0) - 7.f) > 1e-5f);

    report("backward() and zero_grad() do not mutate parameter data");
}

}  // namespace

int main() {
    std::printf("=== Phase 0 characterization ===\n");
    std::printf("[grad accumulation / zero_grad]\n");
    test_backward_accumulates_without_zero();
    test_var_zero_grad_reaches_all_reachable();
    test_module_zero_grad_isolates_intermediates();
    test_optimizer_zero_grad_semantics();
    test_backward_does_not_mutate_data();

    std::printf("[module parameter order]\n");
    test_module_parameter_order();
    test_module_parameter_order_with_conv();

    std::printf("[optimizer trajectories]\n");
    test_sgd_trajectory();
    test_adam_trajectory();
    test_optimizer_state_isolation();

    std::printf("[conv repeated backward]\n");
    test_conv_repeated_backward_accumulates();

    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return (failed == 0) ? 0 : 1;
}