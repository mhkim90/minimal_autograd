// test_characterization.cpp — Phase 0 characterization tests.
//
// These tests pin observable behavior that the architecture refactor must
// preserve. They are intentionally narrow: each test exercises one
// contract gap that the existing core/NN/conv/extensions/diffusion/smoke
// suites do not already cover. They do NOT duplicate grad_check coverage
// (see test_core, test_extensions, test_fft, test_conv).
//
// Coverage mapping lives in docs/PHASE0_BEHAVIOR_CONTRACT.md.

#include "autograd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace ag;

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
    if (std::fabs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "FAIL: %s (%g) vs %s (%g) at %s:%d\n", \
                     #a, _a, #b, _b, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_EQ_MAT(a, b) do { \
    auto _A = (a); auto _B = (b); \
    if (_A.rows() != _B.rows() || _A.cols() != _B.cols() || \
        (_A - _B).cwiseAbs().maxCoeff() > 1e-5f) { \
        std::fprintf(stderr, "FAIL: mat mismatch at %s:%d\n", __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

void report(const char* name) {
    std::printf("  [ok] %s\n", name);
    ++passed;
}

// ── §2 Variable: backward accumulation, Var::zero_grad ─────────────────────

void test_backward_accumulates_without_zero() {
    // Contract: backward() called repeatedly on the same scalar loss
    // accumulates gradients on every visited node, including intermediate
    // activations. This is the natural behavior of the current
    // apply<Fn> + topo traversal design (no implicit intermediate-grad
    // reset).
    //
    // Use a graph with no intermediates between loss and the leaf
    // (`sum(x)`) so the leaf-gradient growth is exactly 2x per pass.
    auto x = Var::make(Mat::Constant(2, 2, 3.f));
    auto y = sum(x);                  // loss = sum(x), 1x1
    y->backward();
    // First backward: dy/dx = 1.
    CHECK_NEAR(x->grad(0, 0), 1.f, 1e-5f);

    y->backward();                    // No zero_grad in between.
    // Second backward: x.grad doubles.
    CHECK_NEAR(x->grad(0, 0), 2.f, 1e-5f);
    CHECK_NEAR(x->grad(1, 1), 2.f, 1e-5f);

    // Var::zero_grad() on the loss zeros the whole reachable graph,
    // including intermediate activations.
    y->zero_grad();
    CHECK_NEAR(x->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);

    // After zero_grad, a fresh backward reproduces the first pass.
    y->backward();
    CHECK_NEAR(x->grad(0, 0), 1.f, 1e-5f);
    report("backward() accumulates grad; zero_grad() resets the graph");
}

void test_var_zero_grad_reaches_all_reachable() {
    // Contract: Var::zero_grad() on a node zeros every node reachable
    // through its parents, including intermediate activations and the
    // loss node itself (not only the leaf parameters).
    auto x = Var::make(Mat::Random(2, 3));
    auto h = relu(x);                    // intermediate activation
    auto y = sum(h);                      // scalar loss

    y->backward();
    // Sanity: every reachable node carries a non-zero gradient before
    // zero_grad.
    CHECK(x->grad.cwiseAbs().maxCoeff() > 0.f);
    CHECK(h->grad.cwiseAbs().maxCoeff() > 0.f);
    CHECK_NEAR(y->grad(0, 0), 1.f, 1e-7f);   // set by backward()

    y->zero_grad();

    // All reachable nodes — leaf, intermediate, and loss — are zeroed.
    CHECK_NEAR(x->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    CHECK_NEAR(h->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    CHECK_NEAR(y->grad(0, 0),                0.f, 1e-7f);

    report("Var::zero_grad() zeros every reachable node (leaf + intermediate + loss)");
}

void test_module_zero_grad_isolates_intermediates() {
    // Module::zero_grad zeros only the registered leaf parameters.
    // Intermediate activation Vars retain their grad fields untouched
    // (they will be re-created by the next forward pass anyway).
    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(3, 2));

    auto x = Var::make(Mat::Random(4, 3));
    auto y = net->forward(x);
    auto loss = sum(mul(y, y));
    loss->backward();

    // Sanity: every parameter got a non-zero gradient.
    for (auto& p : net->parameters()) {
        CHECK(p->grad.cwiseAbs().maxCoeff() > 0.f);
    }
    // Snapshot the activation's grad before module::zero_grad.
    Mat activation_grad_before = y->grad;

    net->zero_grad();

    for (auto& p : net->parameters()) {
        CHECK_NEAR(p->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    }
    // Activation's grad must be untouched by Module::zero_grad.
    CHECK_EQ_MAT(y->grad, activation_grad_before);
    report("Module::zero_grad zeros only registered parameter leaves");
}

void test_optimizer_zero_grad_semantics() {
    // SGD::zero_grad zeros only `grad` on every parameter; it must not
    // touch `data`. Adam::zero_grad zeros only `grad`; the per-parameter
    // first/second moments (`m`, `v`) and the step counter `t` must be
    // preserved. Phase 0 inspects these public fields directly; Phase 6
    // replaces them with explicit `AdamState`.
    auto p1 = Var::make(Mat::Random(2, 2));
    auto p2 = Var::make(Mat::Random(2, 2));
    auto loss = sum(mul(p1, p2));
    loss->backward();
    for (auto& p : {p1, p2}) CHECK(p->grad.cwiseAbs().maxCoeff() > 0.f);

    Mat p1_data_before = p1->data;
    Mat p2_data_before = p2->data;
    SGD sgd({p1, p2}, 0.01f);
    sgd.zero_grad();
    for (auto& p : {p1, p2}) {
        CHECK_NEAR(p->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    }
    CHECK_EQ_MAT(p1->data, p1_data_before);
    CHECK_EQ_MAT(p2->data, p2_data_before);

    // Adam: snapshot the public optimizer state, run a step to populate
    // m/v/t, call zero_grad(), and verify m/v/t are bit-for-bit preserved
    // while grad is cleared.
    auto q = Var::make(Mat::Constant(1, 1, 1.f));
    Adam adam({q}, 0.1f);
    q->grad = Mat::Constant(1, 1, 0.5f);
    adam.step();                                // t=1, populates m[0], v[0]
    CHECK(adam.t == 1);
    Mat m_snapshot = adam.m[0];
    Mat v_snapshot = adam.v[0];
    int t_snapshot = adam.t;

    adam.zero_grad();
    CHECK_EQ_MAT(adam.m[0],  m_snapshot);      // m preserved
    CHECK_EQ_MAT(adam.v[0],  v_snapshot);      // v preserved
    CHECK(adam.t == t_snapshot);               // t preserved
    CHECK_NEAR(q->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);

    // Discriminating step with a different gradient: m/v must reflect
    // carry-over from the previous step (β1·m + (1-β1)·g_new), not a
    // reset state (which would be just (1-β1)·g_new). The expected
    // values pin the recurrence exactly.
    q->grad = Mat::Constant(1, 1, -0.3f);
    adam.step();                                // t=2
    // m_new = 0.9·m_prev + 0.1·(-0.3) = 0.9·0.05 + (-0.03) = 0.015
    CHECK_NEAR(adam.m[0](0, 0), 0.015f, 1e-5f);
    // v_new = 0.999·v_prev + 0.001·(-0.3)² = 0.999·0.00025 + 0.00009
    //       = 0.00024975 + 0.00009 = 0.00033975
    CHECK_NEAR(adam.v[0](0, 0), 0.00033975f, 1e-7f);
    CHECK(adam.t == 2);

    report("Optimizer::zero_grad zeros only grad; Adam m/v/t are preserved");
}

// ── §4 Module parameter order ─────────────────────────────────────────────

void test_module_parameter_order() {
    // Linear: [W, b].
    Linear lin(3, 5);
    auto lp = lin.parameters();
    CHECK(lp.size() == 2);
    CHECK(lp[0].get() == lin.W.get());
    CHECK(lp[1].get() == lin.b.get());
    CHECK(lp[0]->data.rows() == 3 && lp[0]->data.cols() == 5);
    CHECK(lp[1]->data.rows() == 1 && lp[1]->data.cols() == 5);

    // Same ordering across repeated calls (identity, not just equality).
    auto lp2 = lin.parameters();
    CHECK(lp2.size() == lp.size());
    for (size_t i = 0; i < lp.size(); ++i) {
        CHECK(lp[i].get() == lp2[i].get());
    }

    // Sequential: children in add() insertion order, each with its own
    // per-child parameter order. Public fields like `layers` are NOT
    // pinned (Phase 11 removes them) — only the public `add()` API and
    // the resulting `parameters()` traversal are.
    auto net = std::make_shared<Sequential>();
    auto lin_a = std::make_shared<Linear>(4, 6);
    auto relu  = std::make_shared<ReLUModule>();
    auto lin_b = std::make_shared<Linear>(6, 2);

    net->add(lin_a);
    net->add(relu);
    net->add(lin_b);

    auto params = net->parameters();
    CHECK(params.size() == 4);                // ReLUModule has none.
    CHECK(params[0].get() == lin_a->W.get());
    CHECK(params[1].get() == lin_a->b.get());
    CHECK(params[2].get() == lin_b->W.get());
    CHECK(params[3].get() == lin_b->b.get());

    // Repeated traversal returns the same parameter sequence (identity).
    auto params2 = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        CHECK(params[i].get() == params2[i].get());
    }

    report("Module::parameters() order is deterministic and stable across calls");
}

void test_module_parameter_order_with_conv() {
    // Conv2d params are also [W, b] in that order.
    auto conv = std::make_shared<Conv2d>(2, 4, 3, 3, 1, 0);
    auto lin  = std::make_shared<Linear>(16, 3);
    auto net  = std::make_shared<Sequential>();
    net->add(conv);
    net->add(lin);

    auto params = net->parameters();
    CHECK(params.size() == 4);
    CHECK(params[0].get() == conv->W.get());
    CHECK(params[1].get() == conv->b.get());
    CHECK(params[2].get() == lin->W.get());
    CHECK(params[3].get() == lin->b.get());

    // Check shape matchers so the order is clearly tied to data shapes.
    CHECK(params[0]->data.rows() == 4);   // out_ch
    CHECK(params[0]->data.cols() == 2 * 3 * 3);
    CHECK(params[1]->data.rows() == 1);
    CHECK(params[1]->data.cols() == 4);
    report("Conv2d module params: [W (out_ch, in_ch*kH*kW), b (1, out_ch)]");
}

// ── §5 Optimizer: exact trajectories and state isolation ──────────────────

void test_sgd_trajectory() {
    auto p = Var::make((Mat(2, 2) <<
        1.f, 2.f,
        3.f, 4.f).finished());
    SGD sgd({p}, 0.1f);

    Mat g1 = (Mat(2, 2) <<
        0.5f, -1.0f,
        2.0f, -0.5f).finished();

    // Step 1: p <- p - lr * g1.
    p->grad = g1;
    sgd.step();
    CHECK_NEAR(p->data(0, 0), 1.f - 0.05f, 1e-5f);
    CHECK_NEAR(p->data(0, 1), 2.f + 0.1f,  1e-5f);
    CHECK_NEAR(p->data(1, 0), 3.f - 0.2f,  1e-5f);
    CHECK_NEAR(p->data(1, 1), 4.f + 0.05f, 1e-5f);

    // Step 2 (no zero_grad, same grad): another full step.
    p->grad = g1;
    sgd.step();
    CHECK_NEAR(p->data(0, 0), 1.f - 0.10f, 1e-5f);
    CHECK_NEAR(p->data(0, 1), 2.f + 0.20f, 1e-5f);

    // Step 3 with a different grad (only the (1,0) element nonzero).
    Mat g2 = (Mat(2, 2) <<
        0.f, 0.f,
        1.f, 0.f).finished();
    sgd.zero_grad();
    p->grad = g2;
    sgd.step();
    // After steps 1+2 with grad1 = [[0.5,-1.0],[2.0,-0.5]], then step 3
    // with g2 = [[0,0],[1,0]]: (1,0) gets one more -0.1, the others
    // settle at their step-2 values.
    CHECK_NEAR(p->data(0, 0), 0.9f,  1e-5f);
    CHECK_NEAR(p->data(0, 1), 2.2f,  1e-5f);
    CHECK_NEAR(p->data(1, 0), 2.5f,  1e-5f);
    CHECK_NEAR(p->data(1, 1), 4.10f, 1e-5f);

    report("SGD::step() applies p -= lr * grad exactly across multiple steps");
}

void test_adam_trajectory() {
    // Adam trajectory with varying signed/magnitude gradients per step.
    //
    // A reference Adam recurrence (different code structure than the
    // library: out-of-place per-element temporaries vs. in-place
    // Eigen expressions, computed in scalars vs. matrices) is run in
    // parallel. After every step the library's public m[0], v[0], t and
    // the parameter are compared element-wise against the reference.
    //
    // Varying gradients are essential: a constant-gradient trajectory
    // cancels the bias-correction terms and yields a degenerate test
    // (update ≈ lr·sign(g) per step regardless of β1, β2). With varying
    // gradients, the bias correction actually matters and small
    // implementation drift in m̂/v̂ would surface as a measurable
    // parameter mismatch.
    auto p = Var::make((Mat(1, 3) << 2.f, -1.f, 0.5f).finished());
    Adam adam({p}, /*lr=*/0.05f);

    std::vector<Mat> grads = {
        (Mat(1, 3) <<  0.4f, -0.8f,  0.2f).finished(),
        (Mat(1, 3) << -0.5f,  0.3f, -0.1f).finished(),
        (Mat(1, 3) <<  0.2f,  0.6f, -0.3f).finished(),
    };

    // Independent reference recurrence (scalar temporaries, out-of-place).
    Mat p_ref = p->data;
    Mat m_ref = Mat::Zero(1, 3);
    Mat v_ref = Mat::Zero(1, 3);
    int  t_ref = 0;
    const float lr  = 0.05f;
    const float b1  = 0.9f;
    const float b2  = 0.999f;
    const float eps = 1e-8f;

    for (const auto& g : grads) {
        // Library step.
        p->grad = g;
        adam.step();

        // Reference step: independent code path, same Adam math.
        t_ref += 1;
        float bc1 = 1.f - std::pow(b1, static_cast<float>(t_ref));
        float bc2 = 1.f - std::pow(b2, static_cast<float>(t_ref));
        for (int j = 0; j < 3; ++j) {
            float gj = g(0, j);
            m_ref(0, j) = b1 * m_ref(0, j) + (1.f - b1) * gj;
            v_ref(0, j) = b2 * v_ref(0, j) + (1.f - b2) * gj * gj;
            float m_hat = m_ref(0, j) / bc1;
            float v_hat = v_ref(0, j) / bc2;
            float denom = std::sqrt(v_hat) + eps;
            p_ref(0, j) -= lr * m_hat / denom;
        }

        // Per-element comparison: m, v, t, parameter.
        for (int j = 0; j < 3; ++j) {
            CHECK_NEAR(adam.m[0](0, j), m_ref(0, j), 1e-5f);
            CHECK_NEAR(adam.v[0](0, j), v_ref(0, j), 1e-5f);
            CHECK_NEAR(p->data(0, j),    p_ref(0, j), 1e-5f);
        }
        CHECK(adam.t == t_ref);
    }
    report("Adam trajectory with varying signed grads matches reference m/v/t/p");
}

void test_optimizer_state_isolation() {
    // Two Adam instances on disjoint params never see each other's state.
    auto a = Var::make(Mat::Constant(1, 1, 1.f));
    auto b = Var::make(Mat::Constant(1, 1, 1.f));
    Adam adam_a({a}, 0.1f);
    Adam adam_b({b}, 0.1f);
    a->grad = Mat::Constant(1, 1, 1.f);
    adam_a.step();
    // b's data and grad must be untouched.
    CHECK_NEAR(b->data(0, 0), 1.f, 1e-6f);
    CHECK_NEAR(b->grad(0, 0), 0.f, 1e-6f);
    CHECK_NEAR(a->data(0, 0), 0.9f, 1e-3f);

    // Rebuilding Adam on the same param resets m, v, t — current API has
    // no snapshot/restore path, so constructing a new Adam is the only
    // way to reinitialize state. This pins the current behavior.
    auto c = Var::make(Mat::Constant(1, 1, 1.f));
    {
        Adam adam_c_old({c}, 0.1f);
        c->grad = Mat::Constant(1, 1, 1.f);
        adam_c_old.step();
    }  // adam_c_old destroyed; c->data is ~0.9 now.
    // New Adam on the same c: m, v, t start at zero again.
    Adam adam_c_new({c}, 0.1f);
    c->grad = Mat::Constant(1, 1, 1.f);
    adam_c_new.step();
    // Step #1 again (t=1) on a fresh Adam → same first-step update as the
    // single-step test in test_nn::test_adam_step.
    CHECK_NEAR(c->data(0, 0), 0.9f - 0.1f, 1e-3f);
    report("Optimizer state per-instance; rebuilding resets m/v/t");
}

// ── §3 Conv: repeated-backward accumulation ────────────────────────────────

void test_conv_repeated_backward_accumulates() {
    // The global accumulation rule applies to Conv2d as well: the second
    // backward on the same scalar loss adds another full pass on top of
    // the first. Because the conv output `y` is an intermediate, its
    // grad also accumulates (sum's back_fn does y.grad += ones), so the
    // second pass contributes 2x the original conv-grad to the leaf.
    // Net: conv.W.grad after 2 passes = w1 + 2*w1 = 3*w1.
    Conv2d conv(1, 1, 2, 2, 1, 0);
    conv.W->data = Mat::Constant(1, 4, 0.25f);
    conv.b->data = Mat::Zero(1, 1);

    Mat ones16 = Mat::Ones(1, 16);
    auto x = Var::make(ones16);
    auto y = conv.forward(x, 4, 4);          // 1 x 9
    auto loss = sum(y);                      // scalar (no extra ops)
    loss->backward();
    Mat w_after_first = conv.W->grad;
    Mat x_after_first = x->grad;
    CHECK(w_after_first.cwiseAbs().maxCoeff() > 0.f);
    CHECK(x_after_first.cwiseAbs().maxCoeff() > 0.f);

    loss->backward();
    // Second backward: conv.W.grad grew to 3x, x.grad grew to 3x.
    Mat w_ratio = conv.W->grad.cwiseQuotient(w_after_first);
    CHECK_NEAR((w_ratio.array() - 3.f).abs().maxCoeff(), 0.f, 1e-4f);
    Mat x_ratio = x->grad.cwiseQuotient(x_after_first);
    CHECK_NEAR((x_ratio.array() - 3.f).abs().maxCoeff(), 0.f, 1e-4f);

    // Var::zero_grad() on the loss resets the WHOLE graph (including the
    // conv input x), then a fresh backward reproduces the first pass.
    loss->zero_grad();
    CHECK_NEAR(conv.W->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    CHECK_NEAR(conv.b->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    CHECK_NEAR(x->grad.cwiseAbs().maxCoeff(),     0.f, 1e-7f);
    loss->backward();
    CHECK_EQ_MAT(conv.W->grad, w_after_first);
    CHECK_EQ_MAT(x->grad,      x_after_first);

    // Module::zero_grad zeros only the leaf parameters, NOT x (the input).
    // Re-run, then call conv.zero_grad() and verify the partial reset.
    loss->backward();
    conv.zero_grad();
    CHECK_NEAR(conv.W->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    CHECK_NEAR(conv.b->grad.cwiseAbs().maxCoeff(), 0.f, 1e-7f);
    // x's grad (input, not a parameter) is NOT touched by conv.zero_grad.
    CHECK(x->grad.cwiseAbs().maxCoeff() > 0.f);

    report("Conv2d backward accumulates; zero_grad/Module::zero_grad semantics");
}

// ── §2/§4 backward leaves data untouched ──────────────────────────────────

void test_backward_does_not_mutate_data() {
    auto p = Var::make((Mat(1, 1) << 7.f).finished());
    Mat before = p->data;
    auto y = sum(mul(p, p));
    y->backward();
    p->zero_grad();
    // Data must be unchanged after backward + zero_grad.
    CHECK_EQ_MAT(p->data, before);

    SGD sgd({p}, 0.1f);
    p->grad = Mat::Constant(1, 1, 1.f);
    sgd.step();
    // SGD step mutates data; that's expected. The contract is that
    // backward() and zero_grad() do NOT.
    CHECK(std::fabs(p->data(0, 0) - 7.f) > 1e-5f);

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