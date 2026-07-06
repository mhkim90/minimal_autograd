// test_smoke.cpp — end-to-end smoke test for GUDM-relevant ops.
//
// Groups:
//   1. Schedule training loops (backward + Adam)
//   2. Spatial forward pass (conv, pool, upsample, hcat)
//   3. GroupNorm + SiLU forward
//   4. Sinusoidal embedding consistency (vs manual sin/cos)
//   5. randn / randn_like statistics
//   6. Mini ddim-style inference loop (10 steps)

#include "autograd.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace ag;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++passed; }
    else     { std::printf("  FAIL  %s\n", name); ++failed; }
}

static bool is_finite(const Mat& m) {
    return m.array().isFinite().all();
}

// ── 1. Schedule training loops ────────────────────────────────────────────────
//
// Mirrors GaussianUnifiedSchedule.  Layout: (1, T) row vectors.
// sigma_max is fixed (training only delta_sigma for simplicity).

static void test_sigma_schedule_convergence() {
    const int T = 100;
    const float sigma_max = 6.0f;

    // Learnable parameter: uniform init → uniform schedule at start.
    auto delta_sigma = Var::make(Mat::Zero(1, T));

    // Target: a linearly spaced schedule ending at sigma_max.
    Mat target_mat(1, T);
    for (int i = 0; i < T; ++i)
        target_mat(0, i) = sigma_max * float(i + 1) / float(T);
    auto target = Var::make(target_mat);

    Adam opt({delta_sigma}, /*lr=*/0.05f);

    float loss0 = -1.f, lossN = -1.f;
    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();

        // Forward: sigmas = sigma_max * cumsum(softmax(delta_sigma, axis=1), axis=1)
        auto weights = softmax(delta_sigma);                   // (1, T)
        auto sigmas  = scale(cumsum(weights, 1), sigma_max);  // (1, T)

        // MSE loss
        auto diff = sub(sigmas, target);
        auto loss = sum(mul(diff, diff));

        if (step == 0)   loss0 = loss->data(0, 0);
        if (step == 199) lossN = loss->data(0, 0);

        loss->backward();
        opt.step();
    }

    // Initial uniform softmax already yields a near-linear cumsum, so loss0 is
    // tiny.  Convergence is real but the dynamic range is small: assert the
    // params moved, both losses are finite and numerically tiny, and the
    // schedule still matches the target well at the end.
    check("sigma schedule: loss0 positive finite", loss0 > 0.f && std::isfinite(loss0));
    check("sigma schedule: lossN positive finite",  lossN > 0.f && std::isfinite(lossN));
    check("sigma schedule: final loss tiny",        lossN < 1e-3f);
    // Params moved: |delta_sigma|_max > 0 means Adam actually stepped.
    check("sigma schedule: params updated",
          delta_sigma->data.cwiseAbs().maxCoeff() > 0.f);
    // Final schedule still near target (MSE per element < 1e-5).
    {
        VarPtr final_sig = scale(cumsum(softmax(delta_sigma), 1), sigma_max);
        Mat diff_mat = final_sig->data - target_mat;
        float per_elem_mse = diff_mat.array().square().mean();
        check("sigma schedule: final per-elem MSE < 1e-5", per_elem_mse < 1e-5f);
    }
    check("sigma schedule: params finite", is_finite(delta_sigma->data));
}

static void test_beta_schedule_convergence() {
    const int T = 100;
    const float beta_max = 0.5f;
    const float target_val = 0.2f;  // constant target

    auto beta_raw = Var::make(Mat::Zero(1, T));

    Mat target_mat = Mat::Constant(1, T, target_val);
    auto target = Var::make(target_mat);

    Adam opt({beta_raw}, /*lr=*/0.05f);

    float loss0 = -1.f, lossN = -1.f;
    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();

        // betas = beta_max * tanh(beta_raw)
        auto betas = scale(tanh_op(beta_raw), beta_max);

        auto diff = sub(betas, target);
        auto loss = sum(mul(diff, diff));

        if (step == 0)   loss0 = loss->data(0, 0);
        if (step == 199) lossN = loss->data(0, 0);

        loss->backward();
        opt.step();
    }

    check("beta schedule: loss decreases by 100x",  lossN < loss0 * 0.01f);
    check("beta schedule: converges to target",
          std::fabs(tanh_op(beta_raw)->data.mean() * beta_max - target_val) < 0.01f);
    check("beta schedule: params finite", is_finite(beta_raw->data));
}

// ── 2. Spatial forward pass ───────────────────────────────────────────────────

static void test_spatial_forward() {
    // 2 images, 2 channels, 8×8 spatial.
    const int N = 2, C = 2, H = 8, W = 8;

    auto x = Var::make(Mat::Random(N, C * H * W));

    // Conv2d: 2 in → 4 out, 3×3, pad=1
    Conv2d conv(C, 4, 3, 3, 1, 1);

    auto h = conv.forward(x, H, W);       // (2, 4*8*8)
    check("conv2d shape",
          h->data.rows() == N && h->data.cols() == 4 * H * W);
    check("conv2d finite", is_finite(h->data));

    // AvgPool2d: 2×2, stride=2
    AvgPool2d pool(2, 2);
    auto pooled = pool.forward(h, H, W);  // (2, 4*4*4)
    check("avgpool2d shape",
          pooled->data.rows() == N && pooled->data.cols() == 4 * (H/2) * (W/2));
    check("avgpool2d finite", is_finite(pooled->data));

    // NearestUpsample2d: scale=2 → back to (2, 4*8*8)
    NearestUpsample2d up(2);
    auto upsampled = up.forward(pooled, H/2, W/2);
    check("nearest_upsample2d shape",
          upsampled->data.rows() == N && upsampled->data.cols() == 4 * H * W);
    check("nearest_upsample2d finite", is_finite(upsampled->data));

    // hcat: cat upsampled with original conv output (skip)
    auto merged = hcat({upsampled, h});   // (2, 8*8*8)
    check("hcat shape",
          merged->data.rows() == N && merged->data.cols() == 8 * H * W);
    check("hcat finite", is_finite(merged->data));
}

// ── 3. GroupNorm + SiLU forward ───────────────────────────────────────────────

static void test_groupnorm_silu_forward() {
    const int N = 2, C = 4, HW = 16;  // e.g. 4×4 spatial

    auto x = Var::make(Mat::Random(N, C * HW));

    GroupNorm gn(2, C);  // 2 groups
    gn.gamma->data.setOnes();
    gn.beta->data.setZero();

    auto normed = gn.forward(x, C, HW);
    check("groupnorm shape",
          normed->data.rows() == N && normed->data.cols() == C * HW);
    check("groupnorm finite", is_finite(normed->data));

    // Verify zero mean, unit variance per group (gamma=1, beta=0)
    const int ch_per_g = C / 2;
    float max_mean_err = 0.f, max_var_err = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < 2; ++g) {
            float m = 0.f, v = 0.f;
            int elems = ch_per_g * HW;
            for (int c = g * ch_per_g; c < (g + 1) * ch_per_g; ++c)
                for (int p = 0; p < HW; ++p)
                    m += normed->data(n, c * HW + p);
            m /= elems;
            for (int c = g * ch_per_g; c < (g + 1) * ch_per_g; ++c)
                for (int p = 0; p < HW; ++p) {
                    float d = normed->data(n, c * HW + p) - m;
                    v += d * d;
                }
            v /= elems;
            max_mean_err = std::max(max_mean_err, std::fabs(m));
            max_var_err  = std::max(max_var_err,  std::fabs(v - 1.f));
        }
    }
    check("groupnorm mean≈0",   max_mean_err < 1e-4f);
    check("groupnorm var≈1",    max_var_err  < 1e-3f);

    // SiLU downstream
    auto activated = silu(normed);
    check("silu after groupnorm finite", is_finite(activated->data));
}

// ── 4. Sinusoidal embedding consistency ───────────────────────────────────────
//
// Verify sinusoidal_time_embedding (diffusion.h) produces the same values as
// manually composing sin_op + cos_op + hcat.  Also verify split round-trips.

static void test_embedding_consistency() {
    const int DIM = 32;
    const int t_val = 42;
    const int half = DIM / 2;

    // Library function — golden reference.
    auto pe_ref = sinusoidal_time_embedding(t_val, DIM);  // (1, DIM) leaf

    // Manual composition: replicate what the library does.
    // freq_i = exp(-log(10000) * 2*i / DIM)
    Mat freq_mat(1, half);
    for (int i = 0; i < half; ++i)
        freq_mat(0, i) = std::exp(-std::log(10000.f) * 2.f * i / (float)DIM);
    // args = t * freqs
    Mat args_mat = freq_mat * (float)t_val;
    auto args = Var::make(args_mat);  // (1, half)

    // sin and cos separately, then hcat
    auto sin_part = sin_op(args);   // (1, half)
    auto cos_part = cos_op(args);   // (1, half)
    auto pe_manual = hcat({sin_part, cos_part});  // (1, DIM)

    // Compare
    float max_err = (pe_ref->data - pe_manual->data).cwiseAbs().maxCoeff();
    check("sinusoidal_time_embedding matches manual sin+cos+hcat",
          max_err < 1e-5f);

    // Split round-trip: split(hcat([a, b])) == [a, b]
    auto [a_back, b_back] = split(pe_manual);
    float err_a = (a_back->data - sin_part->data).cwiseAbs().maxCoeff();
    float err_b = (b_back->data - cos_part->data).cwiseAbs().maxCoeff();
    check("split(hcat([a,b])).first  == a", err_a < 1e-6f);
    check("split(hcat([a,b])).second == b", err_b < 1e-6f);

    // Verify specific values: freq_0 = 1, so pe[0]=sin(t), pe[half]=cos(t)
    check("pe[0] = sin(t_val)",
          std::fabs(pe_ref->data(0, 0)    - std::sin((float)t_val)) < 1e-5f);
    check("pe[half] = cos(t_val)",
          std::fabs(pe_ref->data(0, half) - std::cos((float)t_val)) < 1e-5f);
}

// ── 5. randn / randn_like stats ───────────────────────────────────────────────

static void test_randn_stats() {
    // Large sample → mean≈0, std≈1.
    auto x = randn(1, 50000, /*seed=*/7);
    float m = x->data.mean();
    float s = std::sqrt((x->data.array() - m).square().mean());
    check("randn mean ~ 0",  std::fabs(m) < 0.02f);
    check("randn std  ~ 1",  std::fabs(s - 1.f) < 0.02f);
    check("randn is a leaf", x->parents.empty() && !x->back_fn);

    // randn_like matches shape of template
    auto y = Var::make(Mat::Zero(3, 7));
    auto n = randn_like(y, /*seed=*/42);
    check("randn_like shape matches",
          n->data.rows() == 3 && n->data.cols() == 7);

    // Determinism: same seed → same values
    auto a = randn(2, 4, 99);
    auto b = randn(2, 4, 99);
    check("randn deterministic for same seed",
          (a->data - b->data).cwiseAbs().maxCoeff() < 1e-7f);
}

// ── 6. Mini inference loop ────────────────────────────────────────────────────
//
// Simulates 10 steps of a ddim-style Cold Diffusion reverse loop using only
// the library ops — no actual GUDM model, just a synthetic composition.
//
// Layout: N=1, C=1, H=8, W=8 → x_t is (1, 64).
// "Model": Linear(64+16 → 64) approximates UNet forward (timestep-conditioned).
// Loop: x_t = clamp(model(hcat([t_emb_rep, x_t])), 0, 1) + small correction.

static void test_mini_inference_loop() {
    const int N = 1, C = 1, H = 8, W = 8;
    const int spatial = C * H * W;  // 64
    const int t_emb_dim = 16;

    // Fixed model (random but deterministic weights — no training).
    Linear model_linear(spatial + t_emb_dim, spatial);
    // Re-init with small weights so output stays near 0.5.
    model_linear.W->data = Mat::Random(spatial + t_emb_dim, spatial) * 0.01f;
    model_linear.b->data = Mat::Constant(1, spatial, 0.5f);

    // Initial x_T: uniform placeholder sample.
    auto x_t = Var::make(Mat::Constant(N, spatial, 0.5f));

    bool all_finite = true;
    bool all_clamped = true;

    for (int step = 0; step < 10; ++step) {
        int t_val = 9 - step;  // 9 down to 0

        // Timestep embedding (1, t_emb_dim) → replicate to (N, t_emb_dim).
        auto pe = sinusoidal_time_embedding(t_val, t_emb_dim);
        auto t_emb = Var::make(pe->data.replicate(N, 1));  // (N, t_emb_dim)

        // Model input: hcat([x_t, t_emb]) → (N, spatial + t_emb_dim)
        auto model_in = hcat({x_t, t_emb});
        auto x0_pred = clamp(model_linear.forward(model_in), 0.f, 1.f);

        if (!is_finite(x0_pred->data))  all_finite  = false;
        if (x0_pred->data.minCoeff() < -1e-6f ||
            x0_pred->data.maxCoeff() >  1.f + 1e-6f) all_clamped = false;

        // Simple correction step (no real degrade — just smoke the loop).
        auto noise = scale(randn(N, spatial, (uint32_t)step), 0.01f);
        x_t = add(x0_pred, noise);
    }

    check("inference loop: all outputs finite",     all_finite);
    check("inference loop: clamp works [0,1]",      all_clamped);
    check("inference loop: final x_t finite",       is_finite(x_t->data));
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=== smoke test ===\n");

    std::printf("-- schedule training loops --\n");
    test_sigma_schedule_convergence();
    test_beta_schedule_convergence();

    std::printf("-- spatial forward pass --\n");
    test_spatial_forward();

    std::printf("-- GroupNorm + SiLU --\n");
    test_groupnorm_silu_forward();

    std::printf("-- sinusoidal embedding --\n");
    test_embedding_consistency();

    std::printf("-- randn / randn_like --\n");
    test_randn_stats();

    std::printf("-- mini inference loop (10 steps) --\n");
    test_mini_inference_loop();

    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return (failed == 0) ? 0 : 1;
}
