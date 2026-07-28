# Phase 0 — Behavior Contract

This document records observable behavior of the legacy
`minimal_autograd` public API and classifies what must survive the architecture
refactor described in `ARCHITECTURE_REFACTOR_PLAN.md`. Unless a row is labeled
legacy-only, explicitly unpinned, a known gap, or code-inspection-only, it is
guaranteed replacement behavior.
Each row maps a current surface to either:

- an existing test (no new test needed), or
- a focused test added in `test_characterization` to fill a Phase 0 gap, or
- an explicitly documented known gap (no public coverage; tracked for
  follow-up phases).

Characterization is evidence of what the legacy code does, not proof that
every observed side effect is desirable. Legacy-only tests may remain while
the compatibility facade exists without constraining the replacement API.

The contract is intentionally scoped to the CPU build (which is the
first-class build). CUDA parity rows are mapped to existing
`test_cuda_core`/`test_cuda_fft` tests; Phase 0 adds no new CUDA
characterization.

---

## 1. Tensor, Shape, Stride

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `Mat = Eigen::MatrixXf` | 2D dense float32 column-major | implicit in all tests | — |
| `Dims` / `Shape` | rank, indexed access, equality, `numel` | `test_core::test_logical_4d_shape` | — |
| Legacy logical stride metadata | first-axis-contiguous to match Eigen; replacement Tensor order is last-axis-contiguous row-major | `test_core::test_logical_4d_shape` | — |
| `Var::make(Mat)` | leaf with `data` and zero `grad` | all tests | — |
| `Var::make4d(Mat, N, C, H, W)` | flat 2D + 4D shape metadata | `test_core::test_logical_4d_shape` | — |
| `Var::set_shape(...)` | numel must equal data size; assert-only | `test_core::test_logical_4d_shape` (via `view`) | — |

Known gap: `set_shape` uses `assert()` (debug-only) on a numel mismatch; the
refactor must replace it with a runtime error. Documented for Phase 2.

---

## 2. Variable, backward, zero_grad, gradient accumulation

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `Var::backward()` (scalar loss) | topo-sorted reverse traversal; shared nodes visited once | `test_core::test_shared_x_plus_x`, `test_shared_x_cubed` | — |
| Repeated-parent grad accumulation | `d/dx (x+x)` = 2, not 4 | `test_core::test_shared_x_plus_x`, `test_shared_x_cubed` | — |
| Legacy `backward()` intermediate amplification | **legacy-only / unpinned:** stored intermediate gradients can amplify a later traversal | — | `test_backward_accumulates_without_zero` |
| Replacement `Variable::backward()` accumulation | each call propagates from a fresh seed and adds exactly one newly computed pass to stored gradients | `test_autograd_core::test_shared_graph_and_repeated_backward` | — |
| `Var::zero_grad()` | zeros every reachable node's `grad` (leaf + intermediate + loss) | — | `test_var_zero_grad_reaches_all_reachable` |
| `Module::zero_grad()` | zeros only registered parameter leaves, not intermediate activations | — | `test_module_zero_grad_isolates_intermediates` |
| `Optimizer::zero_grad()` (SGD/Adam) | equivalent to `Module::zero_grad()` (no internal buffers zeroed) | — | `test_optimizer_zero_grad_semantics` |
| Backward exception rollback | if any `back_fn` throws, all grads are restored to their pre-`backward()` values | `test_core::test_backward_exception_rolls_back_grads` | — |
| Backward non-scalar guard | non-scalar implicit backward is rejected; legacy assertion mechanism is unpinned and replacement uses a runtime error | `test_autograd_core::test_upstream_detach_and_zero_grad` | — |

---

## 3. Forward / backward operations

All "grad_check" entries use `test/grad_check.h` (central finite differences,
default tol 5e-2, eps 1e-3).

### 3.1 Core (test_core / test_extensions)

| Op | Forward pinned | Grad pinned | Existing test | New test |
| --- | --- | --- | --- | --- |
| `add` | elementwise sum | pass-through | `test_core::test_grad_check_ops` | — |
| `mul` | elementwise product | g·b, g·a | `test_core::test_grad_check_ops` | — |
| `matmul` | 2D matmul | g·Bᵀ, Aᵀ·g | `test_core::test_grad_check_ops` | — |
| `relu` | `max(x,0)` | mask | `test_core::test_grad_check_ops` | — |
| `sum` | scalar reduction | broadcast g | `test_core::test_grad_check_ops` | — |
| `mean` | `sum(x)/numel` | `1/numel` | `test_extensions::test_mean` | — |
| `broadcast_add` | row-bias add | pass-through; row-reduce bias | `test_core::test_broadcast_add` | — |
| `scale` | `s·x` | `s·g` | `test_core::test_scale` | — |
| `softmax` | per-row normalized | Jacobian closed-form | `test_core::test_softmax` | — |
| `log_softmax` | per-row log-softmax | `g - sm·sum(g)` | `test_core::test_log_softmax` | — |
| `transpose` | matrix transpose | transpose of g | `test_core::test_transpose` | — |
| `reshape` | data-only view (2D) | reshape g back | `test_core::test_reshape` | — |
| `concat` (row axis) | stack rows | split g row-blocks | `test_core::test_concat` | — |
| `hcat` (col axis) | stack cols | split g col-blocks | `test_extensions::test_hcat` | — |
| `sigmoid` | `1/(1+e^{-x})` | `g·s·(1-s)` | `test_extensions::test_sigmoid` | — |
| `tanh_op` | `tanh(x)` | `g·(1-t²)` | `test_extensions::test_tanh_op` | — |
| `exp_op` | `e^x` | `g·e^x` | `test_extensions::test_exp_op` | — |
| `log_op` | `log(x)` (x>0) | `g/x` | `test_extensions::test_log_op` | — |
| `sqrt_op` | `√x` (x>0) | `g/(2·√x)` | `test_extensions::test_sqrt_op` | — |
| `silu` | `x·σ(x)` | closed form | `test_extensions::test_silu` | — |
| `softplus` | `max(0,x)+log(1+e^{-|x|})` | `σ(x)·g` | `test_extensions::test_softplus` | — |
| `sub` | `a-b` | `g, -g` | `test_extensions::test_sub` | — |
| `div_op` | `a/b` | `g/b, -g·a/b²` | `test_extensions::test_div_op` | — |
| `cumsum` axis 0/1 | prefix sum | suffix sum | `test_extensions::test_cumsum` | — |
| `flip` axis 0/1 | reverse | reverse | `test_extensions::test_flip` | — |
| `sin_op`, `cos_op` | trig | `g·cos`, `-g·sin` | `test_extensions::test_sin_cos` | — |
| `clamp` | `clip(x, lo, hi)` | mask | `test_extensions::test_clamp` | — |
| `col_slice`, `row_slice` | slice; `runtime_error` on bad range | grad_check + rejection | `test_extensions::test_col_slice`, `test_row_slice` | — |
| `split` | two halves | grad_check | `test_extensions::test_split` | — |
| Shape mismatch (`sub`, `div_op`) | `runtime_error` | — | `test_core::test_sub_div_shape_mismatch` | — |

### 3.2 Conv / pool / upsample (test_conv / test_extensions / test_cuda_core)

| Op / module | Forward / classification | Gradient / classification | Existing test | New test |
| --- | --- | --- | --- | --- |
| `im2col` / `col2im` | receptive-field pack/unpack; `+=` on overlap | n/a (pure) | `test_conv::im2col/col2im round-trip` | — |
| `conv2d_op` | matches naive nested-loop reference | grad_check dInput/dWeight/dBias | `test_conv` | — |
| Legacy `conv2d_op` repeated `backward()` amplification | **legacy-only / unpinned:** two calls currently grow Conv weights by `3x` because an intermediate gradient is reused | — | `test_conv_repeated_backward_accumulates` | — |
| `Conv2d::forward(x, H, W)` | module wrapper, kaiming-uniform init | n/a | `test_conv::end-to-end Conv+Pool+Linear` | — |
| `Conv2d::forward(x)` (4D) | shape inferred; throws on channel/geometry mismatch | n/a | `test_conv::validation rejections` | — |
| `maxpool2d_op` | max over kernel window | grad_check | `test_conv::grad_check MaxPool2d` | — |
| `MaxPool2d::forward(x, H, W)` | module wrapper | n/a | `test_conv` | — |
| `avgpool2d_op` | uniform kernel mean | grad_check | `test_extensions::test_avgpool2d` | — |
| `nearest_upsample2d_op` | replicate `scale×scale` | grad_check | `test_extensions::test_nearest_upsample` | — |
| `depthwise_conv2d_op` | per-channel `1×k·k` matmul | grad_check | `test_extensions::test_depthwise_conv2d` | — |
| Conv module validation | channel mismatch / kernel geometry / flat-shape mismatch throws | — | `test_conv::validation` | — |

Known gap: there is no explicit conv-only test that uses the conv output
twice in a single forward pass and verifies single accumulation (this is
the global "no double-count" rule applied to conv); it is implicitly
covered by the same accumulation contract as `add(x, x)` and `x*x*x` in
`test_core`.

---

## 4. Modules

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `Linear(in, out)` | `W:(in,out)` Xavier-scale init `sqrt(2/in)`; `b:(1,out)` zeros | `test_nn::test_linear_forward` | — |
| `Linear::parameters()` order | `{W, b}` | — | `test_module_parameter_order` |
| `Linear::forward(x)` | `broadcast_add(matmul(x,W), b)` | grad (via `test_nn`) | — |
| `Sequential::add` / `forward` | composition in insertion order | `test_nn::test_sequential` | — |
| `Sequential::parameters()` order | flatten children in insertion order; per-child in their defined order | — | `test_module_parameter_order` |
| `ReLUModule`, `SiLUModule`, `SigmoidModule` | stateless wrappers; `parameters()={}` | `test_extensions::test_silu_module` | — |
| `Conv2d` init | Kaiming-uniform on weight, zero bias | `test_conv` | — |
| `Conv2d::parameters()` order | `{W, b}` | — | `test_module_parameter_order` (extended to nested Sequential with Conv2d) |
| `MaxPool2d`, `AvgPool2d`, `NearestUpsample2d` | stride defaults to kernel size when `< 0` | `test_conv`, `test_smoke::test_spatial_forward` | — |
| `DepthwiseConv2d` | `groups = channels`; one filter per channel | `test_extensions::test_depthwise_conv2d` | — |
| `GroupNorm` | forward-only (no `back_fn`); per-group mean/var; `assert`s on bad shape | `test_extensions::test_groupnorm_forward`, `test_smoke::test_groupnorm_silu_forward` | — |

Known gap: `GroupNorm` does not register `gamma`/`beta` in the
autograd graph (forward-only leaf). Documented; the refactor's
backward story for GroupNorm remains Phase 6 bundle 3 work.

---

## 5. Optimizers

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `SGD::step()` | `p -= lr * p->grad` | `test_nn::test_sgd_step` (1 step, scalar) | `test_sgd_trajectory` (multi-step, matrix) |
| `SGD::zero_grad()` | clears `grad` on every parameter | — | `test_optimizer_zero_grad_semantics` |
| `Adam::ctor` | initializes `m`, `v` zero per param (sized to data); `t=0` | `test_nn::test_adam_step` | — |
| `Adam::step()` (1 step) | closed-form `m=0.1,v=0.001,m̂=1,v̂=1,Δ≈-lr` | `test_nn::test_adam_step` | — |
| `Adam::step()` (multi-step, bias-corrected) | `bc1=1-β1^t`, `bc2=1-β2^t`; `m = β1·m + (1-β1)·g`; `v = β2·v + (1-β2)·g²`; `Δ = lr·(m/bc1)/(√(v/bc2)+ε)` | — | `test_adam_trajectory` |
| `Adam::zero_grad()` | clears `grad` only (not `m`/`v`) | — | `test_optimizer_zero_grad_semantics` |
| Optimizer per-param isolation | two `Adam({p})` instances do not share state; rebuilding `Adam` resets `m`, `v`, `t` | — | `test_optimizer_state_isolation` |

**Known gap (no public API):** there is no state snapshot/restore API for
`Adam`. Downstream checkpoint code (see `docs/CPPRESIST_INTEGRATION_INVENTORY.md`)
currently accesses `Adam.t`, `Adam.m`, `Adam.v` directly to serialize
optimizer state. The Phase 0 contract pins the current state-layout
behavior (per-param `m`/`v`/`t` kept inside `Adam`, separate from
parameters' `data`/`grad`) but **does not require any test** for
serialization. The refactor must introduce an explicit `AdamState`
(Phase 5b).

---

## 6. Losses

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `mse_loss(pred, target)` | `mean((pred-target)²)` over all elements | `test_nn::test_xor` (drives convergence) | — |
| `cross_entropy(pred, target)` | `-mean(sum(target·log_softmax(pred)))` for one-hot `target` | `test_nn::test_cross_entropy`, `test_multibatch_cross_entropy_adam` | — |

Both losses are compositions of existing ops; no new Function subclass.

---

## 7. Diffusion helpers

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `randn(r, c, seed)` | leaf; `std::mt19937`+`std::normal_distribution(0,1)` | `test_diffusion::test_randn_*` | — |
| `randn_like(x, seed)` | leaf; same shape as `x` | `test_diffusion::test_randn_like_*` | — |
| `sinusoidal_time_embedding(t, dim)` | leaf; `(1, dim)`; `pe[i]=sin(t·fᵢ)`, `pe[i+half]=cos(t·fᵢ)`; `fᵢ=exp(-log(10000)·2i/dim)`; `assert(dim%2==0)` | `test_diffusion::test_time_emb_*`, `test_smoke::test_embedding_consistency` | — |
| `q_sample` | `sqrt_ab·x0 + sqrt_1mab·noise`; differentiable w.r.t. `x0` and `noise` | `test_diffusion::test_q_sample_*` | — |

Known gap: `sinusoidal_time_embedding` uses `assert(dim%2==0)` (debug-only).
The refactor must convert to a runtime error. Documented for Phase 6 bundle 4.

---

## 8. Complex and FFT

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `real_to_complex` / `make_complex` / `real` / `imag` | pair of `Var` with shape and device validation | `test_fft::test_real_to_complex_and_accessors`, `test_complex_*` | — |
| `complex_mul`, `conj`, `complex_scale`, `abs2` | standard formulas; grad_check | `test_fft::test_complex_grad_check` | — |
| Repeated complex branch grad | accumulates once | `test_fft::test_repeated_branch_gradient` | — |
| `fft2(z)` / `ifft2(z)` | O(MN·(MN)) DFT reference; `Backward` normalization (i.e., forward unscaled, inverse scaled by 1/(M·N)) | `test_fft::test_fft_round_trip`, `test_fft_known_fixtures`, `test_fft_non_square_round_trip` | — |
| `fft2` known fixtures | constant input → DC bin = N·M, others ≈ 0; delta input → all-ones spectrum | `test_fft::test_fft_known_fixtures` | — |
| `fft2` grad_check | spectral filter end-to-end | `test_fft::test_fft_spectral_filter_grad_check`, `test_fft_component_grad_checks` | — |
| `FftNorm::Backward` only | other norms throw at runtime | implicit (no test for the throw) | — |

Known gap: there is no test that asserts the explicit `runtime_error` for
non-`Backward` `FftNorm` values. Trivial; covered by code inspection
(`fft.cpp:72-74`). Not adding a Phase 0 test for this.

---

## 9. CUDA (CPU-build stub and full CUDA build)

When `AUTOGRAD_USE_CUDA=OFF`:

- `Var::cuda()` throws `runtime_error("Var::cuda(): built without AUTOGRAD_USE_CUDA")` — covered by code inspection in `variable.cpp:21-24`. Not asserted in a Phase 0 test.
- Free-function ops do **not** redirect (no `cuda_*_op` available). A CUDA-typed `Var` cannot be created in a CPU build; the `Var::cuda()` throw is the gate.

When `AUTOGRAD_USE_CUDA=ON`:

| Surface | Behavior / classification | Existing test | New test |
| --- | --- | --- | --- |
| `Var::cuda()` / `cpu()` round-trip | preserves data and grad; copies both directions | `test_cuda_core::shaped_*` | — |
| `cuda_add_op`, `cuda_mul_op`, `cuda_matmul_op`, `cuda_broadcast_add_op`, `cuda_scale_op`, `cuda_relu_op`, `cuda_sigmoid_op`, `cuda_tanh_op`, `cuda_exp_op`, `cuda_log_op`, `cuda_sqrt_op`, `cuda_silu_op`, `cuda_softplus_op`, `cuda_sub_op`, `cuda_div_op`, `cuda_sum_op`, `cuda_col_slice_op`, `cuda_row_slice_op`, `cuda_softmax_op`, `cuda_log_softmax_op` | CUDA forward and grad match CPU within `1e-4` | `test_cuda_core::check_unary_op`, `check_binary_op`, etc. | — |
| `cuda_conv2d_op`, `cuda_maxpool2d_op` | CUDA forward and grad match CPU | `test_cuda_core::Conv2d forward/grad`, `MaxPool2d forward/grad` | — |
| `cuda_sgd_step`, `cuda_adam_step` | CUDA optimizer matches CPU reference | `test_cuda_core::cuda SGD`, `cuda Adam` | — |
| Mixed-device error (`Var::cuda()` + CPU leaf) | throws | `test_cuda_core::mixed_threw` | — |
| Unsupported CUDA op (`transpose` on CUDA, `AvgPool2d`, `DepthwiseConv2d`, `NearestUpsample2d` on CUDA) | throws | `test_cuda_core::unsupported_threw` | — |
| `complex_mul`, `conj`, `complex_scale`, `abs2`, `make_complex` CUDA paths | match CPU | `test_cuda_fft::test_cuda_complex_*` | — |
| `fft2`, `ifft2` CUDA forward parity | match CPU; non-power-of-two input throws | `test_cuda_fft::test_cuda_fft_*` | — |
| `fft2`, `ifft2` CUDA grad parity | match CPU | `test_cuda_fft::test_cuda_fft_backward_parity`, `spectral_filter_backward`, `repeated_branch_backward` | — |

Phase 0 adds no new CUDA test; the existing suite (`test_cuda_core`,
`test_cuda_fft`) is the CUDA contract. Per the user's instruction, Phase 0
must not change CUDA production code.

---

## 10. Built-in exceptions and other invariants

| Invariant | Behavior | Source | Test |
| --- | --- | --- | --- |
| `sub` / `div_op` shape mismatch | `std::runtime_error` | `ops.h:597,607` | `test_core::test_sub_div_shape_mismatch` |
| `col_slice` / `row_slice` bad range | `std::runtime_error` | `ops.h:430,623,628` | `test_extensions::test_col_slice/row_slice` |
| Conv/Pool shape/geometry mismatch | `std::runtime_error` | `conv.cpp:11-29, etc.` | `test_conv::validation` |
| CUDA-built unsupported op | `std::runtime_error` | `ops.h:37-39, conv.cpp:184, etc.` | `test_cuda_core::unsupported_threw` |
| Mixed CPU/CUDA op | `std::runtime_error` (via shape path or explicit check) | `ops.h:594-609, conv.cpp:441-442, etc.` | `test_cuda_core::mixed_threw`, `test_cuda_fft::test_cuda_complex_validation` |
| Backward exception rollback | all grads restored | `variable.cpp:118-133` | `test_core::test_backward_exception_rolls_back_grads` |

---

## 11. What Phase 0 does NOT pin

Phase 0 is a freeze, not a redesign. The following are explicitly **not**
contract yet and may change during Phases 1-11:

- legacy repeated-backward amplification through stored intermediate
  gradients, including the Conv2d `3x` result after two calls;
- Internal representation of `Mat` (Phase 2 introduces `Tensor`).
- Public visibility of `parents`, `back_fn`, `cuda_data_`, `cuda_grad_`,
  `grad`, `data` (Phase 11 removes direct access).
- Optimizer state snapshot/restore (Phase 5b introduces explicit
  `AdamState`).
- Per-CUDA-moment `cuda_m` / `cuda_v` duplication on `Adam` (Phase 5b defines
  unified state; Phase 9 migrates CUDA optimizer kernels).
- GroupNorm backward (Phase 6 bundle 3).
- FFT non-`Backward` normalizations.

The replacement API instead guarantees a fresh propagation seed per
`backward()` call. Previously committed gradients accumulate only at commit
time and are not inputs to the next traversal.

## 12. Executable coverage policy

Every guaranteed CPU row must map to a registered test that CI executes with
CTest. Merely building `test_characterization`, `test_shape_device`,
`test_tensor`, `test_autograd_core`, or `test_cpu_ops` does not satisfy the
contract.

CUDA parity rows require execution of `test_cuda_core` and `test_cuda_fft` on
CUDA-capable hardware. Until hosted CUDA CI exists, CUDA PRs must record the
exact external/manual command, hardware, and result. Rows covered only by code
inspection remain labeled gaps and must not be described as executable
coverage.

---

## 13. Files added or modified by Phase 0

- `docs/PHASE0_BEHAVIOR_CONTRACT.md` — this document.
- `test/test_characterization.cpp` — focused tests for the gaps in §2, §4, §5.
- `CMakeLists.txt` — adds the `test_characterization` target.

No public header, source file, or production behavior is modified. CUDA
production code is untouched.
