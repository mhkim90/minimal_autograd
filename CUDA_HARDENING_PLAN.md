# minimal_autograd CUDA Hardening Plan

## Goal

Harden the existing `AUTOGRAD_USE_CUDA` backend so downstream projects such as
CppResist can build CUDA features on top of `minimal_autograd` without
duplicating device memory utilities, inventing parallel CUDA abstractions, or
silently breaking autograd semantics.

This is a prerequisite plan for CppResist CUDA work. CppResist needs:

- reliable CUDA opt-in when `minimal_autograd` is consumed through
  `add_subdirectory`;
- missing CUDA elementwise ops used by CppResist resist formulas;
- clear host/device sync semantics;
- test behavior that is clean on systems with CUDA Toolkit but no GPU;
- enough documented CUDA surface for CppResist to decide whether resist kernels
  should use existing `ag::Var` CUDA support or CppResist-local kernels.

## Current State

Existing CUDA support:

- CMake option: `AUTOGRAD_USE_CUDA`.
- CUDA source: `src/cuda_core.cu`.
- CUDA header: `include/autograd/cuda_core.h`.
- Device transfer: `Var::cuda(device)`, `Var::cpu()`.
- Sync helpers: `sync_data_from_cuda`, `sync_grad_from_cuda`,
  `sync_data_to_cuda`, `sync_grad_to_cuda`.
- Device pointers: `cuda_data()`, `cuda_grad()`.
- CUDA ops:
  - add;
  - multiply;
  - matmul;
  - broadcast add;
  - scale;
  - relu;
  - sum;
  - softmax;
  - log-softmax;
  - conv2d;
  - maxpool2d;
  - SGD step.
- CUDA loss path through `cross_entropy`.
- CUDA test: `test/test_cuda_core.cpp`.
- CUDA example: `examples/mnist_classify_gpu.cpp`.

Known gaps:

- CUDA elementwise activation coverage is incomplete:
  - `sigmoid`, `tanh_op`, `exp_op`, `log_op`, `sqrt_op`, `silu`, `softplus`
    currently route to CPU `apply<Fn>` even for CUDA input.
- Binary ops added later are CPU-only:
  - `sub`, `div_op`.
- Other advanced ops are CPU-only:
  - `transpose`, `reshape`, `concat`, `hcat`, `cumsum`, `flip`, `sin_op`,
    `cos_op`, `clamp`, `col_slice`, `split`.
- `Adam::step()` previously threw for CUDA parameters; Phase 3 adds a tested
  CUDA Adam path with device moment buffers.
- No explicit no-device skip helper exists in tests.
- CUDA Toolkit version/device capability is not reported by the test.
- Public CUDA helpers are minimal and mostly raw pointer level.
- No FFT/cuFFT support exists. This is not required for minimal_autograd
  hardening unless a later project decides FFT belongs in the core autograd
  library.

## Non-Goals

- Do not implement cuFFT/FFT in this hardening pass.
- Do not add complex tensors.
- Do not add a broad tensor framework.
- Do not make CUDA mandatory for default builds.
- Do not add new dependencies beyond CUDA Toolkit for CUDA builds.
- Do not change CPU behavior or public CPU API semantics.
- Do not port every advanced op to CUDA unless a downstream use case needs it.

## Design Rules

- CPU default build must stay unchanged.
- CUDA support must remain opt-in through `AUTOGRAD_USE_CUDA=ON`.
- Unsupported CUDA ops must fail clearly; no silent host fallback from stale
  CPU mirrors.
- CUDA outputs must preserve logical shape metadata just like CPU outputs.
- Any CUDA op exposed through a public free function must have forward and
  backward tests against the CPU implementation.
- Prefer extending existing `ag::Var` CUDA support over adding new device
  tensor types.
- Keep raw CUDA helpers small and documented.

## Phase 0: Audit and Branch Setup

Deliverables:

- Commit this plan.
- Create a CUDA hardening branch, for example:

```bash
git checkout -b codex/cuda-hardening
```

- Record current CUDA surface in this plan or follow-up review note.
- Verify current CPU tests pass.
- Verify current CUDA test status on the local machine if a CUDA device is
  available.

Validation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/test_core
./build/test_nn
./build/test_conv
```

Optional CUDA validation:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/test_cuda_core
```

Exit criteria:

- Plan exists.
- CPU validation is green.
- CUDA baseline is recorded as green, skipped due no device, or blocked with a
  concrete toolchain error.

## Phase 1: CUDA Build and Test Harness Hardening

Goal:

Make CUDA test behavior explicit and friendly for both direct builds and
downstream `add_subdirectory` builds.

Deliverables:

- Add a small CUDA runtime capability helper, guarded by `AUTOGRAD_USE_CUDA`.
  It should answer:
  - CUDA runtime compiled in;
  - device count;
  - current device name;
  - compute capability;
  - driver/runtime version if easily available.
- Add a test helper for CUDA availability:
  - if CUDA is not compiled, CUDA tests are not built;
  - if CUDA is compiled but no device exists, `test_cuda_core` exits cleanly
    with a clear skip message;
  - if CUDA runtime errors occur after a device is selected, fail loudly.
- Confirm `AUTOGRAD_USE_CUDA=ON` works when the project is consumed by
  CppResist through `add_subdirectory`.
- Keep the current `AUTOGRAD_USE_CUDA requires CMake 3.18+` check.
- Document the CUDA build and no-device behavior in `README.md`.

Suggested files:

- `include/autograd/cuda_core.h`
- `src/cuda_core.cu`
- `test/test_cuda_core.cpp`
- `README.md`
- `CMakeLists.txt` only if needed

Validation:

- CPU default build and tests.
- CUDA build and `test_cuda_core` on a GPU machine.
- If possible, CUDA build on a machine/container with Toolkit but no visible
  GPU to confirm skip behavior.

Exit criteria:

- CPU-only users see no CUDA requirement.
- CUDA users get an actionable error or skip message.
- Downstream CMake projects can opt into CUDA without patching
  `minimal_autograd`.

## Phase 2: CUDA Elementwise Core Ops

Goal:

Add CUDA support for the elementwise ops needed by CppResist resist formulas
and common small neural models.

Priority ops:

- `sigmoid`
- `tanh_op`
- `exp_op`
- `log_op`
- `sqrt_op`
- `silu`
- `softplus`
- `sub`
- `div_op`

Rationale:

- CppResist uses sigmoid-like behavior in resist probability and SReLU/ISReLU
  variants.
- `silu` and `softplus` are already public CPU ops and are useful to keep CUDA
  model behavior consistent with README claims.
- `sub` and `div_op` are basic arithmetic gaps after `add` and `mul`.

Deliverables:

- Add CUDA kernels in `src/cuda_core.cu`.
- Add declarations in `include/autograd/cuda_core.h`.
- Route public wrappers in `include/autograd/ops.h` when inputs are CUDA.
- Ensure mixed CPU/CUDA inputs throw clear same-device errors.
- Preserve output shape metadata.
- Add CPU vs CUDA forward/backward tests in `test/test_cuda_core.cpp`.

Test cases:

- Non-square matrices.
- Positive and negative values.
- Edge values for stable functions:
  - sigmoid/silu/softplus around large negative and large positive inputs;
  - log/sqrt on strictly positive inputs.
- Backward checks by comparing CUDA gradients to CPU gradients for
  `sum(op(x))`.
- Binary op gradients for `sub` and `div_op` against CPU.

Exit criteria:

- All priority ops return CUDA Vars when input is CUDA.
- CPU/CUDA data and gradients match within tolerance.
- Existing CPU tests unchanged.

## Phase 3: CUDA Optimizer Hardening

Goal:

Close the largest training-loop gap: `Adam::step()` currently rejects CUDA
parameters.

Two acceptable paths:

1. Implement CUDA Adam.
2. Keep Adam CPU-only but document that CUDA training currently requires SGD.

Recommended path:

- Implement CUDA Adam if CppResist or examples need GPU training loops.
- Otherwise, keep CUDA Adam as a deliberate non-goal and make the error/docs
  precise.

If implementing CUDA Adam:

- Store Adam moment buffers for CUDA parameters on device.
- Avoid copying gradients to CPU every step.
- Keep CPU Adam behavior unchanged.
- Support mixed parameter sets only if implementation stays simple; otherwise
  reject mixed CPU/CUDA Adam with a clear message.
- Add tests comparing CPU Adam and CUDA Adam for a tiny deterministic model
  after one or several steps.

Suggested files:

- `include/autograd/optim.h`
- `src/optim.cpp`
- `include/autograd/cuda_core.h`
- `src/cuda_core.cu`
- `test/test_cuda_core.cpp`

Exit criteria:

- Either CUDA Adam is tested, or the CUDA optimizer limitation is documented
  and not surprising.

## Phase 4: CUDA Shape and Sync Semantics

Goal:

Make host/device mirror behavior explicit and reduce stale-data bugs for
downstream projects.

Deliverables:

- Document when `Var::data` and `Var::grad` are authoritative for CUDA Vars.
- Document correct use of:
  - `cpu()`;
  - `sync_data_from_cuda`;
  - `sync_grad_from_cuda`;
  - `sync_data_to_cuda`;
  - `sync_grad_to_cuda`.
- Add tests for:
  - shape metadata preserved by `cuda()` and `cpu()`;
  - gradients after `backward()` are visible through `cpu()`;
  - `clear_grad()` clears both CPU and CUDA gradients;
  - `sync_*` methods do not change shape metadata.
- Consider adding safer convenience functions if tests reveal misuse risk:
  - `data_host()` / `grad_host()` style methods are optional, not required.

Exit criteria:

- CppResist can safely move an aerial-image `ag::Var` to CUDA, run supported
  ops, and copy results/gradients back without guessing sync behavior.

## Phase 5: CUDA Conv/Pooling Review

Goal:

Reconfirm existing CUDA conv/pool behavior before downstream code relies on it.

Deliverables:

- Keep existing `Conv2d` and `MaxPool2d` CPU/CUDA parity tests.
- Add or confirm tests for:
  - non-square H/W;
  - stride > 1;
  - padding edge cases;
  - multiple input/output channels;
  - logical shape preservation;
  - gradient accumulation after repeated graph branches if applicable.
- Decide whether `AvgPool2d`, `DepthwiseConv2d`, and `NearestUpsample2d` should
  remain CPU-only or get CUDA support.

Recommendation:

- Do not port `AvgPool2d`, `DepthwiseConv2d`, or `NearestUpsample2d` in this
  hardening pass unless CppResist or another immediate downstream user needs
  them.

Exit criteria:

- Existing CUDA conv/pool support is trusted and documented.
- Unsupported conv-stack modules fail clearly on CUDA input.

## Phase 6: Downstream CppResist Readiness Review

Goal:

Decide whether CppResist CUDA can proceed without adding more generic CUDA
support to `minimal_autograd`.

Readiness checklist:

- `AUTOGRAD_USE_CUDA=ON` composes cleanly under CppResist's CMake.
- CUDA elementwise priority ops pass CPU/CUDA parity tests.
- CUDA sync semantics are documented.
- CUDA no-device behavior is clean.
- CUDA optimizer status is documented.
- CppResist can choose between:
  - value-level CUDA optics with CPU resist;
  - CUDA optics plus CUDA `ag::Var` resist once CppResist implements or
    refactors its resist bank.

Exit criteria:

- Update CppResist `CUDA_BACKEND_PLAN.md` with the new
  `minimal_autograd` baseline.
- Then proceed to CppResist CUDA Phase 1.

## Suggested Commit Breakdown

1. `Document CUDA hardening plan`
2. `Harden CUDA runtime test harness`
3. `Add CUDA elementwise activation ops`
4. `Add CUDA binary arithmetic ops`
5. `Document CUDA sync semantics`
6. Optional: `Add CUDA Adam optimizer`
7. `Review CUDA conv and pooling coverage`

Each commit should keep CPU tests green.

## Validation Matrix

CPU default:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/test_core
./build/test_nn
./build/test_conv
```

CPU advanced, when touching advanced public ops:

```bash
cmake -S . -B build-advanced \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_BUILD_ADVANCED_OPS=ON
cmake --build build-advanced --parallel
./build-advanced/test_extensions
./build-advanced/test_smoke
```

CUDA:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/test_cuda_core
```

Downstream CppResist smoke after hardening:

```bash
cmake -S ../CppResist -B ../CppResist/build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPPRESIST_ENABLE_CUDA=ON
cmake --build ../CppResist/build-cuda --parallel
ctest --test-dir ../CppResist/build-cuda --output-on-failure
```

The CppResist command is expected to become valid after CppResist adds its own
CUDA build option. Before that, use it only as a design target.

## Phase 0 Baseline: 2026-07-05

Branch:

- `codex/cuda-hardening`

Plan commit:

- `a8fdf1b Document CUDA hardening plan`

Host CUDA environment:

- `nvidia-smi` sees GPU 0: NVIDIA GeForce RTX 5060, compute capability 12.0,
  8151 MiB, driver 595.71.05.
- `nvcc` is not on the default `PATH`.
- CMake found CUDA compiler `/usr/local/cuda/bin/nvcc`.
- `/usr/local/cuda/bin/nvcc --version`: CUDA compilation tools 12.8,
  V12.8.61.
- `build-cuda/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake` records
  `CMAKE_CUDA_ARCHITECTURES_NATIVE` as `No CUDA devices found.-real`, while
  runtime `nvidia-smi` sees the GPU.
- CMake's CUDA architecture list for this toolkit tops out at `90`; the local
  GPU is compute capability 12.0. Phase 1 should avoid relying on implicit
  architecture defaults and should document or configure the intended
  forward-compatibility behavior.

CPU default validation:

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: passed.
- `cmake --build build --parallel`: passed.
- `./build/test_core`: `ALL CORE TESTS PASSED`.
- `./build/test_nn`: `ALL NN TESTS PASSED`.
- `./build/test_conv`: 21 passed, 0 failed.

CPU advanced validation:

- `cmake -S . -B build-advanced -DCMAKE_BUILD_TYPE=Release
  -DAUTOGRAD_BUILD_ADVANCED_OPS=ON`: passed.
- `cmake --build build-advanced --parallel`: passed.
- `./build-advanced/test_extensions`: 30/30 passed.
- `./build-advanced/test_diffusion`: 17/17 passed.
- `./build-advanced/test_smoke`: 35/35 passed.

CUDA validation:

- `cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
  -DAUTOGRAD_USE_CUDA=ON`: passed with CMake `CMP0104` warning because
  `CUDA_ARCHITECTURES` is empty for target `autograd`.
- `cmake --build build-cuda --parallel`: passed with existing NVCC/Eigen
  constexpr warnings and a deprecated GPU target warning.
- `./build-cuda/test_cuda_core`: `ALL CUDA CORE TESTS PASSED`.

Phase 1 follow-up risks from this baseline:

- CUDA tests pass today, but the build relies on implicit architecture behavior.
- Runtime device visibility and CMake configure-time native architecture
  detection disagree.
- README guidance should mention CUDA compiler discovery when `nvcc` is not on
  `PATH`, for example by using `/usr/local/cuda/bin` or CMake CUDA variables.

## Phase 3 Status: 2026-07-05

Phase 3 implements CUDA Adam rather than documenting it as CPU-only.

Implementation notes:

- Adam keeps existing CPU moment matrices for CPU parameters.
- CUDA parameters get device-resident first and second moment buffers owned by
  internal `Var` objects.
- `Adam::step()` supports mixed CPU/CUDA parameter lists by updating each
  parameter on its own backend.
- CUDA Adam uses the same bias correction and update formula as CPU Adam.

Validation:

- `test_cuda_core` compares CPU and CUDA Adam over three deterministic gradient
  steps and checks that `zero_grad()` clears CUDA gradients.

## Phase 4 Status: 2026-07-05

Phase 4 keeps the existing explicit-sync model and documents it as the public
CUDA contract.

Implementation notes:

- CUDA `Var` host matrices and device buffers are separate mirrors.
- After CUDA forward/backward/optimizer work, device buffers are authoritative;
  callers should use `cpu()`, `sync_data_from_cuda()`, or
  `sync_grad_from_cuda()` before reading host `data` or `grad` directly.
- Host edits to CUDA `Var::data` or `Var::grad` require
  `sync_data_to_cuda()` or `sync_grad_to_cuda()` before later CUDA work.
- Shape metadata is host-side metadata; `cuda()` and `cpu()` preserve it, and
  explicit sync helpers leave it unchanged.

Validation:

- `test_cuda_core` checks 4D shape preservation through `cuda()`/`cpu()`,
  explicit data/grad sync in both directions, gradients observed through
  `cpu()` after backward, and `clear_grad()` clearing host and device gradients.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| CPU behavior regresses while adding CUDA dispatch | Breaks default users | Keep CPU tests mandatory for every phase |
| CUDA op silently reads stale host data | Wrong gradients/results | Unsupported CUDA ops must throw; document sync semantics |
| Elementwise kernels duplicate too much code | Maintenance cost | Use small shared kernel templates where readable |
| CUDA Adam scope grows | Delays CppResist optics work | Treat Adam as optional unless training need is immediate |
| No-device environments fail CI hard | Poor developer workflow | Add explicit skip behavior or document GPU-required CUDA test jobs |
| `minimal_autograd` absorbs CppResist-specific physics | Bad abstraction | Keep FFT/lithography/resist-bank specifics in CppResist |

## Grilled-Me Review

Assumptions confirmed:

- `minimal_autograd` already has real CUDA support through `AUTOGRAD_USE_CUDA`.
- CppResist needs stronger CUDA basics before implementing its own backend.
- FFT/cuFFT is not currently a general `minimal_autograd` capability.

Risks identified:

- Trying to solve FFT/autograd here would overexpand this hardening pass.
- CUDA Adam may be useful but can become a detour.
- Advanced ops could invite a broad port; priority must stay on CppResist
  blockers.

Simplification applied:

- Phase 2 only targets priority elementwise and binary ops.
- Optimizer work is optional unless training use demands it.
- FFT and CppResist physics stay out of this repo for now.

Surviving concerns:

- If CppResist chooses end-to-end differentiable CUDA optics soon, a separate
  FFT/autograd design must decide whether that belongs in `minimal_autograd` or
  CppResist.
