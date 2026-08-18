# R6 Direct Tensor CPU/CUDA Provider Split

Status: draft — plan-only. No implementation is authorized until owner
approval of this exact plan commit.

## Goal

Replace per-operation `#ifdef AUTOGRAD_USE_CUDA` dispatch in the direct
`Tensor` / `Variable` stack with build-selected internal provider sources.

A CUDA-enabled build remains dual-backend: CPU tensors execute the CPU
provider and CUDA tensors execute the CUDA provider. Mixed-device operations
retain their current explicit rejection. A CPU-only build contains no CUDA
toolkit dependency and rejects CUDA tensor construction/transfer at the
existing storage boundary.

```text
AUTOGRAD_USE_CUDA=OFF
  Tensor(cpu) -> CPU provider
  Tensor(cuda) -> existing storage rejection

AUTOGRAD_USE_CUDA=ON
  Tensor(cpu) -> CPU provider
  Tensor(cuda) -> CUDA provider
  CPU + CUDA   -> existing same-device rejection
```

## Roadmap placement

This provider split is the next prerequisite **before** CppResist R5e.1
product implementation. The required order is:

```text
minimal_autograd #69 direct CUDA shape ops (merged)
  -> this R6 provider split and dual-build qualification
  -> CppResist R5e.1 public-API feasibility recheck
  -> CppResist R5e.1 CUDA direct ILT training-core implementation
  -> CppResist R5e.2 execution/state continuation
```

This plan refactors #69's direct CUDA `reshape`, `slice`, and `concat`
support; it must preserve their public behavior, CUDA residency, VJPs, and
existing reflect-border → reshape → CUDA `conv2d` composition. It does not
authorize a CppResist source change or feasibility probe before this plan is
implemented, qualified, and merged.

## Boundary and non-goals

This plan covers only the direct `ag::Tensor` / direct `ag::Variable` stack
used by CppResist. It does **not** remove every macro from the repository.
The separate legacy Eigen/`Var` compatibility stack continues to use its
current guards until a separately approved migration-or-retirement decision.

Excluded:

- `include/autograd/**`, `src/variable.cpp`, `src/ops.cpp`, `src/conv.cpp`,
  `src/loss.cpp`, `src/optim.cpp`, `src/fft.cpp`, `src/cuda_core.cu`, and
  `src/cuda_fft.cu` (legacy stack);
- public API/signature changes, a runtime plugin/registry, new dependencies,
  stream policy changes, and CppResist product changes;
- changes to numerical algorithms, device-transfer semantics, supported-op
  surface, or error categories.

The remaining `AUTOGRAD_USE_CUDA` CMake condition is intentional: it selects
CUDA language/toolkit/linkage and the CUDA-capable provider sources. Direct
operation code must not contain this preprocessor guard after migration.

## Target internal structure

```text
src/detail/tensor_ops.h                 shared direct-op declarations
src/detail/tensor_ops_validation.h      common validation/index helpers
src/cpu/tensor_ops.cpp                  CPU operation implementations
src/core/tensor_dispatch_cpu.cpp        CPU-only dispatch implementation
src/core/tensor_dispatch_cuda.cpp       CUDA-build dispatch implementation
src/cuda/tensor_ops.cu                  CUDA operation implementations
src/core/cuda_runtime_stubs.cpp         CPU-only storage runtime provider
src/cuda/tensor_storage_runtime.cu      CUDA storage runtime provider
```

The CPU-only and CUDA-build dispatcher translation units provide the same
private `detail::tensor_*` symbols. CMake compiles exactly one of them; this
avoids a large CUDA-stub table while keeping common callers free of
preprocessor dispatch.

```cpp
// CUDA-build dispatcher, no #ifdef in the operation body.
Tensor tensor_add(const Tensor& a, const Tensor& b) {
    validate_same_shape_and_device("add", a, b);
    return a.device().is_cuda() ? cuda_ops::add(a, b) : cpu_ops::add(a, b);
}

// CPU-only dispatcher, selected by CMake.
Tensor tensor_add(const Tensor& a, const Tensor& b) {
    validate_same_shape_and_device("add", a, b);
    return cpu_ops::add(a, b);
}
```

No caller may silently copy CUDA values through host storage. Validation stays
in backend-neutral code before dispatch, and CUDA forward/VJP ownership stays
device-resident.

## Approved implementation scope

- `R6_DIRECT_TENSOR_BACKEND_PROVIDER_PLAN.md`
- `CMakeLists.txt`
- `src/detail/tensor_kernels.h`, `src/detail/tensor_ops.h`, and
  `src/detail/tensor_ops_validation.h`
- `src/cpu/tensor_ops.cpp`
- `src/core/tensor_dispatch_cpu.cpp` and `src/core/tensor_dispatch_cuda.cpp`
- `src/detail/tensor_cuda_ops.h`, `src/cuda/tensor_ops.cu`,
  `src/core/cuda_runtime_stubs.cpp`, and `src/cuda/tensor_storage_runtime.cu`
- direct-stack call sites only when their private include changes:
  `src/core/{ops,variable,conv,norm,fft,optim}.cpp`
- `test/test_tensor.cpp`, `test/test_cuda_tensor.cpp`, and
  `test/test_cuda_core.cpp`

Any additional path, public header, legacy path, consumer repository, or API
change stops the phase for owner review.

## Phases and gates

### Phase 1 — source-selection seam

Create the common internal declaration/validation boundary and make CMake
select exactly one future direct dispatcher and exactly one storage runtime
provider. The dispatcher sources are initially empty: existing inline
operations remain authoritative until their individual migration group moves
each symbol exactly once.

Red gate: current CPU and CUDA builds/tests pass before edits.

Green gate:

- CPU-only configure/build links without CUDA toolkit headers/libraries;
- CUDA configure/build links its CUDA sources once, without duplicate symbols;
- the source-selection seam is present without defining a duplicate direct-op
  symbol before Phase 2;
- the static source inventory shows CMake, not direct operation bodies, owns
  the direct-stack build selection.

### Phase 2 — migrate direct operation families

Move direct operation implementations from `tensor_kernels.h` to the CPU
provider and CUDA-capable dispatcher in bounded groups:

1. elementwise, scalar, and activation forward/VJPs;
2. shape, broadcast, reduction, and matmul forward/VJPs;
3. convolution, pooling, and normalization helpers;
4. FFT and optimizer CUDA helper dispatch.

For every group, preserve the exact validation-before-dispatch ordering and
existing device/error behavior. Stop if a family needs a changed public API,
host materialization, or a new CUDA kernel beyond the current supported
surface.

Each group is a manual L3 phase gate; no automatic continuation.

### Phase 3 — direct-stack cleanup and qualification

Remove the old direct-op CUDA include/guard path only after all migrated
families have passed both build modes. Keep legacy guards and the CMake build
selection.

Green gate: `src/detail/tensor_kernels.h` is deleted or reduced to a
macro-free compatibility include, and direct operation bodies in the approved
scope contain no `AUTOGRAD_USE_CUDA` conditional.

## Acceptance matrix

CPU-only build:

```bash
cmake -S . -B build-r6-provider-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-r6-provider-cpu --parallel
./build-r6-provider-cpu/test_core
./build-r6-provider-cpu/test_nn
./build-r6-provider-cpu/test_conv
./build-r6-provider-cpu/test_tensor
```

CUDA-enabled build and visible-device qualification:

```bash
cmake -S . -B build-r6-provider-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake --build build-r6-provider-cuda --parallel
./build-r6-provider-cuda/test_core
./build-r6-provider-cuda/test_nn
./build-r6-provider-cuda/test_conv
./build-r6-provider-cuda/test_cuda_tensor
./build-r6-provider-cuda/test_cuda_core
git diff --check
```

Tests must demonstrate in the same CUDA-enabled executable that CPU tensors
still use CPU behavior, CUDA tensors and gradients remain CUDA-resident, and
mixed-device operations reject without an implicit transfer. Existing direct
CPU/CUDA parity and unsupported-op rejection tests remain required. The merged
direct CUDA `reshape`/`slice`/`concat` parity, VJP, empty-axis concat, and
reflect-border → reshape → CUDA `conv2d` regression coverage must also remain
green.

## Safety, routing, and publication

Safety risk: L3. Implementation difficulty: difficult.

Before Phase 1, run one bounded `sol-expert` preflight on CMake source
selection, duplicate-symbol safety, and the private symbol boundary. Then use
one bounded `luna` implementation session per approved operation group, with
at most three attempts per same blocker. Trigger one independent Claude
read-only review after the first full dual-build qualification, focused on
CPU/CUDA dispatch selection, storage ownership, no-host-fallback behavior, and
VJP correctness.

Use `karpathy` for each code phase, `phase-gated-implementation` for gates and
publication, and `handoff` when pausing between groups. This agent-authored
plan has a `grilled-me` review; `plan-audit` applies only if an external or
materially revised plan replaces it.

Create one draft plan PR containing this exact artifact. Record owner approval
of its commit SHA before implementation. Keep the same PR draft through all
implementation, validation, and review; mark ready only after final
qualification, then request a separate merge approval.

Stop for owner direction if scope expands, direct and legacy paths cannot stay
separate, CPU-only linking needs CUDA, CUDA-enabled CPU tensors change
behavior, a CUDA tensor reaches CPU code without an explicit transfer, or any
acceptance command fails. After this plan merges, hand off only the recorded
minimal_autograd artifact and qualification evidence to the CppResist R5e.1
feasibility recheck.
