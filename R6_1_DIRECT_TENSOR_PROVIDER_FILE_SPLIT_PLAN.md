# R6.1 Direct Tensor provider file split

## Purpose and dependency

This is a source-layout-only follow-up to R6 PR #70.  It may start only after
R6 is merged, or from its merged commit.  It preserves the R6 contract:

- a CPU-only build selects CPU direct-Tensor dispatchers;
- a CUDA-enabled build links both CPU and CUDA providers, and routes each
  direct `Tensor` by `Device` at runtime;
- mixed-device operands remain rejected; no implicit transfer is introduced.

This plan does not change the public API, CMake option names, device semantics,
numerics, operation coverage, legacy Eigen/`Var` code, or CppResist.

## Exact scope

Allowed production paths:

- `CMakeLists.txt`
- `src/core/tensor_dispatch_cpu*.cpp`
- `src/core/tensor_dispatch_cuda*.cpp`
- `src/core/tensor_dispatch_internal.h`
- `src/cpu/tensor_ops*.cpp`
- `src/cpu/tensor_ops_internal.h`
- `src/cuda/tensor_ops*.cu`
- `src/cuda/tensor_ops_internal.cuh`
- `src/detail/tensor_ops.h`
- `src/detail/tensor_cuda_ops.h`
- `tests/**` (only to repair a moved-file build/reference)

No new behavior-specific test is part of this layout refactor.  No changes are
allowed outside these paths without renewed owner approval.

## Target layout

The existing public `ag::detail::tensor_*` dispatcher definitions are divided
into four cohesive groups, selected as a set by CMake:

| Group | CPU dispatcher | CUDA dispatcher | CPU/CUDA provider sources |
|---|---|---|---|
| Elementwise | `tensor_dispatch_cpu_elementwise.cpp` | `tensor_dispatch_cuda_elementwise.cpp` | `tensor_ops_elementwise.cpp` / `.cu` |
| Shape and linear algebra | `tensor_dispatch_cpu_shape.cpp` | `tensor_dispatch_cuda_shape.cpp` | `tensor_ops_shape.cpp` / `.cu` |
| Neural-network kernels | `tensor_dispatch_cpu_nn.cpp` | `tensor_dispatch_cuda_nn.cpp` | `tensor_ops_nn.cpp` / `.cu` |
| Spectral and optimizer | `tensor_dispatch_cpu_spectral_optimizer.cpp` | `tensor_dispatch_cuda_spectral_optimizer.cpp` | `tensor_ops_spectral_optimizer.cpp` / `.cu`, when that provider owns an existing entry point |

Private validation shared by dispatcher groups belongs in the small,
declaration-only `src/core/tensor_dispatch_internal.h`.  CPU and CUDA provider
sharing is limited to small private internal headers for declarations/types;
it must not recreate a monolithic implementation header.  Each source file
owns its group’s public provider entry points and related local helpers.

The old monoliths (`tensor_dispatch_cpu.cpp`, `tensor_dispatch_cuda.cpp`,
`cpu/tensor_ops.cpp`, and `cuda/tensor_ops.cu`) are deleted only after all of
their definitions have moved.  There must be exactly one definition of every
public dispatcher/provider symbol in each selected build.

## Delivery topology

This initiative uses two implementation PRs:

1. **PR #71, after plan approval:** P1 and P2 together. It moves CMake and
   dispatcher ownership, then the CPU provider. These phases share one source
   layout objective, ownership, rollback boundary, and dual-build validation.
2. **A new PR after #71 merges:** P3 only. It splits CUDA providers and carries
   the required CUDA-helper preflight, independent review, and visible-device
   qualification.

The P2→P3 split is required because P3 crosses independent-review and
device-validation boundaries. Do not hide P3 in #71 or create its branch before
the P2 implementation is merged. The plan-only commit in #71 remains draft;
after owner approval it becomes the P1/P2 implementation PR. P3 needs the
plan's declared manual owner gate before its new PR starts.

## Phases and gates

### P1 — CMake and dispatcher split

- **Risk/difficulty:** L2 / standard; route to OpenCode Luna.
- Move public dispatcher wrappers into the four CPU and four CUDA group files.
- Put only cross-group validation declarations/inline helpers in the private
  dispatcher header.
- Change CMake to select all four CPU dispatcher files for CPU builds and all
  four CUDA dispatcher files for CUDA builds.  Do not compile both dispatcher
  sets into one target.
- **Red gate:** the old single dispatcher paths are still referenced by CMake.
- **Green gate:** CPU and CUDA configure/build complete; symbol ownership has
  no duplicate public `tensor_*` definitions; `git diff --check` passes.
- **Manual gate:** owner approval before P2.

### P2 — CPU provider split

- **Risk/difficulty:** L2 / standard; route to OpenCode Luna.
- Split the CPU provider into the same four groups.  Retain the existing
  `cpu_ops::*` declarations and behavior, moving only definitions/helpers.
- **Red gate:** old CPU provider monolith is still compiled or a group has
  unresolved/duplicate provider symbols.
- **Green gate:** CPU build plus `test_core`, `test_nn`, `test_conv`,
  `test_tensor`, and `test_fft` pass; a CUDA build also links and passes its
  CPU-residency paths.
- **Manual gate:** owner approval to start the separate P3 PR after #71 merges.

### P3 — CUDA provider split and final qualification

- **Risk/difficulty:** L3 / difficult.  Obtain one bounded Sol-expert
  preflight on CUDA internal-helper ownership, then route implementation to
  OpenCode Luna.  Trigger one Claude read-only review because source moves can
  break CUDA linkage or CPU fallback behavior without changing interfaces.
- Split CUDA provider entry points and kernels into the same groups.  Keep
  allocation/metadata/launch utilities private and minimal; do not move CPU
  fallback routing into CUDA provider files.
- **Red gate:** CUDA group source fails to compile/link, or a CUDA dispatcher
  no longer falls back to its matching `cpu_ops::*` entry point for CPU data.
- **Green gate:** CPU suite passes; CUDA configure/build succeeds; visible
  device runs of `test_cuda_tensor` and `test_cuda_core` pass without skip;
  `test_fft` passes; `git diff --check` is clean; static checks confirm no old
  monolith is referenced and both dispatcher sets are not compiled together.
- **Manual gate:** final controller gate, then mark the PR ready.  Separate
  owner merge approval remains required.

## Acceptance criteria

1. A CPU-only build contains the CPU provider plus only CPU dispatcher group
   files; it has no CUDA provider linkage.
2. A CUDA-enabled build contains CPU and CUDA providers plus only CUDA
   dispatcher group files.  A CPU tensor reaches `cpu_ops::*`; a CUDA tensor
   reaches `cuda_*`; mixed devices still throw.
3. `tensor_dispatch_*` and provider source files are grouped as above, with no
   replacement mega implementation file or cross-group duplicate definition.
4. Existing CPU/CUDA tests retain their current behavior and numerical results.
5. The R6 CUDA reshape/slice/concat VJP and reflect-border → reshape → CUDA
   `conv2d` regression remain covered by the CUDA core suite.

## Validation commands

```bash
cmake -S . -B build-r61-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-r61-cpu --parallel
./build-r61-cpu/test_core
./build-r61-cpu/test_nn
./build-r61-cpu/test_conv
./build-r61-cpu/test_tensor
./build-r61-cpu/test_fft

cmake -S . -B build-r61-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-r61-cuda --parallel
./build-r61-cuda/test_core
./build-r61-cuda/test_nn
./build-r61-cuda/test_conv
./build-r61-cuda/test_fft
./build-r61-cuda/test_cuda_tensor
./build-r61-cuda/test_cuda_core
git diff --check
```

If no CUDA device is visible, compilation and CPU-residency checks may be
published, but the PR stays draft until the visible-device tests run without
skips.

## Publication and stop rules

Keep #71 as the plan-only draft PR targeted at `main`. Owner approval must cite
this plan commit SHA; it authorizes P1 only, not merge. After P1 and P2 pass
their manual gates, publish them to #71 and merge only with separate owner
approval. Create the P3 PR only from merged #71 after its declared owner gate.
Stop on scope expansion, duplicate/unresolved symbols, changed behavior,
absent CUDA route, topology deviation, or a failed final device gate.

## Grilled-Me preflight

Assumptions confirmed: R6’s CPU/CUDA runtime routing is the behavior to retain,
and this work depends on that branch being merged.  Simplification applied:
there is no new backend interface, build option, test matrix, or behavior
change.  Main risk: source moves can silently break CUDA linkage or CPU
fallbacks; exact build selection, static ownership checks, dual builds, and
visible-device tests are mandatory.  Surviving concern: CUDA internal helpers
are more coupled than CPU helpers, which is why P3 has a bounded CUDA
preflight, independent review, and its own implementation PR rather than
sharing #71's rollback boundary.
