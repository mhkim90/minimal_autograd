# R6 Reconciliation: Superseded Provider-Split Branch

Status: draft — plan-only. This plan authorizes neither a rebase nor a source
change until the owner approves this exact plan commit.

## Decision to resolve

Minimal-autograd PR #81 (`codex/r6-phase3-luna`, head `168af7d`) cannot merge
because current `main` already contains later R6 provider-file splits. The
old branch modifies the pre-split files; `main` has replaced them with grouped
CPU/CUDA dispatcher and CPU-provider files. A normal rebase would therefore
delete or overwrite current provider work.

This plan is pinned to current-main revision `f6def26`. A newer `main`
revision is a stop condition, not an implicit rebase target.

The first question is not how to resolve conflicts. It is whether every
behavioral addition in `168af7d` already exists on current `main` with the
same CPU/CUDA dispatch and VJP behavior. If yes, #81 is superseded and must be
closed rather than rebased. If no, this plan stops with a precise gap inventory
and requires a new implementation plan.

## Current evidence

- #81 is open and ready, but GitHub reports merge state `DIRTY`.
- The old branch changes only five pre-split files:
  `src/core/tensor_dispatch_cpu.cpp`,
  `src/core/tensor_dispatch_cuda.cpp`,
  `src/cpu/tensor_ops.cpp`, `src/detail/tensor_kernels.h`, and
  `src/detail/tensor_ops.h`.
- Current `main` provides the same `tensor_ones`, softmax, softmax-backward,
  log-softmax, and log-softmax-backward symbols in the newer grouped files:
  `src/core/tensor_dispatch_{cpu,cuda}_elementwise.cpp` and
  `src/cpu/tensor_ops_elementwise.cpp`.
- The old branch passed fresh CPU/CUDA qualification and an independent review,
  but that evidence does not qualify current `main` after the newer file split.

## Scope

This plan permits only evidence gathering and qualification on current `main`:

- `CMakeLists.txt`
- `src/detail/tensor_ops.h`
- `src/core/tensor_dispatch_{cpu,cuda}_elementwise.cpp`
- `src/cpu/tensor_ops_elementwise.cpp`
- direct-stack callers needed to trace the five listed operations:
  `src/core/{ops,variable}.cpp`
- `test/test_tensor.cpp`, `test/test_cuda_tensor.cpp`, and
  `test/test_cuda_core.cpp`

It does not permit edits to those files, a rebase, conflict resolution, API
changes, legacy-stack changes, CppResist changes, or cleanup beyond the
explicit disposition of PR #81.

## Delivery topology

One plan-only draft PR records this plan. There is no implementation PR in the
no-gap path because no source change is permitted or needed. If a behavioral
gap is found, stop and create a separate plan-only PR for that exact gap;
do not reuse #81 or this plan to implement it.

PR #81 remains open during evidence gathering. Closing it is a separate owner
action after all gates are green; this plan never treats plan approval as close
or merge approval.

## Phases and gates

### Phase 1 — semantic successor inventory

Map each old-branch addition to current `main`:

1. `tensor_ones`;
2. softmax and softmax backward;
3. log-softmax and log-softmax backward;
4. CPU/CUDA runtime dispatch and validation ordering;
5. corresponding direct `Variable` forward/VJP callers and tests.

Red gate: prove that a textual file-path match is insufficient because the old
files no longer exist on `main`.

Green gate: a symbol-by-symbol inventory shows current-main ownership, CPU
fallback behavior in CUDA builds, CUDA residency, validation ordering, and
test coverage. Any missing symbol, changed error category, host materialization,
or untested VJP is a gap and stops this plan.

Manual gate: owner reviews the inventory before qualification begins.

### Phase 2 — current-main qualification

Use a clean detached worktree at the recorded current-main SHA. Do not use a
stale build directory.

CPU matrix:

```bash
cmake -S . -B build-r6-reconcile-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-r6-reconcile-cpu --parallel
./build-r6-reconcile-cpu/test_core
./build-r6-reconcile-cpu/test_nn
./build-r6-reconcile-cpu/test_conv
./build-r6-reconcile-cpu/test_tensor
```

CUDA matrix (with the toolkit path explicitly exposed):

```bash
cmake -E env PATH=/usr/local/cuda/bin:$PATH \
  cmake -S . -B build-r6-reconcile-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake -E env PATH=/usr/local/cuda/bin:$PATH \
  cmake --build build-r6-reconcile-cuda --parallel
./build-r6-reconcile-cuda/test_core
./build-r6-reconcile-cuda/test_nn
./build-r6-reconcile-cuda/test_conv
./build-r6-reconcile-cuda/test_cuda_tensor
./build-r6-reconcile-cuda/test_cuda_core
git diff --check
```

Green gate: every command passes. CUDA evidence must show CPU tensors retain
the CPU provider, CUDA tensors and gradients remain CUDA-resident, and mixed
devices reject without transfer.

Manual gate: any failure, newer `main` movement, or source change stops for
owner direction.

### Phase 3 — independent review and disposition

Trigger one independent Claude read-only review of the current-main
elementwise provider files and direct callers. It must focus on single-provider
CMake selection, no host fallback, VJP residency, and duplicate-symbol safety.

If Phases 1–3 are green, present evidence and request a separate owner approval
to close #81 as superseded. Do not merge #81. If a gap exists, stop and propose
a new narrow implementation plan.

## Risk, routing, and stop rules

Safety risk is L3; difficulty is standard validation/analysis. Use one bounded
Luna session for each qualification phase and a Claude read-only review after
the dual-build matrix. At most three attempts per same blocker; after two,
request a bounded Sol-expert diagnosis. Controller usage attribution remains
unavailable unless the host exposes an exact context window ID before a new
provider workflow starts.

Stop immediately for an incomplete inventory, behavior mismatch, source edit,
failed test, CUDA-toolkit absence, current-main movement, review blocker, or
scope expansion. No automatic continuation crosses a manual gate.

## Grilled-Me review

Assumptions confirmed: the old branch's five exported additions are declared
and implemented in the grouped current-main elementwise providers; a
file-level conflict is therefore not evidence of a missing behavior.

Risks identified: the newer file split could have changed a VJP, CPU fallback,
or CUDA residency despite matching symbol names; a new `main` commit could
invalidate the comparison. Both are explicit Phase 1/2 gates.

Simplification applied: no rebase or conflict-resolution phase is planned.
The no-gap path validates current main and asks the owner to close the obsolete
PR. Any gap creates a new narrow plan rather than expanding this one.

Surviving concern: qualification cannot prove semantic succession until the
symbol inventory and fresh dual-build matrix both pass.
