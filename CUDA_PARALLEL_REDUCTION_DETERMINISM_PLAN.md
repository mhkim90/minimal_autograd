# CUDA Deterministic Parallel Reduction Plan

## Goal

Keep the run-to-run exactness fixed by the D2/D3 scalar-reduction work while
removing the serial full-tensor reduction bottleneck for production-sized CUDA
tensors.  This is a performance follow-up, not a re-diagnosis of the D3
rank-zero broadcast bug.

## Dependency and numerical contract

This work starts only after minimal_autograd PR #100 (commit `df90141`, or its
merged equivalent) is on `main`.  Until then, that PR remains the correctness
baseline: `cuda_tensor_sum` is a serial left fold and rank-zero broadcast
backward reuses it.

The new large-input reducer must be bit-exact across fresh processes for the
same binary, GPU, input bytes, device, and launch configuration.  It is **not**
required to be bit-identical to the serial left fold, to another CUDA toolkit,
or to another GPU architecture.  Floating-point tree order is part of this
implementation's numerical behavior and must be explicit in source and tests.

Small inputs at or below the selected cutoff retain the serial left fold.  In
particular, the current D3 `{8,8}` and `{1,1,9,9}` semantic-oracle tests remain
bitwise equal to their CPU left-fold expectation.  No existing exact oracle is
weakened.

## Non-goals

- No generic broadcast-backward rewrite; only the existing rank-zero route
  continues to use `cuda_tensor_sum`.
- No changes to `cuda_tensor_sum_axes`, Conv2d, optimizers, CPU code, public
  API, allocator/cache, stream-pool, CMake dependency, or CppResist source.
- No atomic-based per-block finalization, Thrust reduction, or cooperative-grid
  launch.
- No claim of speedup without recorded end-to-end measurements.

## Delivery topology

- One plan-only draft PR: this document.
- One implementation PR after #100 is merged, containing P1--P4 below.
- Split boundary: #100 is independently releasable correctness repair; this
  plan changes the numerical reduction tree and needs separate performance and
  rollback evidence.

## Scope

Implementation PR paths are restricted to:

- `src/cuda/tensor_ops_shape.cu`
- `test/test_cuda_tensor.cpp`

Plan PR path is restricted to this document.  Build directories and benchmark
logs are local evidence only and are never committed.

## Design selected for qualification

For inputs above the cutoff, use a dependency-free, fixed two-stage reduction:

1. A fixed-size CUDA block owns one contiguous, indexed input tile.  Each
   thread consumes a fixed sequence of that tile; a fixed shared-memory tree
   reduces those thread partials; block `b` writes exactly partial `b`.
2. One final CUDA thread consumes partials in increasing block-index order and
   writes the rank-zero result.

There are no atomics in either stage.  GPU block scheduling cannot change the
addition graph because partial locations and the final consumption order are
fixed.  A temporary CUDA `Tensor` owns the indexed partial array; it is local
to the call and uses the existing storage lifecycle.  The initial
implementation does not add a cache or asynchronous allocator.

CUB `DeviceReduce::Sum` is a documented same-GPU deterministic alternative,
but is not selected initially: its temporary-storage query/allocation changes
the hot-path ownership model and its result is not the current serial oracle.
It is a stop-and-replan alternative only if the fixed-tree implementation
cannot meet the gates below.

## Phases

### P1 -- baseline harness and cutoff decision (test only)

Risk: L2 performance/numerical behavior.  Difficulty: standard.  Route:
`agent="luna"`.

Add a test-binary-only `cuda_tensor_sum` CUDA-event benchmark mode (not run by
the ordinary test suite and not a new executable).  It measures warmed
reduction latency, including the existing output allocation, at `64`, `81`,
`255`, `256`, `257`, `1024`, `4096`, `65536`, and one production-sized tensor
shape taken from the CppResist training path.  It emits a small, parseable local
report and is never committed as a performance assertion.  Use
cancellation-heavy and ordinary input separately.  Record fresh-process
repeat bits and complete train-step time where the consumer target is
available.

Acceptance gate: only `test/test_cuda_tensor.cpp` changes; ordinary tests still
pass; measurements identify a cutoff that keeps current D3 sizes on the serial
side.  The initial candidate is `256` elements, but it becomes binding only
when measurement is recorded.

Manual owner gate: approve the recorded cutoff and continuation to P2.  Stop
if the consumer workload is not materially above the cutoff or measurements
cannot be reproduced.

### P2 -- parallel-path RED and test contract (test only)

Risk: L2.  Difficulty: standard.  Route: `agent="luna"`.

P1 freezes the cutoff only.  At P2 entry, the controller records the initial
fixed-tree configuration of `256` threads per block, `4` contiguous items per
thread, and contiguous indexed tiles (`1024` elements per full tile).  This
choice is a correctness contract for P2/P3, not a performance claim; changing
it later requires a plan revision.  Add a CPU mirror of that exact fixed tree
and select a cancellation-heavy input above the cutoff whose fixed-tree bits
differ from the serial left fold.  The new assertion is therefore RED against
the current serial implementation and locks the selected addition graph
without an internal production test hook.
Preserve existing small-input left-fold oracles unchanged.  New large-path
tests cover:

- empty and rank-zero input;
- `31/32/33`, `63/64/65`, `81`, cutoff minus/at/plus one, tile boundaries,
  non-multiples, and a production-sized shape;
- cancellation-heavy and mixed-magnitude data;
- rank-zero output shape and CUDA device;
- repeated bit equality across fresh allocations and at least three fresh
  processes.

For large inputs, assert a documented fixed-tree CPU oracle where practical or
an exactly representable sum plus a separately bounded numerical-accuracy
check.  Never compare a different parallel tree to the serial left fold
bit-for-bit.

Acceptance gate: the deliberate tree-vs-left-fold case is RED before the
reducer exists; cutoff/path tests are unambiguous; no tolerance is relaxed and
no old D3 assertion is altered.

### P3 -- fixed-tree implementation and CUDA qualification

Risk: L3 numerical/runtime.  Difficulty: difficult.  Route: bounded
`sol-expert` preflight, then `agent="luna"`; maximum three attempts, with a
materially different approach required before attempt three.

Implement the selected cutoff and two-stage reducer solely in
`tensor_ops_shape.cu`.  Preserve the serial path at/below cutoff and keep the
existing rank-zero broadcast condition unchanged.  Use `finish_kernel` after
each launch and existing CUDA error/device conventions.  Do not introduce
atomics, recursion, allocator caching, CUB, or a third reduction stage.

Green gate:

- `git diff --check` and exact-path scope pass;
- full `test_cuda_tensor` passes in at least three fresh CUDA processes;
- all new large-path repeat tests are bit-exact in those processes;
- small D3 serial-oracle tests remain exact;
- timing is reported for the P1 size set and complete train step, without a
  fabricated performance target.

Stop if the partial array needs another stage for the selected production size,
if temporary allocation dominates the measured path, or if a fresh-process bit
mismatch occurs.  Report evidence; do not silently add CUB or alter the
contract.

### P4 -- consumer and independent review

Risk: L3 cross-repository numerical trajectory.  Difficulty: standard.  Route:
`agent="luna"` for validation; Claude read-only review before readiness.

Build CppResist against the exact implementation worktree and run the actual
CUDA shared-graph diagnostic three fresh processes.  It must retain exact
parameter, optimizer-state, checkpoint-restore, and resumed-trajectory
equality.  Inspect the minimal_autograd diff and the pre-existing CppResist
diagnostic worktree before and after; do not edit CppResist source.

Acceptance gate: all three consumer runs pass, the independent reviewer finds
no scope/semantic/test masking issue, and the implementation PR remains within
the two declared paths.

## Validation and publication policy

Before every phase: inspect branch/dirty state, pause files, active sessions,
and the right-phase RED evidence.  Use ten-minute progress checkpoints and a
45-minute maximum wait per provider role.  A repeated identical blocker stops
after two attempts.

P1 has a manual continuation gate.  P2--P4 may continue only inside this
approved envelope.  No commit, push, implementation PR, ready-for-review
change, or merge occurs without the phase evidence and the separately required
owner approval.  A final implementation PR remains draft until P4 is green and
independent review is GO; merge always needs a separate owner approval.

## Rollback

The implementation is one revertible PR.  Reverting restores #100's serial
deterministic reducer and rank-zero broadcast fix without touching consumer
source or public APIs.
