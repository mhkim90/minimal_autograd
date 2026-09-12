# CUDA exact reduction determinism plan

## Purpose and decision record

CppResist Phase 11c.3b is blocked by an exact CUDA resume contract.  Its
actual `VariableResistKernelBank` graph was run twice independently on the
RTX 5060 with identical inputs and state.  At training step 1, the forward
loss matched bit-for-bit but the `log_sigma` parameter gradient did not.  The
same first mismatch occurred in two fresh runs.  This precedes checkpoint
write/load, Adam-state comparison, and continuation, so it is not a CppResist
checkpoint-format defect.

The prior minimal_autograd D1 delivery (plan PR #96; test PR #97 at
`a026285`, merged by `4275865`) proves exact behavior for a compact scalar
shared graph only.  It deliberately does not characterize CUDA reductions.
This plan is the separate, narrow follow-up required by the merged CppResist
11c.3b determinism gate.  It authorizes neither a CppResist workaround nor a
checkpoint, optimizer, API, FFT, Abbe, or tolerance change.

## Current evidence and hypothesis

`VariableResistKernelBank::forward()` builds its CUDA graph from the current
minimal_autograd Tensor/Variable operations.  Its Gaussian normalization
computes a scalar `log(sum(exp(exponent)))`, then broadcasts that scalar back
over the Gaussian plane.  The relevant CUDA implementations are:

- `cuda_tensor_sum()` in `src/cuda/tensor_ops_shape.cu`, whose current forward
  kernel atomically adds every input element into one scalar; and
- the scalar-input branch of `cuda_tensor_broadcast_add_backward()` in that
  file, whose general broadcast-backward kernel atomically adds all plane
  gradients into the scalar input gradient.

The latter is the leading hypothesis: it is a backward-only many-to-one
accumulation on the 9x9 Gaussian plane and feeds `log_sigma`; it explains an
exact forward loss with a first gradient mismatch.  This is evidence, not a
license to change both paths speculatively.  `sum_axes_*`, CUDA Conv2d,
optimizer code, CppResist CUDA code, and every non-scalar broadcast case are
out of scope until an exact test proves they participate.

## Scope

| Phase | Allowed changed paths | Purpose |
| --- | --- | --- |
| D2.1 | `test/test_cuda_tensor.cpp` | Add exact-repeat isolation tests for full scalar sum and scalar broadcast-add backward. |
| D2.2 | `src/cuda/tensor_ops_shape.cu`, `test/test_cuda_tensor.cpp` | Conditionally replace only proven scalar many-to-one reductions with a fixed-order CUDA reduction and retain the regression tests. |
| D2.3 | none in this repository | Re-run the existing CppResist actual-graph diagnostic unchanged; classify whether its exact A/B and resume checks are green. |

No CMake, public header, dispatcher, loss, optimizer, Tensor storage, CUDA
Conv2d, `sum_axes_*`, or CppResist source changes are in scope.  In particular,
the implementation must not replace exact checks with a tolerance or change
the v2 checkpoint grammar.

## Delivery topology and dependencies

This is a plan-only PR.  It has one conditional implementation PR for the
coherent reduction regression and correction:

1. D2.1 is an unpublished right-reason RED on that implementation worktree.
   It adds both isolation tests and records whether either fails exactly on the
   visible CUDA device.  It is not a releasable standalone delivery because a
   deliberately failing regression must not merge.
2. A manual owner classification gate follows D2.1.  If the scalar
   broadcast-backward test is RED, the owner may authorize D2.2 on the same
   branch.  If only scalar sum is RED, source scope is narrowed to scalar sum.
   If neither is RED, implementation stops; no source change or broad audit is
   authorized.
3. D2.2 and its now-green tests form one implementation PR.  D2.3 is an
   independent CppResist validation boundary, not a cross-repository commit in
   this PR.  CppResist remains blocked unless D2.3 passes repeatedly.

The plan depends on merged minimal_autograd PR #97 and on the current
CppResist diagnostic evidence.  It does not grant implementation authority;
the implementation approval must name this plan revision, the implementation
branch, and the manual D2.1 classification result.

## Phase D2.1: isolate the reductions (RED)

Add two exact CUDA tests to `test/test_cuda_tensor.cpp`:

1. **Scalar broadcast-add backward.**  Construct two independent, identical
   CUDA scalar Variables and identical 9x9 CUDA planes.  Broadcast-add each
   scalar to its plane, call backward with the same fixed heterogeneous
   upstream gradient directly (do not append `sum()`), synchronize explicitly,
   copy the scalar gradients to host, and compare their IEEE-754 bits.  Repeat
   independent A/B pairs enough to make the observed RTX 5060 discrepancy
   observable rather than treating one matching pair as proof.
2. **Full scalar sum.**  Independently reduce identical,
   cancellation-sensitive CUDA tensors, synchronize explicitly, and compare
   the scalar output IEEE-754 bits across repeated A/B pairs.

The right-reason RED is an exact mismatch from either isolated operation.
The test reports the operation and pair/iteration, never a tolerance.  A
passing D1-style scalar graph is not substitute evidence.

Stop and request a revised plan if both isolation tests are exact while the
unchanged CppResist graph still fails.  The next permitted investigation would
be a separately planned CUDA Conv2d weight-gradient accumulation trace; do
not add it to this work.

## Phase D2.2: conditional fixed-order reduction (GREEN)

Only after D2.1 proves a specific operation RED:

- Replace `sum_kernel` and its `cuda_tensor_sum()` launch only if the scalar
  sum isolation is RED.
- In `cuda_tensor_broadcast_add_backward()`, special-case only a scalar input
  (`input_shape.elements() == 1`) only if the scalar broadcast-backward
  isolation is RED.  Leave general non-scalar broadcasting on its current
  path.

The implementation uses a deterministic fixed-order device reduction: each
thread accumulates a fixed stride locally, a fixed shared-memory tree combines
partials, and exactly one thread writes the scalar.  It must define behavior
for arbitrary input sizes without a host staging path.  Existing dispatch and
public APIs remain unchanged.

GREEN requires all of the following on the visible CUDA device:

- every D2.1 independent A/B comparison is IEEE-754-bit exact over repeated
  runs;
- existing CUDA Tensor tests, including CPU/CUDA sum, mean, broadcast, and MSE
  parity, pass without relaxed tolerances;
- `git diff --check` passes; and
- the changed-file set is exactly the approved D2.2 paths.

If a fixed-order scalar reduction still differs, or if a different first
mismatch emerges, stop rather than expanding to axes, Conv2d, optimizer, or
CppResist code.

## Phase D2.3: consumer validation (manual)

Without editing CppResist, run its existing actual-graph diagnostic on the
RTX 5060 in multiple fresh process invocations.  It must show exact A/B
agreement through all pre-checkpoint steps, exact ordered Adam state and
parameter agreement, and exact interrupted/resumed continuation.  A remaining
`log_sigma`-gradient mismatch keeps CppResist 11c.3b blocked and requires a
new owner-approved narrow plan.

## Risk, routing, and evidence policy

Safety risk is L4 because the result gates CUDA optimizer trajectory and a
downstream checkpoint migration.  Implementation difficulty is difficult.
Use a bounded `sol-expert` read-only preflight before the implementation route
selected by the then-current approved phase policy; do not silently substitute
an unavailable selected route.  Use one triggered independent final review
for CUDA numerical correctness before readiness.  Maximum three attempts per
phase, with ten-minute progress checkpoints and a 45-minute maximum wait.

Controller-side GPU visibility is never a substitute for delegated
visible-device evidence.  Usage accounting is observational only; missing
controller attribution or unresolved provider model data is reported but does
not change the correctness gate.

## Grilled-Me review

Assumptions confirmed: the observed CppResist first mismatch is a
pre-checkpoint `log_sigma` gradient; its VariableResist graph uses core
Tensor/Variable operations; D1 did not cover contended reductions; and the
current CUDA scalar sum and scalar broadcast-backward paths use many-to-one
atomic accumulation.

Risks identified: source inspection alone could misattribute the failure to
an atomic path while CUDA Conv2d is the true producer; a general reduction
rewrite could alter unrelated semantics or performance; and a test that uses a
downstream sum could conflate the two candidates.

Simplification applied: test the two scalar operations independently before
touching source, make each source change conditional on its own RED, and leave
all general broadcast, axes, Conv2d, optimizer, checkpoint, and CppResist code
untouched.

Surviving concerns (manual gates): one or both isolated tests may remain exact
despite the consumer failure, or the fixed-order implementation may expose a
later first mismatch.  Either result stops this plan's implementation scope
and requires a new owner-approved diagnostic boundary.
