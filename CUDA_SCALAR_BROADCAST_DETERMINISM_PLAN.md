# CUDA scalar-broadcast determinism plan (D3)

## Decision and evidence

CppResist Phase 11c.3b still fails the exact CUDA graph gate before
checkpointing:

```
CUDA Tensor/Variable A/B boundary mismatch at step 1: parameter gradients log_sigma
```

D2.2 fixed a separately demonstrated nondeterministic whole-tensor scalar
sum. Its focused CUDA tests pass repeatedly, yet a hash-verified composite
CppResist diagnostic, built with the D2 provider and no include/macro
workaround, reproduces the failure. Therefore this is not a stale build,
stable/expert target boundary, checkpoint, Adam restore, or tolerance issue.

The Astra read-only preflight traced the actual public-op path:

```text
log_sigma -> sigma_sq -> broadcast to [1,1,9,9]       (A)
          -> exponent -> log(sum(exp(exponent)))
          -> broadcast scalar normalizer to exponent   (B)
          -> Gaussian -> Conv2d -> loss
```

Both A and B dispatch through `cuda_tensor_broadcast_add_backward()` in
`src/cuda/tensor_ops_shape.cu`. Its generic CUDA kernel atomically adds 81
upstream values into a rank-0 scalar. D2.1 tested scalar-first rank-2
`{9,9}` only, so it did not prove this scalar-second rank-4 path exact.
Conv2d weight-gradient remains out of scope: the current kernel is one thread
per output weight with serial accumulation and has no atomic reduction.

## Scope and topology

| Phase | Allowed changed paths | Purpose |
| --- | --- | --- |
| D3.1 | `test/test_cuda_tensor.cpp` | Exact rank-0 broadcast-backward and Gaussian-subgraph diagnosis. |
| D3.2 | `src/cuda/tensor_ops_shape.cu`, `test/test_cuda_tensor.cpp` | Conditional scalar-shape dispatch to the existing fixed-order sum. |
| D3.3 | none in either repository | Hash-verified CppResist composite diagnostic, three fresh GPU processes. |

This plan-only PR has one conditional minimal_autograd implementation PR. It
carries forward the existing uncommitted D2.2 diff unchanged (only
`src/cuda/tensor_ops_shape.cu` and `test/test_cuda_tensor.cpp`) after proving
base `66c6251`, its two-path manifest, and its diff. D2.2 and D3 form one PR
because D3.1 must run after the qualified fixed-order scalar sum. D3.3 is a
cross-repository validation boundary; it is never committed into that PR.

No CppResist source, CMake/public header/dispatcher/optimizer/checkpoint code,
Conv2d, axis reductions, generic broadcast, or tolerance change is in scope.

## D3.1 — right-reason RED

Add only exact CUDA tests to `test/test_cuda_tensor.cpp`:

1. Use a true rank-0 scalar as the **second** broadcast-add input with explicit
   heterogeneous upstream gradients. Cover `{1,1,9,9}` and `{8,8}` planes,
   then both operand orders where public rules permit. Synchronize and compare
   independent A/B scalar-gradient IEEE bits across bounded repetitions.
2. Build the public Gaussian subgraph shown above, with rank-4 shape, fixed
   nonuniform upstream, and at least two stable `log_sigma` seeds. Retain
   intermediate Variables and compare A/B `.value()` and `.grad()` bits. This
   distinguishes a normalizer-scalar difference, `sigma_sq`-scalar difference,
   or an earlier first mismatch without private tracing hooks.

The test reports shape, operand order, pair, iteration, and first bits; it
never applies a tolerance. Existing D2 scalar-sum/scalar-first tests remain as
controls.

Manual owner gate after D3.1:

- RED at either rank-0 scalar boundary authorizes D3.2.
- Both exact, or divergence before those boundaries, stops this plan. No
  Conv2d control or source change follows without a new approved plan.

## D3.2 — conditional GREEN

Only after the D3.1 RED, add this scalar-shape branch after validation in
`cuda_tensor_broadcast_add_backward()`:

```cpp
if (input_shape.rank() == 0) return cuda_tensor_sum(g);
```

It reuses the D2.2 fixed-order device reduction, preserves rank-0 scalar shape
and device residency, and leaves every non-scalar or singleton-shaped input on
the generic kernel. Do not use `elements() == 1`, host staging, a new kernel,
or a general broadcast rewrite.

GREEN on the visible CUDA device requires all D3.1 checks bit-exact, existing
CUDA Tensor parity tests green without relaxed tolerance, rank-0
shape/device and empty-upstream behavior covered, `git diff --check` clean,
and only the retained D2.2/D3 paths changed. A moved first mismatch stops this
plan rather than expanding scope.

## D3.3 — consumer gate

Using the approved CppResist composite procedure, verify the exact
13-file diagnostic manifest and filtered binary-diff SHA-256, apply only its
separate two-line expert-target CMake diff in the disposable composite tree,
and configure a new CUDA build against the D3 provider. No global compiler,
include, or macro override is permitted. Run unchanged
`cppresist_train_resist_cuda` three times in fresh processes on the RTX GPU.

Every run must prove exact pre-checkpoint A/B, exact Adam state/parameters,
and exact resumed continuation. Any failure keeps CppResist 11c.3b blocked and
prevents implementation PR publication.

## Risk, route, review, and stop policy

Risk is L4; difficulty is difficult. The completed Astra preflight is current
candidate evidence. Before D3.2, request one bounded `sol-expert` review of
the live D3.1 RED, then use `agent="luna"` for the approved edit. Trigger one
independent final read-only review before readiness. Maximum three attempts per
phase, ten-minute progress checkpoints, and a 45-minute maximum wait apply.
Usage accounting is observational only.

## Grilled-Me review

Assumptions confirmed: the D2 provider and CppResist composite source were
verified; D2 scalar sum is now focused-test exact; `log_sigma` is the first
actual difference; and both scalar broadcast paths exist in the public graph.

Risks identified: rank-2 scalar-first sampling can falsely clear rank-4
scalar-second reduction; `elements()==1` changes singleton semantics; a
passing isolated test can leave an earlier operation unproven; and stale
consumer sources can invalidate D3.3.

Simplification applied: retain D2.2, test the actual scalar shape/order and a
small public subgraph, reuse the existing sum only after a RED, and preserve
the hash-verified three-process consumer gate.

Surviving manual concern: if D3.1 is exact or a green D3.2 leaves D3.3 RED,
stop for a new first-divergence plan; never broaden this work into a CUDA audit.
