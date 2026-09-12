# CUDA Exact-Trajectory Determinism Prerequisite

Status: draft — plan-only. No product edit is authorized until owner approval
is recorded against this exact plan commit.

## Goal and current evidence

Resolve the uncertainty exposed by the blocked CppResist 11c.3b CUDA resume
test without assuming that `minimal_autograd` is the defect. The reported
symptom is an exact-resume mismatch after two independently executed CUDA
training trajectories. CppResist has already shown that the restored model
values, Adam hyperparameters, step count, and all saved moment tensors compare
exactly at the checkpoint boundary. It has not yet compared the two live
trajectories at the interruption step.

`minimal_autograd` already tests CUDA Adam CPU-parity and a basic
`state()`/`load_state()` continuation in `test/test_cuda_tensor.cpp`, but it
does not establish exact equality between independent CUDA executions of a
shared Variable graph at the boundaries needed to classify this symptom.

This prerequisite adds that missing characterization only. It does not claim
that the present CUDA backend is nondeterministic, does not change CppResist,
and does not alter a checkpoint grammar.

## Frozen contract

- The direct Tensor/Variable public API, CPU behavior, CUDA device-residency
  rules, `AdamState` layout, and CppResist source remain unchanged.
- Exact means identical IEEE-754 `float` values after explicit test-only host
  copies. `1e-5` CPU/CUDA parity tolerance is not an acceptable substitute for
  equality between two same-device CUDA executions.
- The diagnostic uses only existing public direct Tensor/Variable/Adam APIs.
  Its host copies are observation points in the test, never a product data
  path, fallback, synchronization workaround, or new public API.
- A synchronization call that merely hides an ordering defect is not a fix.
  If a CUDA operation is proved nondeterministic, the corrective delivery must
  make its computation deterministic and preserve existing numerical/API
  behavior.
- Do not weaken CppResist exact-resume acceptance, change checkpoint v2 bytes,
  threshold/round moments, introduce a CPU bridge, add raw/public CUDA access,
  change CMake, or make unrelated CUDA cleanup in this work.

## Approved scope and delivery topology

This plan PR changes only:

- `CUDA_EXACT_TRAJECTORY_DETERMINISM_PLAN.md`

After this plan merges, the initiative has these separately gated deliveries:

| Delivery | PR | Exact scope | Why it is separate |
| --- | --- | --- | --- |
| D1: CUDA first-divergence characterization | one implementation PR | `test/test_cuda_tensor.cpp` | Observation-only test code; no backend semantics change. |
| D2: corrective plan, only if D1 proves a minimal_autograd boundary failure | one later plan-only PR | exact sources/tests selected from D1 evidence | The affected operation/kernel or optimizer seam is unknown now. |
| D3: corrective implementation, only after D2 merges | one implementation PR | D2's approved exact scope | A deterministic algorithm change has different rollback and review risk. |

CppResist 11c.3b remains a single atomic CppResist implementation PR. It is
not split here, and it does not resume product edits until D1 evidence has
been reviewed at the manual boundary below. If D1 passes, CppResist must add
its own observation-only first-divergence comparison before deciding whether
its blocked resume failure is downstream of `minimal_autograd`.

All source files, headers, CMake files, legacy `ag::Var` files, and CppResist
paths are forbidden in D1. In particular, the known `atomicAdd` use in CUDA
shape reductions is a candidate to investigate only if D1 points there; it is
not pre-authorized as the cause or as an edit target.

## D1 — CUDA first-divergence characterization

Safety risk: L3 (CUDA numerical/state correctness test contract).
Implementation difficulty: difficult. Route: one bounded `sol-expert`
read-only preflight, then one `luna` implementation session. Trigger one
asynchronous Claude review for CUDA numerical determinism and test adequacy.
Checkpoint every 10 minutes; maximum wait 30 minutes; maximum three attempts.
After two identical blockers, stop and use the Sol finding before a materially
different third attempt.

### D1 RED gate

Before editing, record that the current test has only tolerance-based
CPU/CUDA parity and one-state continuation coverage, not this independent
same-device trajectory contract:

```bash
rg -n -C 3 'test_oop_adam_cuda_step_parity_and_moments|\
test_oop_adam_cuda_load_state_preserves_device|check_close' \
  test/test_cuda_tensor.cpp
rg -n -C 3 'optimizer_adam_step|Adam::step|Adam::state|Adam::load_state' \
  src/core/optim.cpp src/core/tensor_dispatch_cuda_spectral_optimizer.cpp \
  src/cuda/tensor_ops_spectral_optimizer.cu
rg -n -C 3 'sum_kernel|sum_axes_kernel|atomicAdd' \
  src/cuda/tensor_ops_shape.cu
```

The right-reason external RED evidence is CppResist 11c.3b's current exact
resume mismatch. D1 is deliberately a characterization test: it may pass and
therefore must not manufacture a failing backend assertion merely to satisfy
a red/green ritual.

### D1 implementation

1. Add private test-only snapshot and exact-comparison helpers in
   `test/test_cuda_tensor.cpp`. On a mismatch they identify, in this order:
   trajectory/step, pre-step loss, parameter gradient and element, post-step
   parameter and element, first/second Adam moment and element, or step count.
   They never use a tolerance and are not added to a public header.
2. Add a fixed, no-randomness CUDA shared-graph training fixture using only
   existing direct APIs. It must have multiple trainable Variables and route at
   least one trainable value through more than one differentiable branch before
   a scalar loss, then run several `zero_grad -> forward -> backward ->
   Adam::step` iterations. Keep the fixture small and row-major; it is a
   backend characterization, not a copied CppResist resist implementation.
3. Execute two independently constructed fixture trajectories on the same
   visible CUDA device. At every step compare exactly, after test-only
   observation copies:
   - scalar loss before `backward()`;
   - each parameter gradient immediately after `backward()`;
   - each parameter value plus Adam `first_moments`, `second_moments`, and
     `step_count()` immediately after `step()`.
4. At a fixed interruption step, take the existing `Adam::state()` snapshot,
   construct fresh CUDA Variables from test-only copies of the interrupted
   parameter values, and call `load_state()` on a fresh Adam. Continue that
   restored trajectory with the same fixed input sequence and compare every
   boundary exactly with the uninterrupted reference. This is an in-memory
   native-state test; it neither serializes a checkpoint nor claims to test
   CppResist persistence.
5. Keep and extend the existing CUDA Adam parity/load-state tests only where
   needed to avoid duplicated setup. Do not change an operation implementation,
   add logging in production code, or alter the current tolerance-based
   CPU/CUDA parity contract.

### D1 GREEN / acceptance

The visible-device qualification is performed through OpenCode; a controller
environment skip is not acceptance evidence.

```bash
cmake -S . -B build-cuda-determinism -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda-determinism --target test_cuda_tensor --parallel 2
./build-cuda-determinism/test_cuda_tensor
./build-cuda-determinism/test_cuda_tensor
git diff --check
```

Acceptance requires a visible CUDA device, both full binary runs passing
without a skip, and exact equality at every listed boundary in both runs. The
final diff must contain only `test/test_cuda_tensor.cpp` plus this already
merged plan artifact in its historical plan PR.

## D1 classification and manual boundary

Publish D1 after its test/validation/review gates. Then stop for owner review;
do not edit a CUDA provider in the same PR.

| D1 result | Meaning | Next action |
| --- | --- | --- |
| Parameter gradients first differ | A direct graph/backward operation is implicated. | Record the first operation boundary; draft D2 naming only that dispatch/kernel path and focused regression test. |
| Gradients match, but Adam values/moments first differ | Adam mutation/state path is implicated. | Draft D2 around the exact `optim.cpp`/optimizer provider boundary identified by the test. |
| Inline trajectories match, but native-state continuation first differs | Adam state clone/load or device-storage transfer is implicated. | Draft D2 with exact state/storage sources demonstrated by the test. |
| All D1 boundaries match exactly | No minimal_autograd defect has been established. | Do not modify minimal_autograd sources. Resume CppResist only to add its actual-graph first-divergence observation; classify there before proposing a downstream fix. |

A D1 failure report must contain the first failing field and step, device and
CUDA build configuration, and whether repeated process execution reproduced
it. A pass report must say explicitly that it characterizes the compact
minimal graph, not the whole CppResist graph.

## D2/D3 conditional corrective delivery

Only the matching D1 classification authorizes drafting D2. D2 must be a new
plan-only PR that names exact affected source/test paths, a deterministic
algorithm, its backward/state semantics, a right-reason regression RED gate,
and a visible-device exact GREEN gate. It must not expand to all CUDA
reductions, all autograd operations, or CppResist simply because they are
possible candidates.

D3 follows only after the D2 plan SHA is owner-approved and merged. It is a
separate implementation PR with a fresh source-level CUDA review. Its
acceptance must retain D1 exact trajectory/state continuation, existing CUDA
tests, and the relevant CppResist exact-resume gate after the upstream revision
is consumed. A tolerance relaxation, checkpoint grammar change, or host
fallback is a stop condition, not a corrective option.

## Stop rules, publication, and approvals

Stop and seek a revised plan if D1 needs a public API, production
instrumentation, CppResist changes, a serializer, a new dependency, an
unlisted test target, no CUDA device is visible through the delegated
qualification, or the failure cannot identify a first differing boundary.

Publish this plan as a draft plan-only PR. Owner approval of the exact plan
SHA is neither approval to merge this plan nor approval to run D1, D2, D3, or
CppResist implementation. After this plan has separately merged, D1 still
requires its own exact owner implementation authorization. Each implementation
PR stays draft through its declared validation and review; owner merge approval
remains separate.

## [Grilled-Me Review]

Assumptions confirmed: current direct CUDA Adam has a state/load test and the
provider seams are `src/core/optim.cpp`,
`src/core/tensor_dispatch_cuda_spectral_optimizer.cpp`, and
`src/cuda/tensor_ops_spectral_optimizer.cu`; CUDA shape reductions contain
`atomicAdd`, but no evidence yet links them to the blocked product test.

Risks identified: a compact fixture can pass while the full CppResist graph
still diverges; an apparent checkpoint symptom may predate persistence; and a
host observation could be mistaken for a product fallback.

Simplification applied: D1 changes one existing CUDA test file only and uses
the native public `AdamState` rather than inventing serialization or copying
the CppResist model.

Surviving concerns (flagged to owner): D1 is a classifier, not proof that the
actual CppResist graph is clean. A passing D1 requires CppResist's own
first-divergence observation before any downstream fix; a failing D1 requires
a new evidence-bound D2 plan before source edits.
