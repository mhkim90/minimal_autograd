# CUDA Device Predicate and Selection Prerequisite for CppResist R5e.1

Status: draft — plan-only. No product edit is authorized until owner approval
is recorded against this exact plan commit.

## Goal and evidence

Add the smallest public direct-`ag::Variable` predicate, selection, and
device-validation surface needed to resume the exact CppResist R5e.1 CUDA ILT
training-core plan without host materializing graph tensors.

The resumed R5e.1 feasibility preflight confirmed that CUDA `slice`/`concat`
and CUDA `conv2d` now permit device-resident reflect smoothing. It stopped on
two remaining public-API gaps:

1. The resist model chooses its exact Gaussian support from `log_sigma`:
   `half = min(max(1, int(exp(log_sigma) * 4)), max_kernel_half)`. Tensor
   shape cannot be selected from a device scalar, and the existing public
   direct API has no predicate/selection primitive.
2. CUDA finite and positive checks cannot preserve the R5e.1 rejection
   contract without a public device-validation status operation. Reading graph
   values with `Tensor::copy_to_host` is forbidden in that product path.

The prerequisite preserves exact support without a dynamic shape: CppResist
will later construct a fixed `[-max_kernel_half, max_kernel_half]` grid and
select weights where `abs(offset) <= clamp(4 * exp(log_sigma), 1,
max_kernel_half)` before normalization. The existing `relu` composition can
express the scalar clamp; no new min/max operation is needed. For integral
offsets this selects precisely the current clamped `floor(4*sigma)` support;
zeroed outer weights make fixed reflect padding numerically equivalent to the
current variable-width convolution.

This plan does not implement CppResist. Its green result is a public upstream
capability plus an isolated CUDA composition test; CppResist R5e.1 resumes
only after its own plan gate is rechecked.

## Frozen public contract

- Add `ag::less_equal(const Variable&, const Variable&)`, returning a
  same-shape, same-device float mask (`1.0f` when `a <= b`, else `0.0f`). It
  validates the existing binary exact-shape/device contract. It is
  nondifferentiable: its output never propagates a gradient to either input.
  Equality is true; NaN comparisons are false.
- Add `ag::where(const Variable& condition, const Variable& when_true,
  const Variable& when_false)`, returning the element from `when_true` where
  the condition is nonzero and from `when_false` otherwise. All three inputs
  require equal shape and device. Backward sends upstream gradient to the
  selected value input and a zero gradient to the condition. Selection uses
  the condition's value only; it has no gradient through the branch choice.
- Add `ag::all_true(const Tensor&)` and `ag::all_finite(const Tensor&)` as
  explicit synchronous control-plane predicates. They return a host `bool`,
  not a Tensor or Variable. CUDA implementations reduce on the selected
  device and copy only an opaque status flag needed for the return value; they
  never expose or materialize a graph tensor through `copy_to_host` or
  `to(Device::cpu())`. Empty input is true for both operations.
- CPU and CUDA results use the same semantics. `all_true` treats every
  nonzero finite value as true; zero and NaN are false. `all_finite` uses
  `std::isfinite`/CUDA `isfinite` semantics.
- Direct graph tensors and gradients remain device-resident. The two bool
  APIs are deliberately explicit synchronization points for validation, not
  hidden transfers or autograd nodes.
- Do not add `floor`, integer tensors/dtypes, dynamic shapes, arbitrary
  comparisons, reduction APIs, implicit host fallback, legacy `ag::Var`
  bridging, raw/public CUDA pointers, or CppResist changes.

## Approved scope

- `CUDA_R5E1_DEVICE_PREDICATE_SELECTION_PLAN.md`
- `include/autograd/core/ops.h`
- `src/core/ops.cpp`
- `src/core/variable.cpp`
- `src/detail/tensor_ops.h`
- `src/detail/tensor_cuda_ops.h`
- `src/core/tensor_dispatch_cpu_elementwise.cpp`
- `src/core/tensor_dispatch_cuda_elementwise.cpp`
- `src/cpu/tensor_ops_elementwise.cpp`
- `src/cuda/tensor_ops_elementwise.cu`
- `test/test_tensor.cpp`
- `test/test_cuda_tensor.cpp`

`CMakeLists.txt`, legacy `include/autograd/ops.h`/`src/cuda_core.cu`, module
and optimizer code, all CppResist paths, and every other source are forbidden.
The existing CPU/CUDA tensor test targets already compile these files; stop
for a revised plan if that proves false.

## Single implementation phase — predicate, selection, and validation

Safety risk: L3 (public API plus CUDA numerical/state correctness).
Implementation difficulty: difficult. Route: bounded `sol-expert` preflight,
then one Luna implementation session. One asynchronous Claude review is
required for CUDA/API/autograd correctness. Checkpoint every 10 minutes;
maximum wait is 30 minutes; at most three implementation attempts.

### RED gate

Before editing, record the absence of the public operations and the existing
elementwise dispatch/VJP seams:

```bash
rg -n -C 2 'less_equal|all_true|all_finite|where\(' \
  include/autograd/core/ops.h src/core/ops.cpp src/core/variable.cpp
rg -n -C 2 'enum class OpKind|cuda_backward_supported|case OpKind' \
  include/autograd/core/variable.h src/core/variable.cpp
rg -n -C 2 'tensor_(relu|mul|div)|cuda_tensor_(relu|mul|div)' \
  src/detail/tensor_ops.h src/detail/tensor_cuda_ops.h \
  src/core/tensor_dispatch_cpu_elementwise.cpp \
  src/core/tensor_dispatch_cuda_elementwise.cpp \
  src/cpu/tensor_ops_elementwise.cpp src/cuda/tensor_ops_elementwise.cu
cmake -S . -B build-r5e1-predicate-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake --build build-r5e1-predicate-cuda --parallel
./build-r5e1-predicate-cuda/test_cuda_tensor
```

Expected baseline: no new public functions; existing CUDA tensor tests pass on
a visible device. If a visible CUDA device is unavailable, stop rather than
claim CUDA qualification.

### Implementation steps

1. Add private CPU/CUDA Tensor helpers for elementwise `less_equal` and
   `where`, plus CUDA device reductions for `all_true` and `all_finite`.
   Reuse the existing row-major dense elementwise provider pattern; validate
   shapes and devices in the existing dispatch layer before CUDA launch.
2. Add the public Variable operations and their `OpKind`/backward handling.
   `less_equal` produces a leaf-like nondifferentiable mask; `where` records
   the condition and returns zero condition gradient plus selected-value
   gradients. Preserve repeated-backward and `zero_grad` behavior.
3. Add the explicit Tensor bool wrappers. CUDA returns only one opaque status
   flag after a device reduction and synchronization; CPU scans locally. Do
   not make them Variables, attach them to the tape, or use them in an
   implicit operation path.
4. Add focused CPU and CUDA tests. Include an isolated public-operation-only
   fixed-grid Gaussian construction whose `less_equal`/`where` support mask
   matches a host reference on both sides of every `4*sigma` integer boundary.
   This is a prerequisite characterization test, not a CppResist link or
   product change.

## GREEN / acceptance

CPU preservation:

```bash
cmake -S . -B build-r5e1-predicate-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-r5e1-predicate-cpu --parallel
./build-r5e1-predicate-cpu/test_core
./build-r5e1-predicate-cpu/test_nn
./build-r5e1-predicate-cpu/test_conv
./build-r5e1-predicate-cpu/test_tensor
```

Visible-device CUDA qualification:

```bash
cmake -S . -B build-r5e1-predicate-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake --build build-r5e1-predicate-cuda --parallel
./build-r5e1-predicate-cuda/test_cuda_tensor
./build-r5e1-predicate-cuda/test_cuda_core
git diff --check
git diff --name-only origin/main...HEAD
```

Acceptance requires:

- CPU/CUDA forward parity for `less_equal` and `where` at `1e-5`, including
  equality, false branch, NaN predicate behavior, and mixed-device/shape
  rejection without transfer.
- `where` VJPs match the selected branch at `1e-5`; condition gradients are
  zero; repeated `backward()` accumulation and `zero_grad()` reset are shown
  on CUDA before comparison copies.
- `all_true` and `all_finite` match CPU for finite, zero, infinity, NaN, and
  empty tensors. CUDA input data remains CUDA-resident before and after the
  explicit bool result.
- Fixed-grid dynamic Gaussian support exactly matches the prior
  `floor(4*sigma)` reference immediately below, at, and above every boundary,
  including the minimum and capped maximum support.
- The final diff contains only approved paths. CppResist R5e.1, its tests,
  CUDA Abbe, optimizer behavior, and checkpoint behavior remain unchanged.

## Stop rules, delivery, and approval

Stop for a revised plan if the bool status cannot be implemented without
materializing a graph Tensor; CUDA implementation needs a new dependency or
raw/public pointer; `where` needs a broader broadcasting/type policy; support
equivalence fails at a boundary; a current test target cannot build the
approved files; or visible-device CUDA validation fails.

Delivery is one plan-only draft PR followed by one implementation PR on this
same branch. The implementation PR remains draft through validation and the
required Claude review, then becomes ready for a separate owner merge
approval. Owner approval of this plan SHA authorizes only this prerequisite
phase; it is not merge approval and does not authorize resuming CppResist.

After this prerequisite merges, rerun the CppResist R5e.1 RED feasibility gate
against that exact merged minimal_autograd revision. The owner must then
explicitly authorize R5e.1 implementation; do not infer it from this plan.
