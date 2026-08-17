# CUDA Direct Shape Ops Prerequisite for R5e.1

Status: draft — plan-only. No product edit is authorized until owner approval
is recorded against this exact plan commit.

## Goal and evidence

Add same-device CUDA support to the direct `ag::Variable` free functions
`reshape`, `slice` (including `row_slice` and `col_slice`), and `concat`
(including `hcat`). This is the smallest upstream prerequisite for CppResist
R5e.1's exact direct-Variable reflect-padding smoothing:

```text
CUDA rank-2 mask
  -> row/column slice + concat reflect border
  -> reshape to NCHW -> existing CUDA conv2d
  -> backward on the same device
```

The R5e.1 public-API RED probe built with CUDA and reached the first missing
operation on a visible RTX 5060. It failed with:

```text
slice: CUDA tensors are not supported in this build (got cuda:0)
```

The same direct stack also currently rejects CUDA `reshape` and `concat`.
R5e.1 must not add a CppResist raw-storage/custom-CUDA workaround; this
upstream slice is therefore a prerequisite, not a scope expansion of R5e.1.

## Frozen contract

- CPU behavior, public signatures, validation messages/categories, canonical
  row-major layout, and existing CUDA rejection behavior for all other shape
  operations remain unchanged.
- `reshape` on CUDA is metadata-only, sharing the original CUDA storage just
  as the existing `Tensor::reshape` does. Its VJP is a metadata-only reshape
  to the saved input shape.
- `slice` accepts the existing rank/axis/start/length contract, including
  negative axes. CUDA forward returns a new dense tensor on the same device;
  it is not a view. CUDA backward returns a same-device zero tensor with the
  output gradient scattered into `[start, start + len)` along the selected
  axis.
- `concat` accepts the existing same-rank, equal-non-axis-dimension,
  same-device contract. CUDA forward returns one dense tensor on that device;
  CUDA backward returns one dense, same-device gradient tensor per input in
  the original input order.
- No implicit host copy, CPU fallback, legacy `ag::Var` reuse/bridge, public
  raw-pointer API, new dependency, or change to CUDA `flip`, `cumsum`,
  `transpose`, trigonometric, clamp, pooling, or module behavior is allowed.

## Approved scope

- `CUDA_R5E1_DIRECT_SHAPE_OPS_PLAN.md`
- `src/core/ops.cpp`
- `src/core/variable.cpp`
- `src/detail/tensor_kernels.h`
- `src/detail/tensor_cuda_ops.h`
- `src/cuda/tensor_ops.cu`
- `test/test_cuda_tensor.cpp`
- `include/autograd/core/ops.h` only if its CUDA-support documentation must be
  corrected to reflect these existing public functions

`CMakeLists.txt`, all public function signatures, sibling CppResist files,
legacy `ag::Var` CUDA files, and all other sources are forbidden.

## RED gate

Before editing, capture the direct-stack boundary and CUDA baseline:

```bash
rg -n -C 3 'Variable reshape|Variable concat|Variable slice|validate_cpu' \
  src/core/ops.cpp
rg -n -C 3 'OpKind::Reshape|OpKind::Concat|OpKind::Slice' \
  src/core/variable.cpp
rg -n -C 3 'reshape / cumsum /|concat' test/test_cuda_tensor.cpp
rg -n -C 3 'cuda_tensor_(conv2d|zeros)|cuda_tensor_.*slice|cuda_tensor_.*concat' \
  src/detail/tensor_cuda_ops.h src/cuda/tensor_ops.cu
cmake -S . -B build-r5e1-shape-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake --build build-r5e1-shape-cuda --parallel
./build-r5e1-shape-cuda/test_cuda_tensor
```

Expected baseline: direct CUDA `reshape`, `slice`, and `concat` reject before
building a graph; CUDA conv2d and Adam remain green. If no visible CUDA device
can execute the test, stop rather than weaken CUDA acceptance.

## Implementation

1. Remove only the direct-stack CUDA rejection guards for `reshape`, `slice`,
   and `concat`; retain their existing validation before dispatch.
2. Reuse `Tensor::reshape` for CUDA reshape forward and backward. Do not add a
   CUDA kernel, allocation, or copy for this operation.
3. Add private CUDA slice forward/backward helpers and wire them from the
   existing `tensor_kernels.h` CPU/CUDA dispatch point. Use the canonical
   `outer × axis × inner` mapping: for each dense output index derive the
   outer index, selected-axis index, and trailing-inner index; map it to the
   contiguous source. Backward writes each output-gradient element to its
   unique source location in a zero-initialized same-device tensor.
4. Add private CUDA concat helpers and wire forward/backward dispatch from
   `tensor_kernels.h`. Validate shapes/devices in the existing host layer,
   then launch one dense copy kernel per input using the same
   `outer × total-axis × inner` mapping and that input's axis offset. Reuse
   CUDA slice-style extraction for backward; do not stage metadata or values
   through host memory.
5. Add `Reshape`, `Slice`, and `Concat` to `cuda_backward_supported` before
   the existing OpKind backward switch; then dispatch their CUDA gradient
   cases there. Keep CPU paths byte-for-byte behaviorally equivalent.
6. Add focused CUDA tests, including a public-operation-only multi-cell
   reflect border synthesized from ordered individual slices and concats,
   followed by reshape → conv2d. This proves the R5e.1 prerequisite without
   editing or linking CppResist and does not require CUDA `flip`.

## GREEN / acceptance

CPU preservation:

```bash
cmake -S . -B build-r5e1-shape-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-r5e1-shape-cpu --parallel
./build-r5e1-shape-cpu/test_core
./build-r5e1-shape-cpu/test_nn
./build-r5e1-shape-cpu/test_conv
```

Visible-device CUDA qualification:

```bash
cmake -S . -B build-r5e1-shape-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DAUTOGRAD_CUDA_ARCHITECTURES=120
cmake --build build-r5e1-shape-cuda --parallel
./build-r5e1-shape-cuda/test_cuda_tensor
./build-r5e1-shape-cuda/test_cuda_core
git diff --check
git diff --name-only origin/main...HEAD
```

The CUDA test must prove:

- rank-2 and rank-3 forward CPU/CUDA parity for slice and concat at `1e-5`;
  negative-axis slice is covered;
- slice and concat VJPs match CPU at `1e-5`, accumulate under repeated
  backward, and restart after `zero_grad()`;
- reshape forward/VJP preserve CUDA device and storage alias semantics;
- all produced forward values and gradients remain on CUDA before final test
  comparison copies;
- invalid ranges, invalid axes, concat shape mismatch, and mixed-device
  inputs still fail without a transfer; and
- the multi-cell reflect-border (individual slices + concat) → NCHW reshape
  → CUDA conv2d forward/VJP composition is CUDA-resident and CPU-parity-valid
  at `1e-5`.

## Stop rules and publication

Stop for an upstream design review if shape metadata cannot remain host-only,
if a generic-rank operation requires host tensor values or exposed raw storage,
if any CPU contract changes, or if the work needs CUDA support beyond these
three operations. Do not add CppResist changes to compensate.

Safety risk is L3; implementation difficulty is difficult. After exact owner
approval, run one bounded Sol-expert CUDA index/VJP preflight, then one Luna
implementation session. Trigger one Claude read-only review for CUDA indexing,
same-device ownership, and VJP accumulation. Maximum three attempts; after two
identical blockers, stop and obtain the Sol consultation before a revised
third attempt. Check asynchronous roles every 10 minutes through 30 minutes.

Publish this plan as a draft PR. Keep it draft through implementation,
validation, and review. A material change to scope, kernel mapping, numerical
gate, or route invalidates approval. A separate owner merge approval is
required.
