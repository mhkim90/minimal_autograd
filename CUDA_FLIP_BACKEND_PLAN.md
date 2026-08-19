# CUDA Backend Completion for Direct `ag::flip`

Status: draft — revised plan-only. The prior approval is invalidated by this
scope correction; no further product edit is authorized until owner approval
is recorded against this exact revised plan commit.

## Goal and evidence

Complete CUDA support for the existing public direct-`ag::Variable`
`flip(const Variable&, int axis)` operation. The API already exists and works
on CPU, but its current public implementation explicitly rejects CUDA before
dispatch.

This is the narrow upstream prerequisite exposed by the CppResist R5e.1
public-API CUDA feasibility probe. On an RTX 5060, the disposable probe built
against `minimal_autograd` merge `465f861` and stopped at:

```text
flip: CUDA tensors are not supported in this build (got cuda:0)
```

`slice`, `concat`, and `conv2d` already dispatch to CUDA. Exact reflect
smoothing needs `slice -> flip -> concat -> conv2d`; no downstream workaround
may introduce host materialization, raw/private APIs, or a CppResist CUDA
kernel.

## Frozen contract

- Keep the existing `ag::flip(const Variable&, int axis = -1)` signature and
  CPU semantics unchanged. No public header change is needed.
- On CUDA, reverse values along one normalized axis for any non-empty rank
  supported by the existing shape model. Preserve the input shape and device.
- Backward is the same axis reversal applied to the upstream gradient; it must
  remain CUDA-resident.
- Follow the existing contiguous row-major shape-kernel pattern: compute
  `inner`, the selected axis coordinate, and its reversed input coordinate.
  Do not allocate host data or create a Tensor-backed metadata workaround.
- Axis validation and negative-axis normalization stay in the existing public
  path. Empty tensors retain the CPU operation's shape/device behavior.
- Do not add CUDA `cumsum`, generic gather/permutation, broadcasting policy,
  integer dtypes, raw/public CUDA pointers, legacy `Var` changes, CppResist
  changes, or CMake changes.

## Approved scope

- `CUDA_FLIP_BACKEND_PLAN.md`
- `src/core/ops.cpp`
- `src/core/variable.cpp`
- `src/core/tensor_dispatch_cpu_shape.cpp`
- `src/core/tensor_dispatch_cuda_shape.cpp`
- `src/cuda/tensor_ops_shape.cu`
- `src/detail/tensor_cuda_ops.h`
- `src/detail/tensor_kernels.h`
- `test/test_cuda_tensor.cpp`
- `test/test_cpu_ops.cpp` only if a focused CPU parity/empty-axis guard is
  missing and cannot be expressed in the existing CUDA test.

Forbidden: all other source, public API headers, legacy
`include/autograd/ops.h`, CMake, CppResist, and unrelated cleanup.
`src/core/variable.cpp` may change only to add the existing `OpKind::Flip` to
the CUDA-autograd supported-operation allowlist. Its existing Flip VJP remains
unchanged. Stop for a revised plan if any other variable/autograd contract or
public API requires modification.

## Single implementation phase — CUDA flip dispatch and VJP

Safety risk: L3 (existing public autograd API and CUDA gradient correctness).
Implementation difficulty: standard. Route: one Luna implementation session.
Trigger one asynchronous Claude review for CUDA axis/index/VJP correctness.
Checkpoint every 10 minutes; maximum wait 30 minutes; at most three
implementation attempts.

### RED gate

Before editing, record the existing CPU rejection and backend seams:

```bash
rg -n -C 3 'Variable flip|validate_cpu\("flip" \
  src/core/ops.cpp
rg -n -C 3 'tensor_flip_nd|OpKind::Flip' \
  src/detail/tensor_kernels.h src/core/variable.cpp
rg -n -C 3 'cuda_tensor_(slice|concat)|tensor_slice_nd' \
  src/core/tensor_dispatch_cuda_shape.cpp src/cuda/tensor_ops_shape.cu \
  src/detail/tensor_cuda_ops.h
```

Compile and run a disposable CUDA direct-Variable probe on a visible GPU that
calls `ag::flip` on `cuda:0`; expected baseline is the explicit CPU-only
rejection. The probe is not committed.

### Implementation steps

1. Separate the existing CPU flip helper from the backend-neutral
   `tensor_flip_nd` dispatch seam without changing CPU behavior.
2. Add CUDA `tensor_flip_nd` dispatch and a private CUDA flip kernel in the
   existing shape provider. Reuse the selected-axis/inner-stride mapping used
   by CUDA slice and concat; launch only for non-empty tensors.
3. Remove only the public `flip` CUDA rejection and add existing
   `OpKind::Flip` to `cuda_backward_supported`. Keep the existing Flip VJP
   unchanged so backward calls the same dispatch and reverses the CUDA
   upstream gradient on the same axis.
4. Add focused CUDA tests for rank-2 axes `0`, `1`, and `-1`; rank-3 internal
   axis; forward CPU parity; CUDA residence; `sum(flip(x, axis)).backward()`
   gradient parity; repeated backward accumulation; and empty tensors. Extend
   CPU coverage only if a contract is otherwise absent.
5. Re-run the disposable CppResist R5e.1 reflect-smoothing probe against the
   implemented upstream branch. It must pass the prior `flip` gate; report a
   later independent blocker without editing CppResist.

## GREEN / acceptance

CPU preservation:

```bash
cmake -S . -B build-cuda-flip-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-flip-cpu --parallel
./build-cuda-flip-cpu/test_cpu_ops
./build-cuda-flip-cpu/test_tensor
```

Visible-device CUDA qualification (performed through OpenCode):

```bash
cmake -S . -B build-cuda-flip-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON \
  -DAUTOGRAD_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda-flip-cuda --parallel
./build-cuda-flip-cuda/test_cuda_tensor
./build-cuda-flip-cuda/test_cuda_core
git diff --check
```

Acceptance requires:

- Exact CPU/CUDA forward agreement for selected axes and ranks, including
  negative-axis normalization and empty tensors.
- CUDA `flip` output, input gradient, and repeated-backward accumulated
  gradient remain on the selected CUDA device until final test-only copies.
- Backward equals a same-axis reversal and matches the CPU/reference result.
- `Variable::backward` accepts the existing Flip node on CUDA only after its
  dispatch/kernel support is present; no other CUDA-autograd allowlist entry
  changes.
- CPU behavior remains unchanged.
- The final diff is limited to approved paths; no host materialization,
  raw/private API, or CppResist change is introduced.
- The CppResist disposable R5e.1 probe gets past reflect smoothing without
  host materialization. A later independent failure is reported as its own
  blocker, not worked around here.

## Stop rules, delivery, and approval

Stop for a revised plan if rank/axis behavior differs from CPU, CUDA VJP
requires changing the existing `OpKind::Flip` contract, a private/raw CUDA
surface is needed, an unsupported operation other than `flip` blocks the
consumer probe, visible-device validation fails, or scope expands beyond the
listed files.

After owner approval of this plan SHA, implement on this branch. Keep the PR
draft through validation and the required Claude review; then mark it ready
for separate merge approval. This approval does not authorize CppResist R5e.1
product edits. R5e.1 may only rerun its feasibility probe after this upstream
change merges.
