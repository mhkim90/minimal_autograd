# Second CUDA Hardening Plan: Differentiable FFT

## Goal

Add the smallest general FFT/autograd surface that downstream projects need for
end-to-end differentiable spectral computations, without moving application
physics into `minimal_autograd`.

Primary target:

- A downstream project can express a differentiable frequency-domain filtering
  pipeline as:

```text
signal   = real_to_complex(x)
spectrum = fft2(signal)
field    = ifft2(filter * spectrum)
image    = abs2(field)
loss     = user_loss(image)
```

and get finite gradients back to the real input.

This is a second hardening pass. The first CUDA hardening pass deliberately kept
FFT/cuFFT out of scope. This plan adds FFT because FFT backward is a general
autograd primitive, while domain-specific filtering, modeling, and objective
policy stay in downstream projects.

## References Checked

- `CUDA_HARDENING_PLAN.md`: current CUDA backend is opt-in, real-valued, and has
  no FFT/cuFFT primitive.
- `include/autograd/variable.h`: `Var` owns real `Mat data`, `Mat grad`, optional
  CUDA mirrors, and a single-output `back_fn`.
- `include/autograd/ops.h`: public ops are real-valued free functions returning
  `VarPtr`.
- `../minimal_tensor/docs/PLAN.md`: useful phase-gated style, device/runtime
  discipline, and rule that compute/storage abstractions should not be rewritten
  casually.
- `../minimal_tensor/docs/plans/tensor_generalization.md`: useful warning that
  true dtype/storage generalization is larger than a CUDA kernel change.
- Existing FFT users in downstream projects need differentiable
  frequency-domain multiplication and inverse transforms, but their physics and
  losses must remain outside this repository.

## Current State

Available now:

- Real `Var` graph with reverse-mode backward.
- CUDA opt-in through `AUTOGRAD_USE_CUDA=ON`.
- CUDA real ops needed by common differentiable models: arithmetic, matmul, broadcast, sum,
  activations, softmax/log-softmax, `Conv2d`, `MaxPool2d`, `SGD`, `Adam`.
- Explicit CUDA sync semantics.
- No-device CUDA tests skip cleanly.

Missing for differentiable FFT pipelines:

- No complex value type.
- No FFT/DFT op.
- No FFT backward rule.
- No CUDA FFT kernel.
- No cuFFT integration.
- No complex multiply / conjugate / magnitude-squared primitives.
- No generic spectral filtering composition test inside `minimal_autograd`.

## Locked Ownership Decisions

- `minimal_autograd` owns generic differentiable primitives:
  - complex pair representation;
  - `fft2` / `ifft2`;
  - complex multiply, conjugate, `abs2`;
  - FFT backward/VJP;
  - optional CUDA FFT execution.
- Downstream projects own domain models, filters, losses, and optimization
  policy.
- Do not add domain-specific APIs to `minimal_autograd`.
- Do not force CUDA dependency in default builds.
- Do not silently fall back from CUDA FFT to CPU. Unsupported device/shape cases
  must throw or skip in tests.

## FFT Backend Decision

Prefer a pure CUDA FFT kernel first if it stays small and testable.

Rationale:

- Keeps dependency surface smaller than cuFFT.
- Matches the existing "minimal teaching library" style.
- First useful grids for this backend are small and power-of-two.

Boundary:

- Pure CUDA FFT v1 supports only `float32` complex C2C, 2-D, power-of-two
  dimensions, and small/medium sizes such as 8, 16, 32, 64, 128, and maybe 256.
- Non-power-of-two sizes must throw a clear error in CUDA v1.
- CPU reference may support only small test sizes at first.
- cuFFT remains an optional fallback plan, not the first default.

Optional cuFFT escape hatch:

- Add `AUTOGRAD_USE_CUFFT=ON` only if pure CUDA FFT becomes too large, too slow,
  or too restrictive for common downstream grids.
- If added, cuFFT is backend implementation detail for `fft2`/`ifft2`, not a
  downstream project dependency.
- Tests must pass under pure-CUDA and cuFFT paths before cuFFT becomes a
  recommended path.

## API Sketch

Keep `Var` real-valued. Add a small complex wrapper over two real `VarPtr`s:

```cpp
namespace ag {

struct ComplexVar {
    VarPtr real;
    VarPtr imag;
};

ComplexVar make_complex(VarPtr real, VarPtr imag);
ComplexVar real_to_complex(VarPtr real);
VarPtr real(const ComplexVar& z);
VarPtr imag(const ComplexVar& z);
ComplexVar conj(const ComplexVar& z);
ComplexVar complex_mul(const ComplexVar& a, const ComplexVar& b);
ComplexVar complex_scale(const ComplexVar& z, float s);
VarPtr abs2(const ComplexVar& z);

ComplexVar fft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);
ComplexVar ifft2(const ComplexVar& z, FftNorm norm = FftNorm::Backward);

} // namespace ag
```

Notes:

- `ComplexVar` is a thin API object, not a new graph node type.
- Each complex output is represented by two `Var` nodes.
- FFT output real/imag nodes need shared backward context so gradients from both
  parts accumulate into both input parts.
- Keep default FFT normalization compatible with PyTorch:
  - `fft2`: unnormalized forward.
  - `ifft2`: divides by `H * W`.
- Backward rules must match that normalization:
  - backward of `fft2` applies adjoint FFT scale (`N * ifft2` for default norm).
  - backward of `ifft2` applies adjoint inverse scale (`fft2 / N` for default
    norm).

## Phase F0: Audit, Branch, and Red Gates

Goal:

Create exact failing tests before implementation.

Deliverables:

- Commit this plan.
- Add red tests for missing API:
  - `ComplexVar`;
  - `real_to_complex`;
  - `fft2`;
  - `ifft2`;
  - `complex_mul`;
  - `abs2`.
- Add a tiny spectral filtering expression test that should later compute:

```text
loss = sum(abs2(ifft2(const_filter * fft2(real_to_complex(x)))))
```

Validation:

- CPU build still passes existing tests.
- New FFT test fails to compile before Phase F1.

Exit:

- Red gate proves the required public surface is absent.

## Phase F1: Complex Pair Core on CPU

Goal:

Add complex arithmetic without FFT and without CUDA.

Deliverables:

- New `include/autograd/complex.h`.
- New `src/complex.cpp` only if helpers cannot stay header-only.
- `ComplexVar` wrapper.
- `real_to_complex`, `complex`, `real`, `imag`, `conj`.
- `complex_mul`, `complex_scale`, `abs2`.
- Shape/device validation:
  - real and imag shapes must match;
  - mixed CPU/CUDA complex pairs throw clearly until CUDA support lands.
- Tests:
  - forward values for multiply/conjugate/abs2;
  - gradient through `abs2(complex_mul(a, b))`;
  - repeated-branch gradient accumulation.

Validation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/test_core
./build/test_fft
```

Exit:

- Complex arithmetic is differentiable on CPU.
- No CUDA behavior changed.

## Phase F2: CPU Reference FFT with Backward

Goal:

Add correct differentiable `fft2` / `ifft2` with small-size CPU reference.

Deliverables:

- New `include/autograd/fft.h`.
- New `src/fft.cpp`.
- CPU reference DFT/FFT implementation.
- Shared FFT backward context for real and imaginary output nodes.
- Explicit shape conventions:
  - input `Mat` rows are `H`, cols are `W`;
  - output preserves `(H, W)`;
  - Eigen column-major storage is internal detail.
- Default normalization documented and tested.

Implementation choice:

- Start with direct O(N^4) DFT for correctness on very small tests if that is
  fastest to implement safely.
- Add iterative radix-2 CPU FFT only if direct DFT is too slow for tests.
- Keep CPU reference simple; CUDA path is the perf target.

Tests:

- `ifft2(fft2(z)) == z` for 4x4 and 8x8.
- Delta, ones, checkerboard, non-square 4x8.
- Real input with zero imaginary part.
- Gradcheck by finite differences on `sum(abs2(ifft2(filter * fft2(x))))`.
- PyTorch fixture parity if fixture generation is available; otherwise numeric
  constants from a small known DFT.

Precision note:

- `minimal_autograd` is currently `float32` (`Eigen::MatrixXf`). Do not promise
  PyTorch-style double `gradcheck` until dtype generalization exists. Use
  finite-difference checks with tolerances appropriate for `float32`.

Validation:

```bash
cmake --build build --parallel
./build/test_fft
./build/test_core
./build/test_nn
```

Exit:

- CPU FFT forward and backward are correct for small grids.
- A generic spectral filtering expression has finite input gradients.

## Phase F3: Pure CUDA Complex Core

Goal:

Make complex arithmetic run on CUDA without FFT.

Deliverables:

- CUDA kernels for complex pair ops:
  - `complex_mul`;
  - `complex_scale`;
  - `conj` if needed;
  - `abs2`.
- Device validation:
  - both parts on same CUDA device;
  - mixed CPU/CUDA throws;
  - host sync semantics match existing CUDA ops.
- Tests mirror F1 on CUDA and compare CPU/CUDA forward/backward.

Validation:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DAUTOGRAD_USE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/test_cuda_core
./build-cuda/test_cuda_fft
```

No-device behavior:

- `test_cuda_fft` prints a skip and exits success when no CUDA device is visible.

Exit:

- Complex arithmetic works on device.
- FFT still CPU-only or throws clearly on CUDA input.

## Phase F4: Pure CUDA FFT Kernel

Goal:

Add a small pure CUDA FFT backend for power-of-two 2-D complex C2C transforms.

Deliverables:

- New CUDA source, for example `src/cuda_fft.cu`.
- Batched 1-D radix-2 Stockham or Cooley-Tukey kernel.
- 2-D transform by row pass then column pass.
- Forward/inverse mode with correct normalization.
- Power-of-two validation:
  - dimensions must be positive;
  - dimensions must be powers of two;
  - supported max size documented.
- Workspace allocation uses existing CUDA helpers or small RAII wrapper.
- Clear runtime errors for unsupported shape, no CUDA build, or no device.

Pure CUDA implementation notes:

- Prefer Stockham autosort if code stays readable; it avoids a separate
  bit-reversal pass.
- Use one kernel per stage for v1. Optimize only after parity tests pass.
- Store complex data as separate real/imag buffers to match `ComplexVar`.
- Keep all math `float32` in CUDA v1.

Tests:

- CPU/CUDA parity for `fft2`, `ifft2`, and round-trip on 8x8, 16x16, 8x16.
- CUDA finite-difference gradient smoke for a small 4x4 or 8x8 case.
- Determinism over repeated calls.
- Unsupported 7x8 or 12x12 shape throws clearly.

Validation:

```bash
cmake --build build-cuda --parallel
./build-cuda/test_cuda_fft
```

Exit:

- Pure CUDA FFT works for small power-of-two grids.
- No cuFFT dependency required.

Phase F4 implementation note:

- Accepted F4 as a pure-CUDA forward stepping stone using separable direct DFT
  row/column kernels, not a radix-2 Stockham/Cooley-Tukey implementation yet.
- CUDA FFT backward remains explicitly deferred to Phase F5 and throws if used.
- The CUDA finite-difference gradient smoke listed above moves to Phase F5 with
  the actual CUDA backward path.

## Phase F5: CUDA FFT Autograd Integration

Goal:

Make CUDA `fft2` / `ifft2` differentiable and usable in composed graphs.

Deliverables:

- CUDA backward path for both real/imag output nodes.
- Shared backward context accumulates gradients exactly once per output part and
  safely accumulates into both input real/imag grads.
- Repeated-branch gradient accumulation tests.
- Cross-device graph errors stay explicit.

Tests:

- `loss = sum(abs2(fft2(z)))` gradient vs CPU.
- `loss = sum(abs2(ifft2(z)))` gradient vs CPU.
- Generic spectral filtering expression:

```text
x -> fft2 -> complex_mul(const_filter) -> ifft2 -> abs2 -> sum
```

  gradient finite and CPU/CUDA close.

Validation:

```bash
./build-cuda/test_cuda_fft
./build-cuda/test_cuda_core
```

Exit:

- End-to-end gradients through frequency-domain filtering are available on CUDA.

## Phase F6: Optional cuFFT Backend Gate

Goal:

Only if pure CUDA FFT is too limited, add cuFFT as an optional backend.

Deliverables:

- CMake option `AUTOGRAD_USE_CUFFT=ON`, valid only with
  `AUTOGRAD_USE_CUDA=ON`.
- Link `CUDA::cufft` only when the option is enabled.
- Backend selector:
  - default pure CUDA for supported power-of-two shapes;
  - optional cuFFT for broader shape/perf coverage;
  - never silently changes numerical normalization.
- Plan cache or RAII plan wrapper with deterministic cleanup.

Tests:

- Same forward/backward suite as F4/F5 under cuFFT.
- Pure CUDA and cuFFT agree on supported shapes within tolerance.

Exit:

- cuFFT is available as an implementation backend, not required by default.

Phase F6 gate decision:

- Deferred. The current pure-CUDA FFT backend is sufficient for the planned
  small power-of-two readiness path after Phase F5, including CUDA
  forward/backward gradient tests.
- No `AUTOGRAD_USE_CUFFT` option or `CUDA::cufft` dependency is added in this
  phase; this preserves the minimal dependency surface.
- Current CUDA FFT support remains intentionally narrow: power-of-two rows and
  columns up to the documented v1 limit, with unsupported shapes throwing
  clearly.
- Reopen F6 before broadening to non-power-of-two shapes, larger grids, or FFT
  performance beyond the pure-CUDA v1 backend.

## Validation Matrix

Always required:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/test_core
./build/test_nn
./build/test_conv
./build/test_fft
```

CUDA required when touching CUDA:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/test_cuda_core
./build-cuda/test_cuda_fft
```

Optional cuFFT validation:

```bash
cmake -S . -B build-cufft \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON \
  -DAUTOGRAD_USE_CUFFT=ON
cmake --build build-cufft --parallel
./build-cufft/test_cuda_fft
```

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Complex support becomes a full tensor rewrite | Large delay | Use `ComplexVar {real, imag}` wrapper first; no dtype/storage rewrite |
| Two-output FFT backward double-counts or misses gradients | Wrong gradients | Shared context tests with repeated branches and finite differences |
| Pure CUDA FFT grows too complex | Slow delivery | Limit v1 to power-of-two small grids; keep cuFFT optional fallback |
| cuFFT becomes mandatory | Breaks minimal dependency story | Gate behind `AUTOGRAD_USE_CUFFT`; pure CUDA remains first target |
| Domain physics leaks into `minimal_autograd` | Bad abstraction | Only generic complex/FFT ops live here |
| Hard frequency filters make gradcheck fragile | False failures | Use smooth or fixed filters for numerical gradient checks |
| Downstream use cases expand scope | Delays core blocker | Add only generic primitives with direct tests |

## Grilled-Me Review

Assumptions confirmed:

- `minimal_autograd` currently has no complex or FFT primitive.
- Downstream frequency-domain models need FFT backward before CUDA paths can be
  autograd-connected.
- `minimal_tensor` is a useful reference for phase discipline and device/runtime
  caution, but copying its full storage abstraction would be too much for this
  pass.

Risks identified:

- Pure CUDA FFT is easy only under narrow constraints. Arbitrary-size FFT is not
  easy and should not be promised in v1.
- Existing `Function` API returns one `Mat`; complex FFT needs a careful
  two-output wrapper design.
- Hard-edged frequency filters can make numerical gradient checks fragile.

Simplification applied:

- Keep `Var` real-valued and add `ComplexVar` pair wrapper.
- Start with CPU correctness, then pure CUDA power-of-two FFT.
- Leave cuFFT optional and later.
- Leave domain policy and physics in downstream projects.

Surviving concerns:

- Exact FFT normalization/backward scale must be locked by tests before
  downstream projects depend on gradients.
- If downstream projects need arbitrary non-power-of-two layouts soon,
  pure CUDA v1 may need cuFFT fallback earlier than planned.
