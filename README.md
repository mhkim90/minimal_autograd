# autograd

A small reverse-mode automatic differentiation library in C++17, built on top
of [Eigen3](https://eigen.tuxfamily.org/). The library implements a tape-based
autograd engine, a small module system (`Linear`, `Sequential`, `Conv2d`,
`MaxPool2d`, `AvgPool2d`, `DepthwiseConv2d`, `NearestUpsample2d`, `GroupNorm`),
a few losses (`mse_loss`, `cross_entropy`), and two optimizers (`SGD`, `Adam`).
It is intended for teaching and small experiments — not production. The default
build focuses on the beginner CPU core; advanced experimental helpers and the
minimal CUDA core are opt-in.

## Features

- Reverse-mode autograd via a shared-`Var` graph with a topological-sort
  backward pass (no double-counting of shared nodes).
- Lightweight logical shape metadata, including 4D `(N, C, H, W)` views over
  the existing flat Eigen matrix layout.
- Element-wise ops (`add`, `mul`, `scale`, `sub`, `div_op`), matmul, ReLU,
  broadcast bias-add with correct bias gradient, sum, softmax, log-softmax,
  transpose, reshape, concat, `hcat` (column-wise / channel-dim cat).
- Activation ops: `sigmoid`, `tanh_op`, `exp_op`, `log_op`, `sqrt_op`,
  `silu`, `softplus` — all gradient-checked.
- Sequence ops: `cumsum` (axis 0 or 1) and `flip` (axis 0 or 1) with
  correct suffix-sum / reverse backward passes.
- Trig / clamp ops: `sin_op`, `cos_op` (sinusoidal embeddings), `clamp`
  (element-wise with zero gradient at the boundary).
- Column ops: `col_slice(x, start, len)` and `split(x)` (even column
  split returning `std::pair<VarPtr, VarPtr>`) — used for AdaGN
  scale+shift decomposition.
- Module system with `Linear` (Xavier init), `ReLUModule`, `SiLUModule`,
  `SigmoidModule`, `Sequential`.
- Losses: `mse_loss`, `cross_entropy`.
- Optimizers: `SGD`, `Adam` (with bias correction).
- 2D conv stack: `im2col` / `col2im` (with correct overlap accumulation),
  `Conv2d` (Kaiming uniform init), `MaxPool2d`, `AvgPool2d`,
  `DepthwiseConv2d` (groups = channels), `NearestUpsample2d` — all
  gradient-checked.
- Normalisation: `GroupNorm` (inference-only; normalises over C/G × HW groups).
- Numerical `grad_check` helper (central finite differences).
- Single static library, no required runtime dependencies beyond Eigen for the
  default CPU build.

## Dependencies

- **Eigen3** (header-only)
- **CMake 3.14+**
- **C++17** compiler (GCC 9+, Clang 10+ tested)
- Optional: **OpenMP** (Eigen uses it for parallelism when enabled)
- Optional: **CUDA Toolkit** and CMake 3.18+ for `AUTOGRAD_USE_CUDA=ON`

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces the static library `libautograd.a` and three core test binaries:

```bash
./build/test_core        # core op correctness + grad_check
./build/test_nn          # end-to-end net training (XOR with Adam)
./build/test_conv        # im2col / col2im / Conv2d / MaxPool2d + grad_check
```

Advanced experimental tests are opt-in:

```bash
cmake -S . -B build-advanced -DCMAKE_BUILD_TYPE=Release -DAUTOGRAD_BUILD_ADVANCED_OPS=ON
cmake --build build-advanced --parallel
./build-advanced/test_extensions  # 30 grad-checks for the extension ops
./build-advanced/test_diffusion   # 17 grad-checks / shape / value assertions
./build-advanced/test_smoke       # 35 end-to-end smoke checks
```

Minimal CUDA core autograd is opt-in and intentionally small:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DAUTOGRAD_USE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/test_cuda_core
```

Current CUDA coverage is `Var::cuda()`, `Var::cpu()`, `add`, `mul`, `matmul`,
`broadcast_add`, `scale`, `relu`, `sum`, `softmax`, `log_softmax`,
`cross_entropy`, and `SGD`, including backward/gradient accumulation on device.
Unsupported CUDA ops throw instead of silently falling back to stale host data.

`test_core`, `test_nn`, and `test_cuda_core` print `ALL ... TESTS PASSED`.
`test_conv` prints `21 passed, 0 failed`. `test_extensions` prints
`30/30 passed`, `test_diffusion` prints `17/17 passed`, and `test_smoke`
prints `35/35 passed`.

### Smoke test (`test_smoke`)

`test/test_smoke.cpp` is a single-file end-to-end smoke test that exercises
every op needed by a Gaussian-Unified Diffusion Model through both a real
**training loop** (forward → loss → backward → Adam step, repeated until
convergence) and a **forward-only inference loop**. It is the fastest
sanity check that the library can still be wired into a working model.

| # | Group                       | Mode                              | Ops exercised                                                                 |
|---|-----------------------------|-----------------------------------|--------------------------------------------------------------------------------|
| 1 | sigma schedule convergence  | **training loop** (200 steps)     | `softmax`, `cumsum`, `scale`, `sub`, `mul`, `sum`, `Adam`                     |
| 2 | beta schedule convergence   | **training loop** (200 steps)     | `tanh_op`, `scale`, `sub`, `mul`, `sum`, `Adam`                               |
| 3 | spatial forward             | forward-only                      | `Conv2d`, `AvgPool2d`, `NearestUpsample2d`, `hcat`                            |
| 4 | GroupNorm + SiLU            | forward-only                      | `GroupNorm` (mean≈0, var≈1 per group), `silu`                                 |
| 5 | embedding consistency       | forward-only                      | `sinusoidal_time_embedding` vs manual `sin_op`+`cos_op`+`hcat`, `split` round-trip |
| 6 | randn stats                 | forward-only                      | `randn`, `randn_like` (diffusion.h): mean≈0, std≈1, determinism, leaf-ness    |
| 7 | mini inference loop         | **inference loop** (10 steps)     | All of the above + `clamp`, `Linear`, `broadcast_add`, `replicate`             |

The training-loop tests deliberately avoid `GroupNorm` in the backward path
since its backward is inference-only. The mini inference loop simulates a
DDIM-style reverse step with a small `Linear` "model" and verifies that
all outputs stay finite and properly clamped to `[0, 1]`.

Build and run:

```bash
cmake -S . -B build-advanced -DCMAKE_BUILD_TYPE=Release -DAUTOGRAD_BUILD_ADVANCED_OPS=ON
cmake --build build-advanced --parallel
./build-advanced/test_smoke
```

## Quick start

The default build always creates the README example executable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target example --parallel
./build/example
```

It trains a 2-layer ReLU MLP to fit `y = 2*x1 + 3*x2` with Adam. The source is
`examples/linear_regression.cpp`:

```cpp
#include "autograd.h"
#include <iostream>
using namespace ag;

int main() {
    // Tiny dataset: y = 2*x1 + 3*x2
    Mat X(4, 2); X << 1, 1,  1, 0,  0, 1,  0, 0;
    Mat Y(4, 1); Y << 5,  2,  3,  0;

    auto net = std::make_shared<Sequential>();
    net->add(std::make_shared<Linear>(2, 8));
    net->add(std::make_shared<ReLUModule>());
    net->add(std::make_shared<Linear>(8, 1));
    Adam opt(net->parameters(), 0.05f);

    for (int epoch = 0; epoch < 200; ++epoch) {
        opt.zero_grad();
        auto pred = net->forward(Var::make(X));
        auto loss = mse_loss(pred, Y);
        loss->backward();
        opt.step();
        if (epoch % 50 == 0)
            std::cout << "epoch " << epoch << "  loss=" << loss->data(0, 0) << "\n";
    }
    std::cout << "final pred:\n" << net->forward(Var::make(X))->data << "\n";
    return 0;
}
```

Expected end-of-training output is approximately:

```
epoch 0    loss=<initial value>
epoch 50   loss=...
...
final pred: ~[[5], [2], [3], [0]]
```

## Architecture

A `Var` is a node in a directed acyclic graph. It holds the result of a forward
computation (`data`) and, after `backward()`, its gradient (`grad`). Each
non-input `Var` also stores its parents and a `back_fn` (a closure that knows
how to push its own gradient back into its parents' gradients).

```
                       forward
       input  ─────────────────►  ops  ─────────────►  Var graph
       (Var)        apply<Fn>      build nodes        (root = loss)
                                                            │
                                                            │  loss->backward()
                                                            ▼
                       ┌────────────────────────────────────────────┐
                       │  1. seed grad on loss = ones(1, 1)         │
                       │  2. DFS from root to build topo order      │
                       │  3. walk topo in REVERSE; at each node:    │
                       │       node->back_fn()                      │
                       │     → reads node->grad                     │
                       │     → calls fn->backward(grad)             │
                       │     → accumulates into parents[i]->grad    │
                       └────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                                  parameters with .grad
```

`Function` subclasses (`AddFn`, `MatMulFn`, `Conv2dFn`, `SiLUFn`, …) implement
`forward(Mats)` and `backward(Mat)`. The free template `apply<Fn>(inputs,
extra_args...)` wraps them: it constructs the function, runs forward, and
captures a `back_fn` lambda that holds a `shared_ptr<Fn>` and the parent
`VarPtr`s. Calling `loss->backward()` on a scalar (1x1) `Var` performs the
topo-sort, then invokes each captured `back_fn` in reverse order.

A `Module` is anything with `forward(VarPtr) → VarPtr` and a list of
`parameters() → vector<VarPtr>`. `Sequential` composes modules; optimizers
iterate over `parameters()` and call `step()` / `zero_grad()`.
`zero_grad()` clears both CPU gradients and CUDA device gradients when CUDA is
enabled.

### Logical 4D Shape

This is important and non-obvious: **`Var::data` is always a 2D
`Eigen::MatrixXf`**. For conv layers a `(N, C, H, W)` batch is stored as
`Mat(N, C*H*W)`, with channels contiguous along the column axis. `Var` can now
carry logical shape metadata, so beginner code can write:

```cpp
auto x = Var::make4d(Mat::Random(N, C * H * W), N, C, H, W);
Conv2d conv(C, 4, 3, 3);
auto y = conv.forward(x);   // infers H and W from x->shape()
```

The older expert API still works:

```cpp
auto y = conv.forward(x, H, W);
```

## Phase-by-phase feature table

| Phase | Capability                         | How to use it                                            |
|-------|------------------------------------|----------------------------------------------------------|
| 1     | Core ops                           | `add`, `mul`, `matmul`, `relu`, `sum` — free functions.  |
| 2a    | Topo-sort backward                 | Any `Var` graph; `x + x` no longer double-counts.        |
| 2c    | Broadcast bias-add                 | `broadcast_add(Var, Var)`; bias grad sums over batch.    |
| 2d    | `grad_check` utility               | `#include "grad_check.h"`; pass closure + leaf Var.      |
| 3     | Softmax / log-softmax / shape ops  | `softmax`, `log_softmax`, `transpose`, `reshape`, `concat`. |
| 4     | Module system                      | `Linear(in, out)`, `ReLUModule`, `Sequential`.           |
| 5     | Training utilities                 | `mse_loss`, `cross_entropy`, `SGD`, `Adam`.              |
| 6a    | `im2col` / `col2im`                | Pure functions in `conv.h`; overlap patches accumulate. |
| 6b    | `Conv2dFn`                         | `conv2d_op(input, weight, bias, N, C, H, W, kH, kW, stride, pad)`. |
| 6c    | `Conv2d` / `MaxPool2d` modules     | `Conv2d(in_ch, out_ch, kH, kW, stride, pad)`, `MaxPool2d(kH, kW)`. |
| C1    | Minimal CUDA core autograd         | `x->cuda()`, then `add`, `mul`, `matmul`, `broadcast_add`, `scale`, `relu`, `sum`, `SGD`, `backward`, `cpu`. |
| C2    | CUDA softmax / log-softmax         | `softmax`, `log_softmax` on device with backward. |
| C3    | CUDA cross-entropy                 | `cross_entropy(pred, target)` on device, built on CUDA `log_softmax`. |
| E1    | Activation + arithmetic ops        | `sigmoid`, `tanh_op`, `exp_op`, `log_op`, `sqrt_op`, `silu`, `softplus`, `sub`, `div_op`. |
| E2    | Sequence ops                       | `cumsum(x, axis)`, `flip(x, axis)` — axis 0 or 1.       |
| E3    | `AvgPool2d`                        | `AvgPool2d(kH, kW, stride)`, or `avgpool2d_op(...)`.    |
| E4    | `NearestUpsample2d`                | `NearestUpsample2d(scale)`, or `nearest_upsample2d_op(...)`. |
| E5    | `DepthwiseConv2d`                  | `DepthwiseConv2d(C, kH, kW, stride, pad)` — one filter per channel. |
| E6    | `GroupNorm`                        | `GroupNorm(G, C)`, call `forward(x, C, HW)` at inference time. |
| E7    | `SiLUModule` / `SigmoidModule`     | Activation modules for use inside `Sequential`.          |
| E8    | Trig + clamp + column ops          | `sin_op`, `cos_op`, `clamp(x, lo, hi)`, `col_slice(x, start, len)`, `split(x)` → `pair`. |
| E9    | `hcat`                             | `hcat({a, b, c})` — column-wise cat; mirrors `torch.cat(..., dim=1)`. |
| E10   | Diffusion primitives               | `randn`, `randn_like`, `sinusoidal_time_embedding`, `q_sample` (in `diffusion.h`). |

## Diffusion Support

`include/autograd/diffusion.h` adds the minimum building blocks every
DDPM / DDIM / score-based model needs on top of the existing engine.
All four helpers are header-only inline; no new translation unit.

| Helper | Purpose |
|--------|---------|
| `randn(rows, cols, seed=0)` | Leaf `Var` of `rows × cols` standard-normal samples from a seeded `std::mt19937`. No back_fn, no grad contribution. |
| `randn_like(x, seed=0)`     | Same as `randn` but sized to `x->data`. |
| `sinusoidal_time_embedding(t, dim)` | Transformer-style positional encoding for timestep `t` → leaf `Var` of shape `(1, dim)`. `dim` must be even. |
| `q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise=nullptr, seed=0)` | Standard DDPM forward step: `x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·noise`. Differentiated w.r.t. `x0` and `noise` via the existing `add` / `scale` chain. |

Because `q_sample` is just `add(scale(x0, a), scale(noise, b))`, its
gradients are inherited from the existing `AddFn` / `ScaleFn` — no new
`Function` subclass is required.

### Quick example — DDPM forward step + time embedding

```cpp
#include "autograd.h"
using namespace ag;

int main() {
    int N = 4, dim = 128;
    auto x0  = Var::make(Mat::Random(N, 3 * 32 * 32));
    auto t_emb = sinusoidal_time_embedding(/*t=*/250, dim);  // (1, 128)

    // Hand-picked noise (or pass nullptr to let q_sample draw fresh).
    auto noise = randn_like(x0, /*seed=*/0);

    auto x_t = q_sample(x0, 250,
                        /*sqrt_alpha_bar=*/0.6f,
                        /*sqrt_one_minus_ab=*/0.8f,
                        noise);
    // t_emb and random noise are leaves. x_t is differentiable with respect
    // to x0 and noise through the scale/add chain.
    return 0;
}
```

### Tests

```bash
./build-advanced/test_diffusion       # 17 grad-checks / shape / value assertions
```

Coverage: `randn` shape, leaf semantics, mean ≈ 0, std ≈ 1, seed
determinism; `sinusoidal_time_embedding` shape, value range, `sin(t)` /
`cos(t)` at `i = 0`; `q_sample` shape, analytic-vs-numeric grad check
through both `x0` and `noise`, and a closed-form zero-noise case.

## Limitations

These are the same items called out in the design plan. They are not bugs;
they are scope decisions.

- **Single-threaded by default.** Eigen is single-threaded unless you link
  OpenMP and call `Eigen::setNbThreads(n)`. The CMakeLists in this repo
  picks up OpenMP automatically when available.
- **No `retain_graph`.** Each forward pass may be backwarded exactly once.
  If you call `backward()` twice on the same graph, gradients will
  accumulate on top of the previous run. If a `back_fn` throws during
  `backward()`, gradients touched by that backward attempt are restored before
  the exception is rethrown.
- **No inplace ops.** Every op allocates a new `Var`. Mutating a leaf
  parameter is fine, but mutating a non-leaf `Var->data` will silently
  invalidate the graph.
- **Limited CUDA.** CUDA is opt-in and currently covers only the minimal core
  autograd slice: `add`, `mul`, `matmul`, `broadcast_add`, `scale`, `relu`,
  `sum`, `softmax`, `log_softmax`, `cross_entropy`, and `SGD`. Unsupported ops
  should stay on CPU until explicit CUDA kernels are added.
- **No dilated / transposed conv.** `Conv2d` is the standard
  cross-correlation. `DepthwiseConv2d` is available (groups = channels).
  Dilated or transposed variants are not implemented.
- **Conv storage is flat (2D).** Values still live in `Mat(N, C*H*W)`, but
  logical 4D metadata lets `Conv2d::forward(x)` infer `(H, W)` when `x` was
  created with `Var::make4d(...)`. The explicit `forward(x, H, W)` API remains.
- **GroupNorm is inference-only.** Use `GroupNorm::forward(x, C, HW)`.
  Backward is not implemented.

## License

MIT. (Replace with your own if you're forking.)
