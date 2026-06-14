# autograd

A small reverse-mode automatic differentiation library in C++17, built on top
of [Eigen3](https://eigen.tuxfamily.org/). The library implements a tape-based
autograd engine, a small module system (`Linear`, `Sequential`, `Conv2d`,
`MaxPool2d`), a few losses (`mse_loss`, `cross_entropy`), and two optimizers
(`SGD`, `Adam`). It is single-threaded, CPU-only, and is intended for teaching
and small experiments — not production.

## Features

- Reverse-mode autograd via a shared-`Var` graph with a topological-sort
  backward pass (no double-counting of shared nodes).
- Element-wise ops (`add`, `mul`, `scale`), matmul, ReLU, broadcast bias-add
  with correct bias gradient, sum, softmax, log-softmax, transpose, reshape,
  concat.
- Module system with `Linear` (Xavier init), `ReLUModule`, `Sequential`.
- Losses: `mse_loss`, `cross_entropy`.
- Optimizers: `SGD`, `Adam` (with bias correction).
- 2D conv stack: `im2col` / `col2im` (with correct overlap accumulation),
  `Conv2d` (Kaiming uniform init), `MaxPool2d`, gradient-checked.
- Numerical `grad_check` helper (central finite differences).
- Single static library, no runtime dependencies beyond Eigen.

## Dependencies

- **Eigen3** (header-only)
- **CMake 3.14+**
- **C++17** compiler (GCC 9+, Clang 10+ tested)
- Optional: **OpenMP** (Eigen uses it for parallelism when enabled)

## Build

```bash
cd autograd
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces the static library `libautograd.a` and three test binaries:

```bash
./test_core    # core op correctness + grad_check
./test_nn      # end-to-end net training (XOR with Adam)
./test_conv    # im2col / col2im / Conv2d / MaxPool2d + grad_check
```

All three should print `ALL TESTS PASSED`.

## Quick start

Train a 2-layer ReLU MLP to fit a tiny linear function `y = 2*x1 + 3*x2` with
Adam. Drop this into `examples/linear_regression.cpp`, add an executable
target to `CMakeLists.txt`, and run.

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
epoch 0    loss=2.5-ish
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

`Function` subclasses (`AddFn`, `MatMulFn`, `Conv2dFn`, …) implement
`forward(Mats)` and `backward(Mat)`. The free template `apply<Fn>(inputs,
extra_args...)` wraps them: it constructs the function, runs forward, and
captures a `back_fn` lambda that holds a `shared_ptr<Fn>` and the parent
`VarPtr`s. Calling `loss->backward()` on a scalar (1x1) `Var` performs the
topo-sort, then invokes each captured `back_fn` in reverse order.

A `Module` is anything with `forward(VarPtr) → VarPtr` and a list of
`parameters() → vector<VarPtr>`. `Sequential` composes modules; optimizers
iterate over `parameters()` and call `step()` / `zero_grad()`.

### Flat conv layout

This is important and non-obvious: **`Var::data` is always a 2D
`Eigen::MatrixXf`** (no 4D tensors). For conv layers a `(N, C, H, W)` batch
is stored as `Mat(N, C*H*W)`, with channels contiguous along the column axis.
The shape `(C, H, W)` is passed explicitly into each conv op / module call
and not encoded in the `Var`. See `include/autograd/conv.h` for the exact
convention and `test/test_conv.cpp` for examples.

## Phase-by-phase feature table

This table mirrors the implementation order in the design plan, but written
for someone using the library.

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

## Limitations

These are the same items called out in the design plan. They are not bugs;
they are scope decisions.

- **Single-threaded by default.** Eigen is single-threaded unless you link
  OpenMP and call `Eigen::setNbThreads(n)`. The CMakeLists in this repo
  picks up OpenMP automatically when available.
- **No `retain_graph`.** Each forward pass may be backwarded exactly once.
  If you call `backward()` twice on the same graph, gradients will
  accumulate on top of the previous run.
- **No inplace ops.** Every op allocates a new `Var`. Mutating a leaf
  parameter is fine, but mutating a non-leaf `Var->data` will silently
  invalidate the graph.
- **No GPU.** `Mat` is `Eigen::MatrixXf`. There is no CUDA path.
- **No dilated / transposed / depthwise conv.** `Conv2d` is the standard
  cross-correlation used in mainstream frameworks. Extending it is a matter
  of adding fields to `Conv2dFn` and the factory.
- **Conv layout is flat (2D).** All conv calls require the caller to track
  `(C, H, W)` separately. `Conv2d::forward(x)` (one-arg) is `assert`-ed
  out; use `forward(x, H, W)`.

## License

MIT. (Replace with your own if you're forking.)
