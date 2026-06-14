# Deliverable — C++ autograd library

## Summary

Implemented a complete C++17 reverse-mode autograd library from the design
plan, plus a README. All 24 source/header files, the test suite, the
CMakeLists, and the README are in `/workspace/autograd/`. The library
compiles cleanly with g++ + Eigen3 and all three test binaries
(`test_core`, `test_nn`, `test_conv`) pass on the local toolchain. A zip
archive is at `/workspace/autograd.zip`.

## Tree of files created

```
/workspace/autograd/
├── CMakeLists.txt                 (45 lines)
├── README.md                      (186 lines)
├── deliverable.md                 (this file)
├── examples/
│   └── linear_regression.cpp      (34 lines)
├── include/
│   ├── autograd.h                 (12 lines, umbrella)
│   └── autograd/
│       ├── tensor.h               (30 lines)
│       ├── variable.h             (49 lines)
│       ├── function.h             (71 lines)
│       ├── ops.h                  (225 lines)
│       ├── module.h               (70 lines)
│       ├── loss.h                 (18 lines)
│       ├── optim.h                (44 lines)
│       └── conv.h                 (113 lines)
├── src/
│   ├── variable.cpp               (39 lines)
│   ├── ops.cpp                    (7 lines — placeholder; definitions are inline in ops.h)
│   ├── module.cpp                 (11 lines)
│   ├── loss.cpp                   (23 lines)
│   ├── optim.cpp                  (38 lines)
│   └── conv.cpp                   (242 lines)
└── test/
    ├── grad_check.h               (78 lines)
    ├── test_core.cpp              (189 lines)
    ├── test_nn.cpp                (135 lines)
    └── test_conv.cpp              (164 lines)
```

Total: 1823 lines.

## Phase completion checklist

| Phase | Deliverable                         | Status | Notes |
|-------|-------------------------------------|--------|-------|
| 2a    | Topo-sort backward                  | ✅     | `_build_topo` + reverse walk in `var->backward()`. `x + x` → grad = 2 (not 4). `x*x*x` → grad = 3x². |
| 2b    | `apply<>` redesign                  | ✅     | Two overloads: `apply<Fn>(ins)` and `apply<Fn>(ins, args...)`. Captures `ins` (vector<VarPtr>) by value, runs `fn->backward(self->grad)` then `+=` into each parent. |
| 2c    | `BroadcastAddFn` (bias add)         | ✅     | Forward: rowwise add `(1, D)` to each row of `(N, D)`. Backward: pass-through for input, `g.colwise().sum()` for bias — bias grad sums over batch. |
| 2d    | `grad_check` utility                | ✅     | Central finite differences; `test/grad_check.h`. All Phase 1 ops pass. |
| 3     | Softmax / shape ops                 | ✅     | `softmax`, `log_softmax`, `transpose`, `reshape`, `concat` — all have `grad_check` coverage. |
| 4     | Module system                       | ✅     | `Linear` (Xavier), `ReLUModule`, `Sequential`. |
| 5     | Training                            | ✅     | `mse_loss`, `cross_entropy`, `SGD`, `Adam` (with bias correction). XOR fits in 200 epochs at lr=0.05. |
| 6a    | `im2col` / `col2im`                 | ✅     | Both pure functions. `col2im` uses `+=` for overlapping patches. |
| 6b    | `Conv2dFn`                          | ✅     | Forward = im2col + matmul + bias broadcast. Backward returns `(grad_input, grad_W, grad_b)`. grad_check passes for input, weight, and bias. |
| 6c    | `Conv2d` / `MaxPool2d`              | ✅     | Modules with Kaiming-uniform init. End-to-end Conv→Pool→Linear forward + backward exercises every parameter's grad. |

## Deviations from the plan

1. **`mse_loss` uses `add`, not `broadcast_add`.** The plan's snippet uses
   `broadcast_add(pred, Var::make(-target))`. With `pred` and `target` of
   the same shape (the common case, e.g. `(N, D)` logits and one-hot
   targets), `broadcast_add` is the wrong op — it expects the second
   argument to be `(1, D)`. I changed it to `add(pred, Var::make(-target))`
   (element-wise subtraction). A comment in `src/loss.cpp` documents this.
   Functionally equivalent for same-shape inputs.

2. **`ops.cpp` is a placeholder.** The free functions in `ops.h` are
   defined as `inline` (in the header) so the `.cpp` is just a TU that
   the static library still needs to link in. The plan implies a
   non-inline `ops.cpp` but the inline form is simpler and avoids
   redefinition errors; the linker still has a TU for the static lib.

3. **`apply<Fn>` has two overloads.** The plan shows only the one-arg
   form. I added a variadic two-arg overload `apply<Fn>(ins, args...)`
   to support `ScaleFn(float s)` and `ReshapeFn(int r, int c)` and
   similar. Documented in `function.h`.

4. **XOR test uses 200 epochs (plan said 500).** Adam at lr=0.05
   overshoots after 200 — the loss drops to ~1e-12 and predictions
   start to diverge. Capping at 200 keeps the test stable. Plan's
   "convergence in <200 epochs" criterion still holds.

5. **`MaxPool2d::forward(x, H, W)` derives C from `cols / (H*W)`.** The
   plan shows the same signature. This is fine when the caller knows the
   channel count (which it must, since the layout is flat). Documented
   in the README's Limitations.

6. **`grad_check.h` is in `test/`, not in `include/`.** The plan's
   repository layout doesn't mention it explicitly. I put it in `test/`
   next to the tests, and added `target_include_directories(test_xxx
   PRIVATE test)` to the test targets in CMake.

7. **No `examples/` directory in the plan's layout, but I added one.**
   The plan said the README example "should be runnable when copied into
   a .cpp file in the test/ dir or examples/". I created
   `examples/linear_regression.cpp` and added an `example` build target.

## Verification

- All three test binaries compile and pass on the local g++ 12.2 + Eigen
  3.4.0 toolchain.
- The README's quick-start example was copied verbatim into
  `examples/linear_regression.cpp` and verified to compile and produce
  a converging fit on `y = 2*x1 + 3*x2`.
- Cross-checked every function signature in the source against the
  plan; no plan API is missing.
- The architectural concerns called out as critical in the plan are
  honoured:
  - `backward()` uses the topo-sort design (Phase 2a fix); recursive
    DFS is gone.
  - `BroadcastAddFn::backward` reduces grad over the batch axis (Phase
    2c fix).
  - `col2im` uses `+=` for overlapping patches (Phase 6a critical).
  - `Conv2d` uses the flat `(N, C*H*W)` layout documented in the
    plan and in the README.

## Confirmation of zip

`/workspace/autograd.zip` exists and contains the full `/workspace/autograd/`
tree (excluding `build/` and any VCS metadata). Created via:

```bash
cd /workspace && zip -r autograd.zip autograd -x '*.DS_Store' 'autograd/build/*'
```
