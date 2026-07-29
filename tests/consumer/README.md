# Consumer smoke tests for `minimal_autograd`

This directory contains minimal external-consumer smoke projects that
demonstrate the supported consumption boundaries of `minimal_autograd`:

- `find_package/` — CONFIG-based consumption via
  `find_package(autograd CONFIG REQUIRED)`.
- `add_subdir/`  — tree-embedded consumption via `add_subdirectory(...)`.

Each consumer links the namespaced target `autograd::autograd` and
produces an executable named `autograd_smoke`. The two consumer projects
share the same minimal `main.cpp`: a deterministic compile/link/run check
that does fixed `Var` / `scale` / `sum` / `backward` work and verifies
exact forward and gradient values. There is no training loop, no random
init, and no internal / Eigen / CUDA header includes — only the normal
public umbrella `#include "autograd.h"`.

Both projects live under `tests/consumer/` so they are visible to repo
maintainers, but they are NEVER picked up by the autograd build itself
(the project's top-level `CMakeLists.txt` only lists source files
explicitly and does not recurse into subdirectories).

## `tests/consumer/find_package/`

Standalone `find_package(CONFIG REQUIRED)` smoke project. The consumer
does **not** influence the autograd build itself: it discovers the
installed `autogradConfig.cmake` from `CMAKE_PREFIX_PATH` (or
`autograd_DIR`) and links the resulting `autograd::autograd` target.
Whether CUDA is in the package was decided at autograd install time.

## `tests/consumer/add_subdir/`

Standalone `add_subdirectory` smoke project. The consumer consumes the
autograd source tree via `add_subdirectory(... AUTOGRAD_SRC_DIR)`, so
passing `-DAUTOGRAD_USE_CUDA=ON` to the parent does enable CUDA on the
embedded autograd build.

This consumer's own `CMakeLists.txt` enforces the no-inheritance
contract at configure time: after `add_subdirectory(...)` it checks for
any of the known autograd in-tree test/example targets
(`test_core`, `test_nn`, `test_conv`, `test_fft`,
`test_characterization`, `test_extensions`, `test_diffusion`,
`test_smoke`, `test_cuda_core`, `test_cuda_fft`, `test_cuda_tensor`,
`linear_regression`, `mnist_classify_cpu`, `mnist_classify_gpu`, and the
legacy `example`)
and fails the configure step if any of them are visible. The check
exists so the package boundary is verified by the build itself rather
than only by manual inspection. It can be silenced with
`-DAUTOGRAD_CONSUMER_ALLOW_TARGET_LEAK=ON` for the explicit opt-in case
(see `-DAUTOGRAD_BUILD_TESTS=ON` / `-DAUTOGRAD_BUILD_EXAMPLES=ON`).

## Building the consumers

```sh
# CPU top-level build → install to a repo-local prefix
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix .scratch/install

# find_package consumer
cmake -S tests/consumer/find_package -B .scratch/findpkg_build \
    -DCMAKE_PREFIX_PATH=.scratch/install
cmake --build .scratch/findpkg_build
.scratch/findpkg_build/autograd_smoke   # prints "OK: forward=60, grad=2.5"

# add_subdirectory consumer (the blueprint must live outside the autograd
# source tree or the add_subdirectory call would recurse into itself)
cp -r tests/consumer/add_subdir .scratch/addsub_consumer
cmake -S .scratch/addsub_consumer -B .scratch/addsub_build \
    -DAUTOGRAD_SRC_DIR="$PWD"
cmake --build .scratch/addsub_build
.scratch/addsub_build/autograd_smoke   # prints "OK: forward=60, grad=2.5"
```

For CUDA-enabled autograd:

```sh
PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release -DAUTOGRAD_USE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH cmake --build build-cuda --parallel
PATH=/usr/local/cuda/bin:$PATH cmake --install build-cuda \
    --prefix .scratch/install_cuda

PATH=/usr/local/cuda/bin:$PATH cmake -S tests/consumer/find_package \
    -B .scratch/findpkg_cuda_build \
    -DCMAKE_PREFIX_PATH=.scratch/install_cuda
PATH=/usr/local/cuda/bin:$PATH cmake --build .scratch/findpkg_cuda_build
.scratch/findpkg_cuda_build/autograd_smoke   # prints "OK: forward=60, grad=2.5"
```
