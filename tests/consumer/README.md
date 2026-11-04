# Downstream Consumer Checks

This directory contains standalone projects for the supported consumption
boundaries of `minimal_autograd`:

- `find_package/` uses an installed CONFIG package.
- `add_subdir/` embeds the source tree with `add_subdirectory(...)`.
- `eigen_custom/` explicitly requests the `expert` component and uses the
  opt-in Eigen extension header for a copied custom operation.
- `cuda_custom/` explicitly requests the `expert` component and uses the
  installed CUDA package with an external CUDA kernel.
- `api_freeze/` checks installed headers/aliases and intentional removed-surface
  failures, including with explicit Eigen expert, plus version selection.
- `find_package/unknown_component/` is expected to fail at configure time.

The `find_package` and `add_subdir` smoke programs are identical. They use the
replacement Tensor/Variable API, `nn::Module` modules, canonical free
functions, and `optim::Adam`. They include both normal umbrellas and direct core headers without
any legacy graph or Eigen surface. The smoke check covers a fixed
forward/backward pass, registered parameter names and traversal, an Adam step,
and an AdamState snapshot/load_state round trip with deterministic values.

The normal Tensor storage contract is dense row-major. The opt-in
`autograd/extension/eigen.h` helpers are explicit copies: `tensor_from_eigen`
creates a CPU Tensor and `tensor_to_eigen` materializes a rank-2 CPU Tensor as
an Eigen matrix. Both preserve logical `(row, col)` values while reordering
between Tensor row-major and Eigen column-major storage. They do not alias
storage, create raw views, or perform hidden device transfers.

Both normal consumers require `AUTOGRAD_CUDA_ENABLED` on the normal target to
be exactly `0` or `1`, including when embedded. Pass the independent
`-DAUTOGRAD_EXPECT_CUDA_ENABLED=0` or `=1` to verify provider polarity. This
metadata describes build capability only, not runtime or GPU availability.
The CUDA custom consumer requires `1` and still links the expert target for
its extension. Neither target exports private backend/test definitions.

## CPU Checks

Build and install the library to a local prefix:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix .scratch/install
```

Run the installed normal `find_package` smoke check with Eigen discovery
disabled. This proves the normal package does not require or export Eigen:

```sh
cmake -S tests/consumer/find_package -B .scratch/findpkg_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install" \
    -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE \
    -DAUTOGRAD_EXPECT_CUDA_ENABLED=0
cmake --build .scratch/findpkg_build
.scratch/findpkg_build/autograd_smoke
```

Run the explicit expert consumers:

```sh

cmake -S tests/consumer/eigen_custom -B .scratch/eigen_custom_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install"
cmake --build .scratch/eigen_custom_build
.scratch/eigen_custom_build/autograd_eigen_custom

```

The unknown required component must reject configuration:

```sh
cmake -S tests/consumer/find_package/unknown_component \
    -B .scratch/unknown_component_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install"  # expected to fail
```

Run the source-tree consumer from a copy outside the source tree so the
embedded `add_subdirectory` call cannot recurse into itself:

```sh
cp -r tests/consumer/add_subdir .scratch/addsub_consumer
cmake -S .scratch/addsub_consumer -B .scratch/addsub_build \
    -DAUTOGRAD_SRC_DIR="$PWD" -DAUTOGRAD_EXPECT_CUDA_ENABLED=0
cmake --build .scratch/addsub_build
.scratch/addsub_build/autograd_smoke
```

The add-subdirectory project also fails at configure time if known in-tree test
or example targets leak into the consumer.

Hosted CI runs the CPU build, install, normal and expert installed consumers,
and source-tree consumer. It does not claim to run CUDA without a CUDA runner
and device.

For a CUDA-built normal package, use `-DAUTOGRAD_EXPECT_CUDA_ENABLED=1`.
Eigen discovery can still be disabled for either library type; toolkit
package discovery can also be disabled for CUDA SHARED, but CUDA STATIC
requires its existing link-only `CUDA::cudart` dependency. Use
`-DBUILD_SHARED_LIBS=ON` when building the shared provider.

## Manual CUDA Check

Build and install a CUDA-enabled library, then configure the standalone CUDA
consumer against that install. The toolkit path may be omitted when `nvcc` is
already on `PATH`.

```sh
PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DAUTOGRAD_USE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH cmake --build build-cuda --parallel
PATH=/usr/local/cuda/bin:$PATH cmake --install build-cuda \
    --prefix .scratch/install_cuda

PATH=/usr/local/cuda/bin:$PATH cmake -S tests/consumer/cuda_custom \
    -B .scratch/cuda_custom_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install_cuda" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH cmake --build .scratch/cuda_custom_build --parallel
PATH=/usr/local/cuda/bin:$PATH \
    .scratch/cuda_custom_build/autograd_cuda_custom
```

The CUDA custom consumer keeps ownership in `Tensor`, obtains only borrowed
device views from `autograd/extension/cuda.h`, launches and synchronizes its
own kernels, and checks forward and backward results through host copies.
No legacy fixture remains; API-freeze negative controls prove expert opt-in
does not restore removed names. CUDA qualification requires a visible device.

## Installed API freeze and version controls

Use a fresh prefix/build for every candidate (never install over old headers).
The freeze fixture first compiles valid package/umbrella/header controls, then
requires removed names, all twelve old headers, and private graph/state fields
to fail with matching diagnostics. The negative matcher is GCC-diagnostic-qualified;
Clang and other diagnostic formats are not qualified (Clang is not installed
on the qualification host). Expert mode first proves explicit Eigen
copies compile, then repeats removed-name controls with Eigen included.

```sh
cmake -S tests/consumer/api_freeze -B .scratch/freeze-normal \
  -DCMAKE_PREFIX_PATH="$PWD/.scratch/install" \
  -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
cmake --build .scratch/freeze-normal --parallel 4
ctest --test-dir .scratch/freeze-normal --output-on-failure
cmake -S tests/consumer/api_freeze -B .scratch/freeze-expert \
  -DCMAKE_PREFIX_PATH="$PWD/.scratch/install" -DAUTOGRAD_FREEZE_EXPERT=ON
cmake --build .scratch/freeze-expert --parallel 4
ctest --test-dir .scratch/freeze-expert --output-on-failure
```

For fresh configure directories, pass `-DAUTOGRAD_REQUEST_VERSION=1.0`, or
`-DAUTOGRAD_REQUEST_VERSION=1.0.0 -DAUTOGRAD_REQUEST_EXACT=ON`; both must accept.
Omitting the request must accept. Requests 1.1, 1.0.1, 0.9 and 2.0 must reject
through the actual installed ConfigVersion file (not a compiler/SDK error).
A SHARED install must report SONAME `libautograd.so.1` via `readelf -d`.
These are package-selection controls, not a general ABI compatibility promise.
Host CMake 3.28 was used; actual 3.14 runtime remains untested.

Local qualification completed all four CPU/CUDA STATIC/SHARED boundaries:
installed and embedded normal consumers, explicit Eigen expert and actual CUDA
custom-device checks, plus API-freeze/version controls. Eight default/advanced
provider configurations and versioned downstream 32/32/41 passed locally. The
corrected downstream driver is unmerged and belongs to a separate delivery;
these local results do not claim merged/released status.
