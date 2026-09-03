# Downstream Consumer Checks

This directory contains standalone projects for the supported consumption
boundaries of `minimal_autograd`:

- `find_package/` uses an installed CONFIG package.
- `add_subdir/` embeds the source tree with `add_subdirectory(...)`.
- `eigen_custom/` explicitly requests the `expert` component and uses the
  opt-in Eigen extension header for a copied custom operation.
- `cuda_custom/` explicitly requests the `expert` component and uses the
  installed CUDA package with an external CUDA kernel.
- `legacy_expert/` is a source-preserving `autograd.h`/`Mat`/`VarPtr` fixture;
  it requests the expert target and uses the existing CUDA runtime helper only
  for a CUDA-enabled install.
- `find_package/unknown_component/` is expected to fail at configure time.

The `find_package` and `add_subdir` smoke programs are identical. They use the
replacement Tensor/Variable API, `nn::Module` modules, canonical free
functions, and `optim::Adam`. They intentionally do not include the umbrella
header or any legacy graph and Eigen surface. The smoke check covers a fixed
forward/backward pass, registered parameter names and traversal, an Adam step,
and an AdamState snapshot/load_state round trip with deterministic values.

The normal Tensor storage contract is dense row-major. The opt-in
`autograd/extension/eigen.h` helpers are explicit copies: `tensor_from_eigen`
creates a CPU Tensor and `tensor_to_eigen` materializes a rank-2 CPU Tensor as
an Eigen matrix. Both preserve logical `(row, col)` values while reordering
between Tensor row-major and Eigen column-major storage. They do not alias
storage, create raw views, or perform hidden device transfers.

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
    -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE
cmake --build .scratch/findpkg_build
.scratch/findpkg_build/autograd_smoke
```

Run the explicit expert consumers:

```sh

cmake -S tests/consumer/eigen_custom -B .scratch/eigen_custom_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install"
cmake --build .scratch/eigen_custom_build
.scratch/eigen_custom_build/autograd_eigen_custom

cmake -S tests/consumer/legacy_expert -B .scratch/legacy_expert_build \
    -DCMAKE_PREFIX_PATH="$PWD/.scratch/install"
cmake --build .scratch/legacy_expert_build
.scratch/legacy_expert_build/autograd_legacy_expert
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
    -DAUTOGRAD_SRC_DIR="$PWD"
cmake --build .scratch/addsub_build
.scratch/addsub_build/autograd_smoke
```

The add-subdirectory project also fails at configure time if known in-tree test
or example targets leak into the consumer.

Hosted CI runs the CPU build, install, normal and expert installed consumers,
and source-tree consumer. It does not claim to run CUDA without a CUDA runner
and device.

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
The CUDA-enabled `legacy_expert` fixture can be run with the same installed
prefix; it prints `SKIP` when no device is visible.
