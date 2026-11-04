# autograd

Small C++17 reverse-mode automatic differentiation library for teaching and
experiments. Owned `Tensor` values use dense row-major storage; `Variable`
handles share a private graph. CPU is the default; CUDA is optional.

## Build and dependencies

Building the provider requires Eigen3, CMake 3.14+, and a C++17 compiler.
OpenMP is used when found. CUDA builds additionally require a CUDA toolkit and
CMake 3.18+. Installed normal consumers do not need Eigen headers or discovery.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Default builds are STATIC with 15 registered tests. Use
`-DBUILD_SHARED_LIBS=ON` for SHARED, `-DAUTOGRAD_BUILD_ADVANCED_OPS=ON` for
18 CPU tests (diffusion/extensions/composed training), and
`-DAUTOGRAD_BUILD_EXAMPLES=ON` for examples. Embedded `add_subdirectory`
builds default to no inherited tests/examples and preserve parent C++ defaults.

```sh
PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc \
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DAUTOGRAD_BUILD_ADVANCED_OPS=ON
PATH=/usr/local/cuda/bin:$PATH cmake --build build-cuda --parallel 4
ctest --test-dir build-cuda --output-on-failure
```

CUDA advanced builds register 21 tests. Architecture defaults to 75 unless
`CMAKE_CUDA_ARCHITECTURES` or `AUTOGRAD_CUDA_ARCHITECTURES` is supplied. Device
qualification requires actual execution on a visible GPU, not a capability
probe or a no-device skip.

## Public API

Include `autograd/autograd.h` (canonical) or `autograd.h` (forwarder).
Both include the normal API, custom-op construction, and complex/FFT, without
Eigen or CUDA headers. Existing `autograd/core/*.h`, `tensor.h`, `shape.h`, and
`device.h` include paths remain supported.

- `Tensor`: owned storage, explicit `from_host`/`copy_to_host`/
  `copy_from_host`, shape/device, clone and transfer. No writable raw normal
  view or Eigen alias. Tensor copies share storage; use `clone` for isolation.
- `Variable`: `value()`, `grad()`, `has_grad()`, `requires_grad()`, `backward`,
  `zero_grad`, `detach`, `to`. Graph fields are private. `zero_grad` clears
  only its receiver; module/optimizer clearing visits registered parameters.
  Backward uses a fresh propagation seed, accumulates committed gradients,
  and restores gradients if propagation fails.
- Arithmetic, activations, reductions, N-D shape/slice/broadcast ops, matrix
  multiplication, losses, NCHW spatial operations and differentiable GroupNorm
  are declared in `core/ops.h` and `core/loss.h`.
- `ag::nn`: `NamedParameter`, `Module`, `Linear`, `ReLU`, `Sequential`,
  `Conv2d`, `MaxPool2d`, `DepthwiseConv2d`, `AvgPool2d`,
  `NearestUpsample2d`, `GroupNorm`.
- `ag::optim`: `SGD`, `Adam`, `AdamState`; one portable optimizer state with
  deep-copied snapshot/load semantics, not host/device mirror states.
- `ag::diffusion`: seeded `randn`, `randn_like`, rank-1 odd-safe
  `sinusoidal_time_embedding`, and differentiable `q_sample`.
- Every listed nn/optim type also has a flat `ag` **identity alias**; the four
  diffusion functions have flat using-declarations, retaining default args.
  Declarations and symbols still belong to their existing namespaces.
- `ComplexVariable` pairs real/imaginary Variables. `fft2`/`ifft2` apply to the
  final two axes, with arbitrary positive extents, CPU/CUDA support and
  `FftNorm::Backward` normalization. No cuFFT dependency.

### Minimal forward/backward

```cpp
#include <autograd/autograd.h>

int main() {
    const float values[] = {2.f, 3.f};
    ag::Variable x(ag::Tensor::from_host(values, ag::Shape{1, 2}), true);
    auto loss = ag::sum(ag::scale(x, 2.f));
    loss.backward();
    float result = 0.f, gradient[2];
    loss.value().copy_to_host(&result, 1);  // 10
    x.grad().copy_to_host(gradient, 2);     // 2, 2
}
```

## CUDA boundaries

Use explicit `Tensor::to` / `Variable::to` with `Device::cuda()` or
`Device::cpu()`. A cross-device Variable transfer creates a fresh leaf; value
and stored-gradient Tensor transfers are separate. There are no host/device
public mirror fields or sync methods. CUDA-built providers still execute CPU
Tensor operations on CPU. Compiled capability is not GPU availability.

CUDA supports core arithmetic/activations, matrix multiplication, reductions,
softmax/losses, slices/concat/flip/reshape, Conv2d/MaxPool2d, complex FFT, and
SGD/Adam. Not every CPU operation has a CUDA implementation: transpose, cumsum,
sin/cos/clamp, DepthwiseConv2d, AvgPool2d, NearestUpsample2d and GroupNorm remain
CPU-only and reject CUDA inputs. See operation tests for precise validation;
there is no silent CPU fallback or new device feature in this retirement.
Modules have no general `Module::to`; the GPU example registers device-owned
parameters and uses public functional ops.

## Consumers and explicit extensions

```cmake
find_package(autograd 1.0 CONFIG REQUIRED) # unversioned also supported
# or: add_subdirectory(minimal_autograd)
target_link_libraries(my_app PRIVATE autograd::autograd)
get_target_property(cuda_enabled autograd::autograd AUTOGRAD_CUDA_ENABLED)
# Literal "0" or "1" describes the compiled provider only.
```

Installed normal SHARED consumption needs no Eigen/toolkit discovery. CUDA
STATIC consumption retains link-only `CUDA::cudart`; recorded OpenMP dependency
semantics remain. Source/embedded provider compilation still needs its build
dependencies. Neither normal nor expert consumers receive `AUTOGRAD_USE_CUDA`
or the private in-tree test selector.

```cmake
find_package(autograd CONFIG REQUIRED COMPONENTS expert)
target_link_libraries(my_custom_op PRIVATE autograd::autograd_expert)
```

The explicit expert target supplies Eigen and, for CUDA providers, cudart.
`extension/eigen.h` exposes `tensor_from_eigen(const Eigen::MatrixXf&)` and
`tensor_to_eigen`: rank-2 CPU copies with logical-index reordering and no
aliasing or implicit device transfer. `extension/cuda.h` supplies borrowed
expert views, not owning raw normal storage; callers retain ownership and
synchronization/lifetime responsibility. `extension/custom_op.h` supplies
validated forward/backward construction and is also in the normal umbrella.
Unknown package components reject. Standalone commands and installed
positive/negative controls: [tests/consumer/README.md](tests/consumer/README.md).

## Current examples

```sh
cmake -S . -B build-examples -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_BUILD_EXAMPLES=ON
cmake --build build-examples --parallel 4
./build-examples/linear_regression
./build-examples/mnist_classify_cpu data/MNIST/raw 4 2 1 2
# CUDA-enabled example build only:
./build-cuda/mnist_classify_gpu data/MNIST/raw 4 2 1 2
```

Linear regression trains the two-layer ReLU MLP on `y=2*x1+3*x2` using Adam.
MNIST examples keep the CNN, IDX `/255` row-major values, mini-batches and
train/test evaluation; CPU uses Adam, GPU uses SGD. Arguments are data directory,
train count, test count, epochs, batch size. Data must already exist; these
commands do not download it. Enable examples in the CUDA build for the GPU
binary. Bounded execution is not an accuracy or speedup claim.

## Migration and 1.0 policy/status

This is an intentional legacy **source break**. Removed: `Var`, `VarPtr`,
`Function`, `Mat`/`Mats`, old activation wrappers, public graph/mirror fields,
legacy constructor/forward adapters, and the twelve old top-level
`autograd/{variable,function,ops,module,loss,optim,conv,norm,diffusion,complex,fft,cuda_core}.h`
headers, including the removed `CudaRuntimeInfo` / `cuda_runtime_info` diagnostics
from `cuda_core.h`. No expert opt-in restores these names. Migrate to owned Tensor values,
Variable handles, registered modules and explicit transfers/copies; rebuild
legacy binaries. Spatial values are true NCHW, not flat Eigen plus metadata;
flatten explicitly at a Linear boundary. Complex values use `ComplexVariable`.

The candidate declares project/package VERSION **1.0.0**, shared VERSION
**1.0.0** and SOVERSION **1**. Generated `SameMajorVersion` selection accepts
unversioned, 1.0 and exact 1.0.0 requests and rejects newer unsatisfied requests
or other majors. Source-compatible evolution is intended within 1.x; breaking
source changes require a new major. This is **not** a universal ABI promise
across compiler/stdlib, static/shared or CPU/CUDA variants, nor established
future same-major binary compatibility. No release/tag is published here.

Local qualification completed all eight default/advanced CPU/CUDA STATIC/SHARED
configurations, installed/embedded normal consumers, explicit Eigen expert and
actual CUDA device consumers, and versioned downstream 32/32/41. The downstream
checks use the corrected consumer driver merged separately in
[CppResist #174](https://github.com/mhkim90/CppResist/pull/174) at `befe0e5`;
its merged tree matches the qualified consumer candidate.
Final controller review and local commit/publication status are recorded in the
root plan and controller journal; no provider merge/release is claimed here.
Actual minimum CMake 3.14 execution has not been tested on the current 3.28 host;
the minimum is unchanged.
Historical guarantees and explicit retirement mapping are appended to
[the Phase0 contract](docs/PHASE0_BEHAVIOR_CONTRACT.md).

## Limitations

Teaching-scale implementation, no production/performance guarantee. No dilated
or transposed convolution. Only Backward FFT normalization is implemented.
Mutating aliased Tensor data between forward and backward invalidates saved
values; build fresh graphs after such mutation. Explicit expert borrowed views
require the documented ownership and synchronization discipline.

## License

Apache-2.0
