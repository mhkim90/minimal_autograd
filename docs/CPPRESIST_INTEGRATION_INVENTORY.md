# CppResist Integration Inventory

## Purpose

This Phase 0 inventory records how the sibling `CppResist` repository consumes
the current `minimal_autograd` API. It defines downstream requirements for the
new public and extension APIs; it does not preserve direct access to current
implementation fields.

Inspected sibling path: `../CppResist`.

## Build coupling

- `CppResist/CMakeLists.txt` locates `minimal_autograd` through
  `CPPRESIST_AUTOGRAD_DIR` and embeds it with `add_subdirectory`.
- It forces `AUTOGRAD_BUILD_EXAMPLES=OFF` and
  `AUTOGRAD_BUILD_ADVANCED_OPS=OFF`.
- CUDA builds force `AUTOGRAD_USE_CUDA=ON`.
- It links the un-namespaced `autograd` target.

Required replacement:

- a namespaced consumable target;
- subdirectory-safe build defaults;
- optional installed-package consumption;
- CUDA feature propagation without editing backend internals.

## Tensor and host-data coupling

`CppResist` uses `ag::Mat` throughout public interfaces, implementations,
examples, tools, datasets, optics, and checkpoints. It relies on Eigen
construction, arithmetic, indexing, row/column queries, and column-major
layout.

It also reads and mutates `Var::data` and `Var::grad` directly for:

- parameter initialization;
- model outputs and scalar losses;
- shape checks;
- parameter snapshots and restoration;
- checkpoint validation and serialization;
- custom backward accumulation.

Representative locations:

- `CppResist/src/optics.cpp`
- `CppResist/src/calibration.cpp`
- `CppResist/src/checkpoint.cpp`
- `CppResist/examples/TrainSingleSample.cpp`
- `CppResist/examples/TrainResist.cpp`
- `CppResist/tools/ModelHardening.cpp`

Required replacement:

- `Tensor` shape/value/device queries;
- explicit checked host import/export;
- opt-in Eigen interoperation;
- controlled parameter initialization and restoration;
- read-only value and gradient access through `Variable`.

Image layout contract to preserve explicitly:

- rows represent image height;
- columns represent image width;
- `(y, x)` indexing;
- existing host/CUDA transfers assume Eigen-compatible column-major order.

## Autograd graph coupling

CPU and CUDA domain operations construct graph nodes by assigning:

- `parents`;
- `back_fn`;
- gradients directly.

Representative locations:

- `CppResist/examples/TrainResist.cpp`
- `CppResist/src/cuda_resist_kernels.cu`

Required replacement:

- a supported custom-operation constructor;
- validated input/output gradient counts, shapes, and devices;
- saved-tensor lifetime rules;
- library-owned graph traversal and gradient accumulation.

## CUDA coupling

CUDA code uses:

- `cuda()`, `cpu()`, `is_cuda()`, and `cuda_device()`;
- `cuda_data()` and `cuda_grad()` raw pointers;
- explicit synchronization between public host and device representations;
- CUDA-resident custom-operation outputs and optimizer slots.

Representative locations:

- `CppResist/src/cuda_resist_kernels.cu`
- `CppResist/src/cuda_runtime.cpp`
- `CppResist/examples/TrainResist.cpp`
- `CppResist/src/checkpoint.cpp`

Required replacement:

- explicit `Device` and `Tensor::to`;
- borrowed CUDA tensor views with ownership/lifetime rules;
- library-owned CUDA output allocation;
- controlled gradient-output access for custom kernels;
- explicit synchronization contract.

## Module and parameter coupling

`CppResist` manually assembles ordered `std::vector<ag::VarPtr>` parameter
lists and exposes model parameters through individual accessors. Checkpoint
slot identity depends on stable parameter order.

Required replacement:

- registered parameters;
- deterministic `parameters()` and `named_parameters()` traversal;
- stable names/order for checkpoints;
- no mutable public parameter containers.

## Optimizer coupling

Checkpoint code directly accesses and mutates public `Adam` members:

- `params`;
- `lr`, `beta1`, `beta2`, and `eps`;
- step counter `t`;
- CPU moments `m` and `v`;
- CUDA moments `cuda_m` and `cuda_v`.

It validates slot counts, shape, device residency, and the invariant that the
checkpoint training step equals `Adam.t`.

Representative locations:

- `CppResist/src/checkpoint.cpp`
- `CppResist/examples/TrainResist.cpp`
- `CppResist/include/cppresist/checkpoint.h`

Required replacement:

- explicit serializable `AdamState`;
- stable parameter-slot identity;
- atomic validated restore;
- same-device moment tensors;
- step-count and hyperparameter accessors;
- no duplicated public CPU/CUDA moment representation.

## Migration-spike acceptance criteria

A future Phase 10 `CppResist` spike succeeds when it can:

1. build against the namespaced public target;
2. exchange image tensors without normal-header Eigen exposure;
3. implement CPU and CUDA resist operations through supported extensions;
4. traverse model parameters deterministically;
5. save and restore Adam state without optimizer field access;
6. run representative CPU and CUDA forward/backward training paths;
7. avoid all access to graph fields, owned raw allocations, and private
   implementation headers.
