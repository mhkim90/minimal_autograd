# minimal_autograd Architecture Refactor — Final Plan

## 1. Status

This document defines the target architecture and phased migration plan for a
full refactor of `minimal_autograd`.

It is a planning document, not authorization to implement all phases at once.
Each phase should be implemented and reviewed separately, with compatibility
tests established before old interfaces are removed.

## 2. Purpose

`minimal_autograd` is a deliberately small descendant of `minimal_tensor`.
It should remain suitable for teaching and small experiments while also being
usable as the numerical and automatic-differentiation backend of small C++
projects such as `CppResist`.

The refactor has four primary goals:

1. Hide Eigen and CUDA implementation details from the normal public API.
2. Separate numerical tensor storage from automatic-differentiation graph
   state.
3. Provide a small, stable OOP interface for downstream applications.
4. Preserve direct, understandable CPU/CUDA execution without introducing a
   large runtime or speculative backend framework.

The intended relationship is:

```text
minimal_tensor
    parent/reference implementation with broader runtime features

minimal_autograd
    small descendant with a narrow, stable API and direct backends

CppResist and other small projects
    downstream users of the minimal_autograd public API
```

`minimal_autograd` must not depend on `minimal_tensor`. Concepts may be learned
from the parent project, but the child should remain independently buildable
and substantially smaller.

## 3. Architectural Decision

The library will use OOP for long-lived objects and state:

- `Tensor` owns numerical values and device storage.
- `Variable` owns an autograd graph node containing a `Tensor`.
- `Module` owns registered parameters and child modules.
- `Optimizer` owns update policy and optimizer state.

Mathematical operations remain free functions. This keeps expressions simple
and avoids turning every operation into a large virtual class hierarchy.

```text
Application
    |
    v
Public API
    Tensor        numerical value and device
    Variable      autograd value and graph
    Ops           functional mathematical operations
    Module        OOP model composition and parameter ownership
    Optimizer     controlled updates and serializable state
    |
    v
Internal direct dispatch
    |
    +-- CPU kernels implemented with Eigen
    |
    +-- optional CUDA kernels
```

The library will not introduce a public runtime-selectable `ComputeBackend`
interface. Backend choice is a property of `Tensor::device()`, and operations
dispatch internally. CPU-only builds remain first-class builds.

## 4. Proposed Repository Layout

```text
include/
  autograd.h                       compatibility umbrella include
  autograd/
    autograd.h                     canonical umbrella include
    shape.h
    device.h
    tensor.h
    variable.h
    ops.h
    module.h
    optim.h
    loss.h
    conv.h
    norm.h
    diffusion.h
    complex.h
    fft.h

    extension/
      custom_op.h
      host_view.h
      eigen.h
      cuda.h
      optimizer_state.h

src/
  core/
    tensor.cpp
    variable.cpp
    tape.cpp
    ops.cpp
    loss.cpp

  cpu/
    tensor.cpp
    ops.cpp
    conv.cpp
    norm.cpp
    fft.cpp

  cuda/
    tensor.cu
    ops.cu
    reductions.cu
    conv.cu
    pool.cu
    optim.cu
    fft.cu

  nn/
    module.cpp
    conv.cpp
    norm.cpp

  optim/
    optim.cpp

  fft/
    fft.cpp
    complex.cpp

  detail/
    tensor_impl.h
    variable_node.h
    tape.h
    dispatch.h
    cpu_kernels.h
    cuda_kernels.h

tests/
  core/
  autograd/
  ops/
  nn/
  optim/
  fft/
  cuda/
  extension/
  consumer/
```

The exact number of source files may remain smaller when an area is tiny. The
layout expresses dependency boundaries; it is not a requirement to create
empty or one-function files.

## 5. Public API Design

### 5.1 Shape

Provide a small library-owned shape type instead of leaking Eigen dimensions.
It must support:

- rank and element-count queries;
- indexed dimension access;
- checked construction;
- equality and readable formatting;
- overflow-safe element-count calculation.

The initial implementation may use a simple `std::vector<std::size_t>` or
equivalent value type. Do not introduce a symbolic-shape system.

### 5.2 Device

Provide a small value type:

```cpp
enum class DeviceType {
    Cpu,
    Cuda,
};

class Device {
public:
    static Device cpu();
    static Device cuda(int index = 0);

    DeviceType type() const noexcept;
    int index() const noexcept;
};
```

The first implementation only needs CPU and CUDA. It does not need runtime node
IDs, distributed rank information, streams, or remote-device semantics.

### 5.3 Tensor

`Tensor` represents numerical storage only. It does not contain parents,
backward functions, or gradient state.

Conceptual interface:

```cpp
class Tensor {
public:
    Tensor();

    static Tensor empty(const Shape&, Device = Device::cpu());
    static Tensor zeros(const Shape&, Device = Device::cpu());
    static Tensor ones(const Shape&, Device = Device::cpu());
    static Tensor from_host(
        const float* data,
        const Shape& shape,
        Device target = Device::cpu());

    const Shape& shape() const noexcept;
    Device device() const noexcept;
    std::size_t elements() const noexcept;
    bool empty() const noexcept;

    Tensor to(Device) const;
    Tensor clone() const;
    Tensor reshape(const Shape&) const;

    void copy_to_host(float* destination, std::size_t count) const;
    void copy_from_host(const float* source, std::size_t count);

private:
    std::shared_ptr<detail::TensorImpl> impl_;
};
```

Initial guarantees:

- float32 only;
- dense, contiguous storage only;
- N-dimensional logical shape;
- ordinary copies share storage;
- `clone()` makes an independent deep copy;
- one authoritative storage allocation per tensor;
- no public Eigen matrix member;
- no public raw CUDA allocation member;
- no host mirror that can silently become stale.

Adding dtype polymorphism or arbitrary strided views is explicitly outside this
refactor.

### 5.4 Variable

`Variable` is a value-like handle to a private autograd node.

Conceptual interface:

```cpp
class Variable {
public:
    Variable();
    explicit Variable(Tensor value, bool requires_grad = false);

    const Tensor& value() const;
    bool requires_grad() const noexcept;
    bool has_grad() const noexcept;
    const Tensor& grad() const;
    Device device() const noexcept;

    Variable to(Device) const;
    Variable detach() const;
    void backward();
    void backward(const Tensor& upstream_gradient);
    void zero_grad();

private:
    std::shared_ptr<detail::VariableNode> node_;
};
```

The private node owns:

- the forward `Tensor`;
- optional gradient `Tensor`;
- parent node references;
- the backward closure or operation record;
- any saved tensors required for backward.

Downstream code must not mutate parent lists, gradients, backward closures, or
device pointers directly.

### 5.5 Operations

Operations remain free functions:

```cpp
Variable add(const Variable&, const Variable&);
Variable mul(const Variable&, const Variable&);
Variable matmul(const Variable&, const Variable&);
Variable sum(const Variable&);
Variable relu(const Variable&);
```

Each operation follows one path:

1. validate shapes and devices;
2. dispatch its forward calculation to the appropriate tensor kernel;
3. return a plain result when no input requires gradients;
4. otherwise create a private graph node with the required saved state;
5. dispatch backward tensor operations when `backward()` runs.

Public operations must not contain large inline Eigen implementations.

### 5.6 Module

`Module` is the OOP composition boundary:

```cpp
class Module {
public:
    virtual ~Module() = default;
    virtual Variable forward(const Variable&) = 0;

    Variable operator()(const Variable& input);
    std::vector<Variable> parameters();
    std::vector<NamedParameter> named_parameters();
    void zero_grad();

protected:
    void register_parameter(std::string name, Variable parameter);
    void register_module(std::string name, std::shared_ptr<Module> module);

private:
    // Parameter and child-module registries.
};
```

Concrete modules such as `Linear`, `Conv2d`, `GroupNorm`, and `Sequential`
register their state. Their weights, implementation buffers, and child
containers should not be mutable public fields.

Parameter traversal order must be deterministic because downstream checkpoint
and optimizer behavior may depend on it.

### 5.7 Optimizers

Optimizers accept registered parameters and update their tensors through
controlled library APIs. Optimizer state must not depend on public Eigen
objects or exposed CUDA buffers.

`AdamState` conceptually contains:

- step count;
- hyperparameters required for exact continuation;
- first-moment tensors;
- second-moment tensors.

Moment tensors live on the same device as their corresponding parameters.
There must not be separate public `cuda_m` and `cuda_v` representations.

State snapshot and restore should be explicit and fully validated before
mutating a live optimizer.

## 6. Extension Boundary

Hiding implementation details must not prevent small downstream projects from
adding domain-specific operations.

### 6.1 Custom autograd operations

Provide a narrow expert API similar to:

```cpp
using BackwardFunction =
    std::function<std::vector<Tensor>(const Tensor& output_grad)>;

Variable make_custom_variable(
    Tensor output,
    std::vector<Variable> inputs,
    BackwardFunction backward);
```

The exact callable form may be refined, but it must:

- keep graph internals private;
- make input/output ownership clear;
- validate returned gradient counts, shapes, and devices;
- preserve exception safety during backward;
- permit `CppResist`-style domain operations without editing graph fields.

### 6.2 Host views

Normal public headers must not include Eigen. Generic host access should use:

- checked host copies;
- optional borrowed contiguous spans/views when lifetime can be made explicit.

`extension/eigen.h` is an installed expert header for CPU interoperation. It
may include Eigen and provide mapped or copied views, but use of this header is
an explicit opt-in.

### 6.3 CUDA views

`extension/cuda.h` provides an explicitly borrowed view:

```cpp
struct CudaTensorView {
    float* data;
    ShapeView shape;
    int device_index;
};
```

The final shape representation can differ, but the contract must state:

- the library retains ownership;
- the view cannot outlive its source tensor;
- callers must obey device and synchronization requirements;
- downstream code creates owned outputs with `Tensor::empty(..., cuda_device)`
  rather than installing raw allocations into a tensor.

CPU-only builds must provide clear compile-time or runtime diagnostics for CUDA
extension use without exposing CUDA headers through normal public headers.

## 7. Internal Backend Boundary

Dispatch is private and direct:

```text
operation
    -> validate operands
    -> inspect Device
    -> detail::cpu_* or detail::cuda_*
```

This is intentionally smaller than ArrayFire, Flashlight, or the current
`minimal_tensor` runtime model.

Rules:

- CPU implementation details may use Eigen freely under `src/cpu` and
  `src/detail`.
- CUDA implementation details remain under `src/cuda` and CUDA-specific
  private headers.
- Normal public headers compile without Eigen or CUDA headers.
- Unsupported device/operation combinations fail explicitly.
- Mixed-device operations fail unless an operation specifically documents a
  transfer.
- `.to(device)` is explicit; ordinary operations do not perform hidden device
  transfers.

## 8. Scope Guard

The refactor retains:

- float32 tensors;
- dense contiguous storage;
- reverse-mode autograd;
- current mathematical operations;
- current module set;
- SGD and Adam;
- complex/FFT support already in scope;
- optional CUDA equivalents for currently supported operations;
- OpenMP where already useful.

The refactor does not add:

- multiple dtypes or mixed precision;
- sparse or arbitrary strided tensors;
- distributed execution, NCCL, or MPI;
- remote tensors or runtime node IDs;
- a process-global runtime singleton;
- a global storage registry;
- a public virtual backend plugin system;
- stream pools or general asynchronous scheduling;
- a caching allocator framework;
- JIT compilation;
- automatic graph optimization;
- new vendor libraries without separate justification.

These exclusions are essential to keeping the library minimal.

## 9. Refactor Phases

Each phase should leave the repository buildable and testable. Prefer vertical
slices over a long-lived half-migrated rewrite.

### Phase 0 — Freeze behavior with characterization tests

Before replacing interfaces, record current observable behavior:

- shape and layout conventions;
- representative forward results for every operation;
- representative analytical and finite-difference gradients;
- repeated-parent and shared-node graphs;
- gradient accumulation and `zero_grad`;
- backward exception and rollback behavior;
- deterministic module parameter order;
- exact SGD and Adam update trajectories;
- optimizer state continuation;
- FFT normalization, round trips, and gradients;
- CPU/CUDA numerical parity;
- unsupported-operation and wrong-device diagnostics.

Also create a small `CppResist` integration inventory: which tensor, graph,
optimizer, Eigen, and CUDA details it currently touches. This is an input to
the extension API, not a reason to preserve unsafe public fields.

Exit criterion: behavior that must survive the refactor is executable and
documented.

### Phase 1 — Build and package boundary

Make the library safe to consume before changing its data model:

- define a namespaced CMake target and alias;
- make project tests optional for subdirectory consumers;
- avoid changing parent-project build defaults;
- use proper build/install include interfaces;
- export and install package configuration;
- propagate the C++17 requirement through the target;
- preserve position-independent code where required;
- register tests with CTest;
- add a minimal `find_package` or `add_subdirectory` consumer smoke test.

Exit criterion: an external project can link the library without inheriting
its tests or private implementation dependencies.

### Phase 2 — Introduce Shape, Device, and CPU Tensor

Add the new value layer without deleting the old API:

- implement `Shape` and `Device`;
- implement CPU `Tensor` with hidden storage;
- define shallow-copy and `clone()` semantics;
- add checked host import/export;
- add reshape and basic factory functions;
- ensure errors are runtime validations, not debug-only assertions.

Test ownership, copying, empty tensors, invalid shapes, overflow, host
round-trips, and reshape rules.

Exit criterion: CPU numerical storage works without exposing Eigen in normal
public headers.

### Phase 3 — Introduce Variable and the private tape

Implement:

- private `VariableNode`;
- deterministic topological traversal;
- gradient initialization and accumulation;
- `backward`, `detach`, and `zero_grad`;
- exception-safe graph traversal;
- the custom-operation extension boundary.

Prove the design with a narrow vertical slice:

- add;
- multiply;
- scale;
- sum;
- a custom operation.

Exit criterion: a nontrivial graph can run forward and backward using only the
new `Tensor` and `Variable` APIs.

### Phase 4 — Migrate core operations and losses

Move the remaining current operations and losses to the new architecture:

- forward tensor kernels;
- autograd wrappers;
- shape/device validation;
- backward formulas;
- implementation bodies out of public headers.

For each operation, require:

- forward parity with the characterized result;
- analytical or finite-difference gradient checks;
- invalid-input tests;
- shared-node and gradient-accumulation coverage where applicable.

Exit criterion: the old operation implementation is no longer needed by the
new API.

### Phase 5 — Migrate modules in bounded bundles

#### Phase 5a — Module, Linear, and Sequential

- implement registration and deterministic traversal;
- migrate `Linear`;
- migrate `Sequential`;
- validate nested-module parameter discovery.

#### Phase 5b — Convolution and pooling

- migrate `Conv2d`;
- migrate `DepthwiseConv2d`;
- migrate `MaxPool2d`;
- migrate `AvgPool2d`;
- migrate `NearestUpsample2d`.

#### Phase 5c — Normalization

- migrate `GroupNorm`;
- validate parameter and gradient behavior separately.

#### Phase 5d — Diffusion helpers

- migrate `randn`;
- migrate sinusoidal time embeddings;
- migrate `q_sample`.

Exit criterion: all current modules use private registered state and the new
public value types.

### Phase 6 — Migrate optimizers and state

- migrate SGD;
- migrate Adam;
- place moment tensors with their parameters;
- define explicit state snapshot and restore;
- validate a complete state before applying it;
- preserve exact update trajectories within documented tolerance.

Exit criterion: downstream code can checkpoint and restore optimizer state
without accessing implementation fields.

### Phase 7 — Migrate complex and FFT support

- define the minimal complex-tensor public representation;
- migrate CPU FFT forward paths;
- migrate FFT backward paths;
- preserve documented normalization conventions;
- isolate FFT implementation details under `src/fft` and `src/cpu`.

Exit criterion: FFT round trips and gradients pass on the new tensor/autograd
model.

### Phase 8 — Add the CUDA Tensor foundation

Only after the CPU model is stable:

- implement RAII CUDA storage;
- implement explicit CPU/CUDA copies;
- implement `Tensor::to`;
- add checked borrowed CUDA views;
- define synchronization and lifetime rules;
- remove the mirrored host/device representation from the new API;
- provide clear CPU-build stubs or diagnostics.

Test moves, copies, destruction, repeated transfers, wrong-device access, and
CPU-only compilation.

Exit criterion: CUDA storage is owned safely and is usable without public raw
allocation fields.

### Phase 9 — Migrate CUDA operation bundles

Migrate and validate in small bundles:

1. elementwise operations and reductions;
2. matrix multiplication and losses;
3. optimizer kernels;
4. convolution and pooling;
5. FFT.

Every migrated CUDA operation must be compared with the CPU path for forward
and backward behavior. Do not broaden vendor-library dependencies during this
phase without a separate decision.

Exit criterion: all currently supported CUDA behavior works through the new
public API.

### Phase 10 — Qualify extensions and downstream use

Build explicit consumer tests:

- a normal consumer whose translation unit includes neither Eigen nor CUDA;
- an Eigen-based CPU custom operation;
- a CUDA custom kernel using borrowed input/output views;
- optimizer checkpoint and restore;
- installed-package consumption;
- a focused `CppResist` migration spike.

Use the migration spike to rewrite the `CppResist` refactor plan around:

- `Tensor` and `Variable`;
- registered `Module` parameters;
- stable optimizer state;
- custom-operation extension APIs;
- explicit device boundaries.

Do not permanently maintain compatibility for direct graph-field or raw-buffer
mutation discovered in the spike.

Exit criterion: a representative downstream pipeline works without private
library access.

### Phase 11 — Remove legacy interfaces and freeze versioned API

After all in-repository and downstream migration tests pass:

- remove public `Mat`;
- remove `VarPtr`;
- remove mutable public graph fields;
- remove normal-API raw CUDA pointers;
- remove duplicated host/device optimizer state;
- remove old implementation paths;
- retain root `autograd.h` only as a compatibility include path that forwards
  to the canonical umbrella header;
- document breaking changes and migration examples;
- assign a version to the new public API.

Exit criterion: one implementation path remains and no normal public header
leaks Eigen, CUDA, or graph internals.

## 10. Validation Matrix

| Area | CPU build | CUDA build | Installed consumer | CppResist spike |
|---|---:|---:|---:|---:|
| Tensor ownership/copies | required | required | required | required |
| Core forward operations | required | parity required | smoke | required |
| Autograd/grad checks | required | parity required | smoke | required |
| Modules/parameters | required | parity where supported | smoke | required |
| Optimizer continuation | required | required | smoke | required |
| FFT | required | required when enabled | optional | as used |
| Eigen extension | required | not applicable | required | as used |
| CUDA extension | not applicable | required | required | as used |
| CPU-only header hygiene | required | required | required | required |

The existing core, NN, convolution, extension, diffusion, smoke, and CUDA tests
should be retained or replaced by equivalent tests rather than discarded.

## 11. Completion Criteria

The refactor is complete when:

- normal public headers expose no Eigen or CUDA types;
- `Tensor` contains numerical/device state only;
- `Variable` graph internals are private;
- modules expose parameters through registration and traversal;
- optimizers expose stable state without backend-specific duplicate fields;
- CPU-only and CUDA-enabled builds both pass their applicable suites;
- public-header and installed-package consumer tests pass;
- a custom Eigen operation and custom CUDA operation work through extension
  APIs;
- the representative `CppResist` migration spike needs no private access;
- legacy public fields and duplicate implementation paths have been removed;
- documentation clearly states supported operations, devices, ownership,
  copying, graph lifetime, and extension lifetimes.

## 12. Risks and Controls

### Risk: accidental framework expansion

Control: enforce the scope guard and require a separate design decision for
dtypes, distributed execution, runtime backend plugins, allocator frameworks,
or new vendor libraries.

### Risk: hidden downstream requirements

Control: inventory `CppResist` usage in Phase 0 and prove extension APIs with a
focused migration spike before legacy removal.

### Risk: CPU/CUDA semantic drift

Control: use shared validation rules and systematic parity tests for every
migrated CUDA operation.

### Risk: long-lived dual architecture

Control: migrate in vertical slices, record removal criteria per phase, and
remove the old path immediately after all consumers of a slice move.

### Risk: opaque abstractions make a teaching library harder to read

Control: use PImpl only for storage and graph encapsulation, keep operation
dispatch direct, and avoid a general runtime or public backend hierarchy.

### Risk: refactoring minimal_tensor simultaneously

Control: finish and stabilize the smaller `minimal_autograd` API first. Apply
lessons to `minimal_tensor` later through a separate plan rather than coupling
both migrations or extracting a shared core prematurely.

## 13. Recommended Execution Order

Start with Phase 0 and Phase 1. Then implement one complete CPU vertical slice
through Phases 2–4 before migrating all operations. Stabilize modules,
optimizers, and FFT on CPU before introducing the new CUDA storage model.

The first implementation milestone should therefore be:

```text
external consumer
    -> Tensor(CPU)
    -> Variable
    -> add/mul/sum
    -> backward
    -> custom operation
```

This milestone tests the important architectural boundaries with the smallest
possible feature surface.
