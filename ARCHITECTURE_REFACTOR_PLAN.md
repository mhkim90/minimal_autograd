# minimal_autograd Architecture Refactor — Final Plan

## 1. Status

This document defines the target architecture and phased migration plan for a
full refactor of `minimal_autograd`.

It is a planning document, not authorization to implement all phases at once.
Each phase should be implemented and reviewed separately, with compatibility
tests established before old interfaces are removed.

Current delivery state:

- Shape/Device, hidden CPU Tensor, and private Variable/tape foundations are
  merged;
- CPU N-D operations and losses are merged through PR #36;
- Phase 4 is closed: Gate 4.5 runs the registered CPU contract inventory in
  hosted CI;
- the OOP training stack is merged through PR #37;
- the CPU spatial, normalization, and diffusion replacement APIs are merged
  through PR #38;
- the OOP CPU complex and FFT APIs are merged through PR #39;
- the CUDA Tensor foundation is merged through PR #40;
- the OOP CUDA elementwise and reduction bundle is implemented in the current
  branch, pending review.

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

`Variable` operations are N-dimensional by contract. No operation in the new
API may reject an input merely because its rank is greater than two. Operations
that need axes expose them explicitly; operations that need matrix dimensions
use the final two axes and preserve leading batch axes. The legacy Eigen API
may retain 2-D adapters during migration, but those adapters do not define the
new core API.

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
- last-axis-contiguous element order (canonical row-major):
  `stride[n-1] = 1` and `stride[i] = stride[i + 1] * shape[i + 1]`;
- rank-2 storage uses flat index `row * columns + col`;
- ordinary copies share storage;
- `clone()` makes an independent deep copy;
- `reshape()` changes logical metadata without reordering elements and shares
  the same storage;
- one authoritative storage allocation per tensor;
- no public Eigen matrix member;
- no public raw CUDA allocation member;
- no host mirror that can silently become stale.

Tensor storage is deliberately shared-mutable rather than copy-on-write:

- explicit `copy_from_host()` mutation is visible through ordinary aliases;
- callers use `clone()` when mutation isolation is required;
- optimizers and backend kernels update parameters through a narrow internal
  mutation boundary, not public writable pointers or element references;
- public API documentation identifies every mutating operation and its
  aliasing effect.

This policy preserves parameter identity and keeps optimizer behavior
understandable. Hidden copy-on-write and unrestricted public mutation are both
out of scope.

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
Variable sum(const Variable&, const std::vector<int>& axes, bool keep_dims);
Variable relu(const Variable&);
Variable softmax(const Variable&, int axis = -1);
Variable reshape(const Variable&, const Shape&);
Variable transpose(const Variable&, int axis0, int axis1);
Variable concat(std::vector<Variable>, int axis = 0);
Variable slice(const Variable&, int axis, int64_t start, int64_t length);
```

N-D operation rules:

- elementwise operations preserve the complete logical shape;
- broadcasting aligns trailing axes and permits an input dimension of one;
- negative axes are normalized against the input rank and validated;
- `reshape` accepts any validated `Shape` with the same element count;
- `transpose` swaps two selected axes;
- reductions, softmax, cumulative operations, slicing, concatenation, and
  splitting operate on explicit axes;
- `matmul` requires rank two or greater, treats the final two axes as matrix
  dimensions, and initially requires identical leading batch dimensions;
- cross-entropy treats the final axis as classes and averages over all leading
  sample axes;
- dense storage remains last-axis-contiguous (canonical row-major), with the
  final axis varying fastest;

Convenience names such as `row_slice`, `col_slice`, and `hcat` may remain as
thin compatibility wrappers over axis-aware N-D operations. They must not own
separate 2-D-only kernels.

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
- N-dimensional `Tensor` and `Variable` operations with explicit axis
  semantics;
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

Characterization is not automatically a permanent guarantee. The behavior
contract labels each observed behavior as one of:

- **guaranteed** — the replacement API must preserve it;
- **legacy-only** — retained only while the legacy facade exists;
- **known defect / explicitly unpinned** — observed by a test but intentionally
  not carried into the replacement API.

Repeated `backward()` calls must accumulate one newly computed gradient pass
per call. Legacy amplification caused by feeding previously stored
intermediate gradients into a later traversal—including the Conv2d `3x`
result after two calls—is characterized but not guaranteed.

Also create a small `CppResist` integration inventory: which tensor, graph,
optimizer, Eigen, and CUDA details it currently touches. This is an input to
the extension API, not a reason to preserve unsafe public fields.

Exit criterion: every guaranteed behavior maps to an executable test, every
legacy-only or unpinned behavior is labeled, and applicable CPU contract tests
run in CI rather than merely compile.

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
- define last-axis-contiguous order and reshape semantics;
- define shared-mutable aliasing and the controlled internal mutation boundary;
- add checked host import/export;
- add reshape and basic factory functions;
- ensure errors are runtime validations, not debug-only assertions.

Test ownership, alias-visible mutation, clone isolation, copying, empty
tensors, invalid shapes, overflow, host round-trips, byte order, and reshape
rules.

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

Each `backward()` traversal computes propagation from a fresh seed. Existing
stored gradients may be accumulated into committed node gradients, but they
must not become inputs to the next traversal. Two identical calls therefore
add two identical leaf-gradient passes unless the caller changes the graph or
upstream gradient.

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

The migrated API must be N-dimensional:

- no new operation is rank-2-only;
- axis-taking operations accept normalized positive or negative axes;
- matrix multiplication batches over identical leading dimensions;
- broadcasting follows trailing-axis compatibility;
- existing 2-D behavior remains a tested compatibility subset.

For each operation, require:

- forward parity with the characterized result;
- analytical or finite-difference gradient checks;
- invalid-input tests;
- rank-1, rank-3, or rank-4 coverage as applicable;
- axis, broadcasting, and batched-shape validation;
- shared-node and gradient-accumulation coverage where applicable.

Exit criterion: the new API has one N-D kernel path. Any still-live legacy
implementation has a named remaining consumer and retirement gate.

#### Legacy retirement rule for every migrated slice

For each operation, module, optimizer, or backend bundle:

1. establish replacement API and consumer coverage;
2. route the compatibility facade through the replacement implementation when
   that remains simpler than a hybrid adapter;
3. remove the superseded implementation immediately after its last consumer
   moves;
4. if adaptation is temporarily unsafe, record the exact remaining consumer
   and deletion gate, and add no new feature to the legacy path.

The goal is one implementation with two facades during compatibility—not two
implementations until final cleanup. A facade may remain until downstream
qualification; duplicated kernels or state should not.

Current recorded exception: the legacy `Mat`/`VarPtr` operation path remains
for legacy modules and `CppResist`, whose graph representation has not yet
migrated. It receives no new features. Its core operations are adapted or
deleted as the Phase 5–10 consumers move, with final facade deletion in
Phase 11.

#### Phase 5a transitional naming

Phase 5a places the new `Module`/`Linear`/`SGD` declarations under
`ag::nn` and `ag::optim` (in `include/autograd/core/module.h` and
`include/autograd/core/optim.h`) instead of the flat `ag::Module` /
`ag::Linear` / `ag::SGD` names the legacy facade already owns. The new
namespace prefix is a deliberate anti-collision choice for the
transitional period. Phase 11 will remove the legacy facade and expose
flat canonical aliases; until then, `using`-aliases or header
re-exports must not be added (doing so would widen scope and obscure
the legacy deletion gate).

#### Phase 5a legacy bridge exception

Building a bridge between the new `Variable` graph and the legacy
`VarPtr` graph would require adapter code that crosses two graph
representations. Adapting either side is unsafe during Phase 5a
because (a) the new graph's mutation boundary is still being
calibrated, and (b) the legacy graph retains legacy
amplification/rollback behavior the new graph deliberately does not
replicate. Phase 5a therefore does NOT edit, adapt, or bridge to the
legacy `Module`/`Linear`/`Sequential`/`SGD`/`Adam` implementations; the
legacy path remains untouched and continues to receive no new
features.

The legacy facade's retirement is gated on the migration of every
actual consumer — spatial modules (Phase 6), CUDA op bundles
(Phases 8–9), and the CppResist migration spike (Phase 10) — not on
the OOP training stack alone. Phase 5b (Sequential + Adam + state
snapshot/restore) is a prerequisite for the OOP training slice but
does NOT, by itself, unlock any legacy deletion. Final facade deletion
remains Phase 11's responsibility and runs only after the in-tree
consumers and downstream qualification have moved.

### Gate 4.5 — Executable-contract closeout

Before Phase 5 feature work:

- run the default and advanced registered CTest inventories in hosted CPU CI;
- confirm every guaranteed Phase-0 row maps to a test that CI executes;
- label code-inspection-only and CUDA-hardware-only rows honestly;
- keep legacy repeated-backward amplification tests only as legacy
  characterization;
- verify the replacement Variable test enforces fresh-seed, one-pass-per-call
  accumulation;
- verify Tensor tests cover byte order, alias-visible mutation, reshape
  sharing, and clone isolation.

Exit criterion: the words “executable contract” describe tests that actually
run, and remaining hardware/manual gaps are explicit.

### Phase 5 — OOP training stack, vertical gate first

Do not begin with a bulk module hierarchy. First prove the complete training
boundary using only the replacement API.

#### Phase 5a — Linear + SGD vertical training gate

- implement the minimal parameter registry and deterministic traversal;
- migrate `Linear`;
- implement controlled parameter mutation through the private Tensor/Variable
  boundary;
- migrate SGD;
- run a complete two-layer MLP or equivalent nontrivial training loop:

```text
Tensor input
    -> Linear
    -> activation
    -> Linear
    -> loss
    -> backward
    -> SGD step
    -> measurable loss decrease
```

This gate verifies parameter identity, alias-visible optimizer updates,
gradient clearing, deterministic parameter order, and absence of Eigen in the
consumer translation unit.

#### Phase 5b — Composition and Adam

- migrate `Sequential` and nested-module registration;
- migrate Adam;
- place moment tensors with their parameters;
- define explicit optimizer state snapshot and restore;
- validate complete state before mutating a live optimizer;
- preserve exact SGD and Adam trajectories within documented tolerance.

Exit criterion: an end-to-end MLP trains through the new API, nested parameter
discovery is deterministic, and optimizer state can be restored without
accessing implementation fields.

**Phase 5b delivery (current):** `nn::Sequential` composes children with
deterministic numeric names (`0`, `1`, ...); `nn::Module::register_module`
adds nested composition with empty / null / duplicate / cross-kind
collision rejection; `parameters()`, `named_parameters()`, and `zero_grad()`
recurse depth-first with dotted-prefix names (`0.weight`, `0.0.bias`,
`child.weight`, ...). `optim::Adam` validates hyperparameters in the
constructor, pre-validates every eligible parameter in `step()`, computes
all new moment/parameter values into temporaries, then commits — a
failure cannot leave state half-updated. `state()` returns deep-cloned
moments; `load_state()` validates the entire snapshot (step count,
hyperparameters, moment counts, shapes, devices) before mutating any
live state, so a failed load leaves both the optimizer and the parameter
values unchanged. `nn::ReLU` is a parameter-free module that routes
through the public `ag::relu` free function.

Phase 5b is the last OOP-stack slice before the spatial/CUDA/CppResist
migration. Phase 5b does NOT, by itself, unlock any legacy facade
deletion; the legacy facade's retirement still depends on the Phase 6
spatial modules, Phase 8–9 CUDA bundles, and Phase 10 CppResist spike
moving their actual consumers, with the final facade deletion remaining
Phase 11's responsibility.

### Phase 6 — CPU spatial and special modules

Migrate in bounded bundles:

1. `Conv2d` and `MaxPool2d`;
2. `DepthwiseConv2d`, `AvgPool2d`, and `NearestUpsample2d`;
3. `GroupNorm`, including parameter and gradient behavior;
4. `randn`, sinusoidal time embeddings, and `q_sample`.

Each bundle must have a replacement consumer test before its corresponding
legacy implementation is adapted or retired.

Exit criterion: all current CPU modules and diffusion helpers use private
registered state and the new public value types.

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
- delete the compatibility facade after it has become a thin adapter;
- remove mutable public graph fields;
- remove normal-API raw CUDA pointers;
- remove duplicated host/device optimizer state;
- remove old implementation paths;
- make Eigen an implementation-only (`PRIVATE`) target dependency;
- remove the public `AUTOGRAD_USE_CUDA` compile definition, or replace it with
  a narrowly documented installed-package feature mechanism that does not
  expose conditional graph/storage fields;
- verify a CPU-only installed package does not require Eigen or CUDA for a
  normal consumer using only the canonical API;
- in a CUDA-enabled package, propagate only linker/runtime dependencies that
  are technically required—never CUDA headers, raw types, or ABI-changing
  compile definitions;
- retain root `autograd.h` only as a compatibility include path that forwards
  to the canonical umbrella header;
- document breaking changes and migration examples;
- assign a version to the new public API.

Exit criterion: one implementation path remains, CPU-only consumers receive no
Eigen/CUDA dependency, CUDA-enabled consumers receive no public CUDA types or
ABI-changing macro, and no normal public header leaks Eigen, CUDA, or graph
internals.

## 10. Bundled Pull Request Delivery Plan

The phased migration is delivered as 12 `minimal_autograd` PRs. Each PR may
span multiple internal phase-gated commits; the PR is the unit of review,
merge, and history.

1. **PR #33 — Shape + Device foundation.** Introduce `Shape` and `Device`
   value types without changing the existing public API.
2. **PR #34 — Hidden-storage CPU Tensor + Eigen extension.** Move CPU tensor
   storage behind PImpl, add checked host import/export, and ship
   `extension/eigen.h` for explicit CPU interoperation.
3. **Autograd core.** Private `VariableNode`, tape, backward, `detach`,
   `zero_grad`, and the narrow vertical slice (`add`, `mul`, `scale`, `sum`,
   custom-op boundary).
4. **CPU N-D operations.** Remaining ops, activations, axis-aware reductions
   and layout transforms, batched matmul, broadcasting, losses, and their
   gradients with 2-D parity plus N-D validation. Close the bundle by making
   hosted CPU CI execute the complete registered contract inventory.
5. **OOP training stack.** Begin with the Linear + SGD end-to-end training
   gate, then add module composition, Adam, and optimizer snapshot/restore.
6. **CPU spatial and special modules.** `Conv2d`, `DepthwiseConv2d`,
   `MaxPool2d`, `AvgPool2d`, `NearestUpsample2d`, `GroupNorm`, and diffusion
   helpers (`randn`, sinusoidal time embeddings, `q_sample`).
7. **CPU complex and FFT.** Minimal complex-tensor representation, forward
   and backward FFT, and documented normalization conventions.
8. **CUDA foundation.** RAII CUDA storage, explicit transfers, `Tensor::to`,
   borrowed CUDA views, and lifetime rules. CPU-only builds expose clear
   diagnostics.
9. **CUDA core compute.** Elementwise ops, reductions, matmul, losses, and
   optimizer kernels against the CPU path.
10. **CUDA spatial and FFT.** `Conv2d`, pooling, and FFT migrated with parity
    tests.
11. **Extension and downstream qualification.** Eigen and CUDA custom-op
    samples, installed-package consumers, optimizer checkpointing, and a
    focused `CppResist` migration spike. The spike rewrites the linked
    `CppResist` repository PR around `Tensor`/`Variable`, registered
    parameters, stable optimizer state, custom-op APIs, and explicit device
    boundaries.
12. **Breaking cleanup and API freeze.** Remove the thin legacy facade and
    public API (`Mat`, `VarPtr`, mutable graph fields, raw CUDA pointers,
    duplicated optimizer state, old implementation paths), make Eigen and
    CUDA implementation dependencies non-public, keep `autograd.h` as a
    forwarding compatibility include, document breaking changes, and assign a
    version to the new public API.

A bundle is split only when its scope is independently reviewable or a
blocker or risk requires separating it. CPU autograd is not combined with
CUDA ownership; CUDA foundation is not combined with CUDA op migration;
legacy removal does not precede downstream qualification.

Normal workflow is sequential PRs from updated `main`, each merged before the
next is opened. Stacking a PR on another is permitted only when explicitly
chosen and noted in the PR description.

## 11. Validation Matrix

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

Building a test executable is not contract execution. CPU CI must run the
complete registered inventory:

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build-advanced --output-on-failure
```

New characterization, Shape/Tensor, autograd-core, and N-D operation tests are
part of this inventory. CUDA parity requires execution on a CUDA-capable
runner before merging CUDA bundles; when hosted CI lacks a GPU, the PR records
the external/manual CUDA command, hardware, and result rather than presenting
a CPU-only build as CUDA validation.

Automated model reviews are review aids, not acceptance evidence. They may be
recorded under review notes, but completion gates rely on executable tests,
sanitizers/static analysis where applicable, reproducible measurements, and
human review when available.

## 12. Completion Criteria

The refactor is complete when:

- normal public headers expose no Eigen or CUDA types;
- a CPU-only installed target does not propagate Eigen or CUDA;
- `Tensor` contains numerical/device state only;
- Tensor memory order, aliasing, and mutation semantics are documented and
  tested;
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

## 13. Risks and Controls

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

Control: migrate in vertical slices, record the last legacy consumer and
removal criterion per slice, adapt the facade when that is simpler, and remove
the old implementation immediately after the last consumer moves. No new
feature is added only to a superseded path.

### Risk: tests compile but contract behavior is not executed

Control: run complete CTest inventories in CPU CI, require explicit CUDA
execution evidence for CUDA bundles, and label code-inspection-only gaps rather
than calling them executable coverage.

### Risk: opaque abstractions make a teaching library harder to read

Control: use PImpl only for storage and graph encapsulation, keep operation
dispatch direct, and avoid a general runtime or public backend hierarchy.

### Risk: refactoring minimal_tensor simultaneously

Control: finish and stabilize the smaller `minimal_autograd` API first. Apply
lessons to `minimal_tensor` later through a separate plan rather than coupling
both migrations or extracting a shared core prematurely.

## 14. Recommended Execution Order

Phases 0–6 establish the CPU Tensor, Variable, operation, training, spatial,
normalization, and diffusion foundation. The next implementation milestone is
Phase 7: migrate complex values and CPU FFT support before beginning the CUDA
Tensor foundation.

### Phase 7 delivery facts (CPU complex + FFT replacement)

The replacement public surface ships in `include/autograd/core/complex.h`,
`include/autograd/core/fft.h`, and the canonical Eigen-free
`include/autograd/core/fft_norm.h`, with implementations in
`src/core/fft.cpp` and a new `detail::tensor_dft2_last2` helper in
`src/detail/tensor_kernels.h`. The legacy `include/autograd/complex.h`,
`include/autograd/fft.h`, and `src/fft.cpp` remain as the compatibility
implementation; the canonical `FftNorm` enum now lives in
`autograd/core/fft_norm.h` so the legacy and replacement free functions
share one definition. Both surfaces coexist in the umbrella
`autograd.h` as overloads by parameter type — `ag::ComplexVar` and
`ag::ComplexVariable`, `ag::fft2(const ComplexVar&)` and
`ag::fft2(const ComplexVariable&)` — so legacy and replacement
consumers compile unchanged. The replacement header set is Eigen-free
and CUDA-free; the FFT free functions are CPU-only and reject non-CPU
devices in code with `std::runtime_error`. The replacement FFT forward
and adjoint both call the same `detail::tensor_dft2_last2` kernel with
swapped `inverse` / `scale_output` flags; each output component is a
custom-op node built through the public `ag::make_custom_variable`
boundary, so no new `OpKind` or speculative graph fields were
introduced. The dedicated replacement test target is `test_oop_fft`,
which validates rank-2 fixtures and round trips, rank-3 / rank-4
batched final-two-axis FFT with explicit independent references,
complex ops on rank > 2, the full 2x2 real/imag Jacobian blocks of
`fft2` against central finite differences for both `sum(real(...))`
and `sum(imag(...))` (off-diagonal blocks catch sign errors in the
real↔imag cross terms), a small rank-3 batched FD gradient test on a
mixed real/imag objective, a spectral-filter end-to-end gradient,
repeated / shared-branch gradient accumulation with an independent
oracle (each branch measured in an isolated fresh graph and the
combined graph's gradient checked against the elementwise sum of the
two isolated branch gradients), invalid shape rejections,
normalization, and row-major batch isolation. Non-CPU device
rejection is enforced in code but is not covered by an executable
test on this branch; it remains a code-inspection item until the CUDA
Tensor foundation makes a non-CPU replacement `Tensor` constructible.
