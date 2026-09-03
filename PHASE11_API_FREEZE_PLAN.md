# Phase 11 API Freeze and Legacy Retirement Plan

## Purpose and current evidence

Phase 11 is the final `minimal_autograd` compatibility removal.  It cannot
begin by deleting the legacy Eigen/`Var` facade: CppResist R7 is merged at
`cebb71e8089b7a63e213cdb65f8a81a401b730ac`, but its expert headers still
publish and use `ag::Mat`, `ag::VarPtr`, legacy public graph fields, and the
legacy CUDA mirror API.  R7 deliberately did not migrate those contracts.

The merged replacement surface already provides `Tensor`, `Variable`, core
operations, registered modules, optimizer state, custom-variable construction,
and explicit CUDA views.  The in-tree installed consumers qualify those
replacement contracts.  They do not prove that CppResist can leave the legacy
facade.

This plan therefore separates the missing downstream migration from the final
breaking deletion.  It authorizes no product code by itself.

## Decisions and non-goals

The intended final boundary is:

```text
normal consumer -> autograd::autograd -> no Eigen/CUDA headers or macros
expert consumer -> explicit extension target(s) -> only required dependency
```

- The normal `autograd::autograd` shared/static target must not propagate
  `Eigen3::Eigen`, `AUTOGRAD_USE_CUDA`, raw CUDA types, or legacy
  graph/storage fields.  A CUDA-built **static** archive may propagate only
  the technically required `CUDA::cudart` link dependency; it must not expose
  CUDA headers, a CUDA language requirement, or an ABI-changing macro.  A
  CUDA-built shared library must self-resolve that runtime dependency.
- The final expert-target spelling is proposed as
  `autograd::autograd_expert`.  It is an explicit opt-in boundary, not a
  compatibility umbrella.  Its exact composition is a Phase 11a owner gate:
  it may expose the copied Eigen extension and/or borrowed CUDA extension only
  where a migrated consumer demonstrably needs them.
- Root `autograd.h` remains only a forwarding include to the canonical stable
  umbrella after the freeze.  It must not re-export legacy headers.
- No forwarding alias, deprecated `Var` adapter, raw CUDA normal API, or
  second optimizer state representation survives the freeze.
- This plan does not change numerical algorithms, tensor layout, checkpoint
  format, CUDA provider selection, CppResist behavior, package RPATH policy,
  or `minimal_tensor`.

The API removal is intentionally source-breaking for legacy consumers.  The
owner must separately accept that compatibility boundary before the deletion
PR.  Unknown external consumers cannot be inventoried locally.

## Delivery topology and dependencies

Four implementation PRs are required after this plan-only PR.  They are split
because they have different owners, rollback boundaries, and executable
evidence.  A replacement-surface gap found by 11b is a topology deviation:
11b stops, and the owner must approve a separate narrow feature plan before a
revised Phase 11 plan can continue.  That feature is not silently added to
this delivery.

1. **11a — expert dependency boundary (`minimal_autograd`).** Add and qualify
   the explicit extension target contract without deleting legacy API.  This
   is the only phase allowed to change target dependency propagation.
2. **11b — CppResist replacement-API spike (CppResist).** Use only installed
   replacement headers plus explicitly approved extensions to prove one CPU
   and one CUDA resist/training vertical.  Stop if the replacement surface
   lacks a required contract; a separate approved `minimal_autograd` PR must
   add that contract before continuing.
3. **11c — CppResist migration (CppResist).** Migrate every public and
   in-repository CppResist consumer from `Mat`/`VarPtr` and direct legacy
   CUDA/optimizer/graph fields.  Retire each migrated legacy subsystem there.
4. **11d — legacy deletion and version freeze (`minimal_autograd`).** Delete
   the legacy facade only after 11b and 11c are merged and requalified against
   the exact 11d base.

This plan PR changes only `ARCHITECTURE_REFACTOR_PLAN.md` and this file.  Each
implementation PR is independently reviewable and revertible.  No phase may
start automatically: 11a, 11b, 11c, and 11d each require its own approved
implementation plan and owner gate.

## Phase 11a — expert dependency boundary

Scope draft: `CMakeLists.txt`, `cmake/autogradConfig.cmake.in`, explicit
extension-target CMake support, installed consumer fixtures under
`tests/consumer/**`, focused CMake boundary tests, and documentation required
to name the supported targets.  It does not delete legacy headers or alter
`Tensor`/`Variable` algorithms.

Red gate: the current normal installed target requires Eigen unconditionally,
and CUDA builds publicly propagate `AUTOGRAD_USE_CUDA` and CUDA dependencies.

Green gate:

- a normal CPU installed consumer includes only canonical headers, links only
  `autograd::autograd`, and compiles without Eigen or CUDA headers/macros;
- an explicit Eigen/CUDA extension consumer links the named extension target
  and receives only the dependency it actually uses;
- a legacy-shaped `Mat`/`VarPtr` fixture representing the unmigrated
  CppResist expert boundary compiles and links through the explicit expert
  target after only its CMake link line changes; this temporary proof retains
  no dependency on the normal target and adds no source-level legacy bridge;
- CPU and CUDA configured target graphs have one `autograd` provider, no
  dependency cycle, and no public ABI-changing backend macro;
- source-tree and installed consumer tests pass in their applicable CPU/CUDA
  environments.

Stop for a target-name/API decision, package-export limitation, hidden macro
leak, duplicate provider, or any need to preserve legacy public fields.

## Phase 11b — CppResist replacement-API spike

Scope draft: a separate CppResist plan and one temporary, representative
consumer vertical.  The spike must use `Tensor`/`Variable`, registered
parameters, optimizer state APIs, `make_custom_variable`, and borrowed CUDA
views where needed.  It must not modify the legacy facade or add a bridge.

Red gate: CppResist R7 expert headers still include `autograd.h` and publish
`ag::Mat` / `ag::VarPtr`; CPU/CUDA resist code directly accesses legacy graph
and CUDA fields.

Green gate:

- the spike builds through the installed namespaced target(s), not
  `add_subdirectory` internals;
- one CPU forward/backward/update/checkpoint path preserves the documented
  `(y, x)` image-value contract;
- one visible-GPU CUDA forward/backward/update path uses owned Tensor storage
  plus borrowed views only, with explicit synchronization and no raw owning
  pointer or legacy mirror access;
- parameter names/order and optimizer snapshot/restore are deterministic;
- normal CppResist stable headers remain free of Eigen/CUDA/legacy exposure.

Stop on a missing replacement contract, numerical/checkpoint mismatch, device
or lifetime ambiguity, or any request to add a permanent legacy bridge.

## Phase 11c — CppResist migration

Scope is defined only after the merged 11b spike gives a complete consumer
inventory.  It includes the CppResist public headers, sources, applications,
tools, tests, and CUDA code that still consume the legacy facade; it excludes
new optics/resist algorithms and package policy.

Green gate: no CppResist public or in-repository consumer includes legacy
`autograd.h` for `Mat`/`VarPtr`, accesses legacy graph/CUDA/optimizer fields,
or requires public `AUTOGRAD_USE_CUDA`; CPU and device-visible CUDA suites pass
against the exact installed 11a package.  The CppResist plan must record the
final residual scan and dynamic dependency evidence.

## Phase 11d — deletion and version freeze

Scope draft: legacy public headers and implementations, their tests/examples,
legacy CMake source lists and dependency propagation, root umbrella forwarding,
canonical API/version documentation, and focused source/build/package boundary
tests.  No CppResist source changes occur in this phase.

Red gate: before deletion, prove the legacy source/header exports, fields, and
target propagation remain present while 11c's exact migrated consumer proves
they are unused.

Green gate:

- source scans and compile-boundary tests prove `Mat`, `VarPtr`, legacy graph
  fields, raw CUDA normal APIs, public `AUTOGRAD_USE_CUDA`, duplicate optimizer
  state, and old implementation paths are absent from the supported surface;
- root `autograd.h` forwards only to the canonical stable umbrella;
- CPU-only installed normal consumer has no Eigen/CUDA dependency;
- explicit extension consumers still build only through their named target;
- CPU full CTest, installed consumers, and affected examples pass;
- CUDA full CTest plus a visible-GPU downstream requalification pass with no
  skipped required device parity claim;
- exported symbols and dynamic dependencies match the intended single-provider
  graph; no package/RPATH guarantee is claimed.

The deletion PR is L4 safety risk and difficult.  It uses a bounded
Sol-expert preflight, Luna implementation, and final independent read-only
review; each has a three-attempt cap, ten-minute progress checkpoints, and a
45-minute maximum wait.  Stop on any residual legacy consumer, replacement
gap, package regression, CUDA visibility failure, incompatible checkpoint
contract, or scope expansion.  Rollback is a revert of the one deletion PR;
no partial deletion is published.

## Plan review

[Grilled-Me Review]

Assumptions confirmed: R7's merger removed only CppResist's local umbrella;
the current CppResist expert surface still uses the legacy `minimal_autograd`
facade.  The replacement API and standalone consumer checks exist, but they
do not establish the CppResist migration.

Risks identified: deleting first would break the live downstream expert API;
making Eigen/CUDA private without an explicit extension target would break
legitimate extension consumers; a migration spike can expose a missing custom
operation or device-lifetime contract.

Simplification applied: no speculative adapter or broad all-at-once migration;
one dependency-boundary PR, one spike, one CppResist migration, then one
deletion/freeze PR.

Surviving concerns (owner gate): accept the eventual legacy source break and
the proposed explicit expert target name before Phase 11a; accept each
implementation plan separately.
