# Phase 11a Expert Dependency Boundary Plan

## Purpose

Phase 11a creates the first executable boundary required by the merged
`PHASE11_API_FREEZE_PLAN.md`: a normal `autograd::autograd` consumer must not
receive Eigen/CUDA headers or the `AUTOGRAD_USE_CUDA` definition, while a
temporary legacy-shaped consumer can explicitly request the dependencies it
needs through `autograd::autograd_expert`.

This is a build/package boundary only.  It preserves the legacy
`Mat`/`VarPtr` implementation and API until the separately planned CppResist
migration and Phase 11d deletion.  It does not change tensor layout,
numerics, checkpoints, CUDA provider selection, public C++ operation APIs, or
CppResist source.

## Current evidence and decisions

Current `CMakeLists.txt` publishes all three concerns through the normal
target:

- `Eigen3::Eigen` is `PUBLIC`;
- CUDA builds publish `AUTOGRAD_USE_CUDA` and `CUDA::cudart`; and
- the installed `autogradConfig.cmake` unconditionally runs
  `find_dependency(Eigen3 REQUIRED)`.

The current replacement-only consumer already proves normal headers do not
include Eigen/CUDA.  The current CppResist expert boundary, however, still
uses `autograd.h`, `Mat`, `VarPtr`, and legacy CUDA state.  The Phase 11 plan
requires a temporary, source-preserving expert fixture for that consumer shape
before CppResist migration begins.

The 11a target contract is:

```text
normal CPU CMake/package consumer
  find_package(autograd CONFIG REQUIRED)
  target_link_libraries(app PRIVATE autograd::autograd)
  -> no Eigen/CUDA find requirement, include, or AUTOGRAD_USE_CUDA macro

normal CUDA CMake/package consumer
  -> no Eigen/CUDA include, CUDA language requirement, or AUTOGRAD_USE_CUDA
     macro; a STATIC CUDA package may propagate CUDA::cudart solely to link
     its already-built archive, while a SHARED CUDA package self-resolves it

explicit expert consumer
  find_package(autograd CONFIG REQUIRED COMPONENTS expert)
  target_link_libraries(app PRIVATE autograd::autograd_expert)
  -> legacy Eigen/CUDA requirements only when that built package needs them
```

`autograd::autograd_expert` is an INTERFACE target.  It links the normal
library plus the copied Eigen extension dependency and, in CUDA builds, the
CUDA runtime and legacy CUDA compile definition needed by the still-live
legacy headers.  It is temporary compatibility containment, not a forwarding
header, adapter, second library, or normal-consumer fallback.

The package must export normal and expert targets separately (or equivalently
load the expert export only for the `expert` component).  The normal **CPU**
config must not call `find_dependency(Eigen3)` or
`find_dependency(CUDAToolkit)`.  A normal CUDA **static** package may find
only `CUDAToolkit` to satisfy its required runtime link interface; a normal
CUDA shared package must not need that configure-time dependency.

`autogradConfig.cmake` must inspect `autograd_FIND_COMPONENTS`: it includes
the normal export first, recognizes only `expert`, and sets
`autograd_expert_FOUND` false for an unknown/unavailable component.  Only when
`expert` is requested does it load the expert export and call
`find_dependency(Eigen3)`.  It calls `find_dependency(CUDAToolkit)` only when
both the installed package has CUDA and the selected target form requires it
(expert, or normal static CUDA runtime linkage).  The implementation must test
each of those component/build-form combinations rather than rely on an
install-time CUDA flag alone.
The exact CMake technique is deliberately gated: `PRIVATE` linking a static
library can still emit a `LINK_ONLY` imported-target requirement.  The
implementation must prove the normal installed consumer configures with
Eigen discovery disabled; if it does not, it must use a narrowly scoped
build-only Eigen include/usage mechanism that removes the exported normal
requirement.  It must not weaken the normal-consumer gate or make the normal
target depend on the expert export.

## Delivery topology and scope

This plan-only PR changes only this file and the narrowly scoped static/shared
CUDA clarification in `PHASE11_API_FREEZE_PLAN.md`.  After owner approval,
one `minimal_autograd` implementation PR delivers Phase 11a.  It may touch
only:

- `CMakeLists.txt` and `cmake/autogradConfig.cmake.in`;
- new or changed package/export CMake helper files under `cmake/`;
- `tests/consumer/{find_package,add_subdir,eigen_custom,cuda_custom,legacy_expert}/**`;
- focused CMake boundary test support under `tests/consumer/**`;
- consumer-facing target documentation in `README.md` and
  `tests/consumer/README.md`.

No legacy header/source, replacement operation/module implementation,
checkpoint fixture, CppResist file, package RPATH policy, or version number is
in scope.  A missing replacement API, a CppResist source change, or a need for
a legacy bridge stops 11a and requires a new approved plan.

## Red gate

Before edits, record all of the following from an installed CPU package and an
installed CUDA package when a visible CUDA environment is available:

1. Normal `find_package(autograd)` reads a config that requires Eigen.
2. CUDA normal-consumer compile commands contain `AUTOGRAD_USE_CUDA` and
   receive CUDA linkage.
3. No `legacy_expert` consumer fixture exists; current legacy-shaped source
   can only link the normal target.
4. The current CMake export/config has one target set, so an expert dependency
   cannot be loaded independently.

These failures are right-reason observations, not tests to preserve.

## Implementation phases and gates

### 11a.1 — split target/export contract

Create the build-tree `autograd::autograd_expert` INTERFACE target and install
it through an explicit `expert` package component/export.  Keep the existing
normal target name and its build artifact unchanged.  Move Eigen/CUDA usage
out of its public interface only after the installed normal-package configure
gate is executable.

Green criteria:

- `find_package(autograd CONFIG REQUIRED)` imports only the normal target;
- with `CMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE`, the normal installed
  replacement-only consumer configures, builds, and runs on CPU;
- `find_package(autograd CONFIG REQUIRED COMPONENTS expert)` imports the
  expert target and resolves Eigen; CUDA expert consumption resolves the CUDA
  runtime only for a CUDA-built package;
- CUDA shared normal consumers configure without CUDAToolkit discovery or
  CUDA language/macro leakage; CUDA static normal consumers receive only the
  `CUDA::cudart` link requirement and still receive no CUDA language/macro
  leakage;
- source-tree consumers retain their existing no-test/no-example inheritance
  property, and the normal source-tree consumer has no Eigen/CUDA macro leak.

Stop if CMake's exported static target still requires Eigen, a CUDA static
package needs more than its `CUDA::cudart` runtime link requirement, a CUDA
shared package needs CUDAToolkit discovery, an expert component cannot be
represented on supported CMake versions, or target splitting creates a
duplicate provider/cycle.

### 11a.2 — legacy-shaped expert qualification

Add `tests/consumer/legacy_expert/`: a minimal consumer using current
`autograd.h`, `Mat`, `VarPtr`, and (for CUDA) the existing legacy CUDA helper
surface.  It changes only the link target to `autograd::autograd_expert`; it
must not add a bridge, adapter, or direct private include.

Green criteria:

- CPU installed and source-tree fixtures configure, build, and run through
  the explicit expert target;
- CUDA installed and source-tree fixtures compile with the required legacy
  CUDA definition and run on a visible device when one is required by the
  fixture;
- normal and expert compile databases prove opposite macro polarity:
  normal has no `AUTOGRAD_USE_CUDA`; expert has it only in CUDA builds;
- a source scan proves normal consumer fixtures and canonical headers contain
  no Eigen/CUDA includes or legacy aliases, while legacy names occur only in
  the named expert fixture and existing legacy implementation paths.

### 11a.3 — full boundary qualification

Run the existing CPU test inventory, package/install consumer matrix, and
affected documentation examples.  Run the CUDA build/test and consumer matrix
on a visible GPU through OpenCode.  Inspect generated target exports and Linux
dynamic dependencies; report only the actual library relationship, never an
RPATH guarantee.

Green is all CPU checks passing, device-visible CUDA checks passing without
skipping a required parity claim, no normal dependency/macro/header leak, one
provider graph, and the legacy fixture succeeding exclusively through the
expert target.

## Risk, routing, and publication

Safety risk is L3: this changes package/target dependency propagation while
the legacy CppResist consumer remains live.  Implementation difficulty is
difficult because CMake export behavior differs between static/shared and
CPU/CUDA builds.

Before implementation, run one bounded Sol-expert preflight on CMake export
and component behavior.  Luna performs the implementation.  Claude performs
the required independent read-only review of the final CMake/package boundary.
Use a three-attempt cap, ten-minute progress checkpoints, and a 45-minute
maximum wait.  Stop on a dependency leak, missing package component, legacy
fixture failure, duplicate provider, changed runtime behavior, no visible GPU
for a required CUDA claim, or any out-of-scope product change.

Publish one implementation PR only after all gates pass, with
`Phase-gate: manual`.  It remains draft until final validation and Claude
review are green.  Owner approval marks it ready; separate owner approval
merges it.  After merge, stop for the separately planned CppResist 11b spike.

## Grilled-Me review

Assumptions confirmed: existing normal consumers use the replacement surface;
the legacy facade needs Eigen and legacy CUDA compile definitions; current
CMake publishes those dependencies globally.

Risks identified: a private CMake dependency can still leak from a static
export; an expert target in the same normal export can force `find_package`
to resolve Eigen; a fixture that rewrites source would hide the actual CppResist
compatibility risk.

Simplification applied: one normal target, one temporary explicit expert
target, one component-gated package export, and one source-preserving fixture.
No adapter, new runtime backend, or CppResist edit is added.

Surviving owner decisions: approve the exact expert target/component naming
and the intentional temporary legacy fixture before implementation.
