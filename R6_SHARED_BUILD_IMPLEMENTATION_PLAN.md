# R6.0a Shared Autograd Build Capability Plan

Status: draft, plan-only. This plan is the minimal_autograd prerequisite for
CppResist R6's approved shared-library topology. It authorizes no product edit
until this exact revision is owner-approved and merged as its own plan-only PR.

## Goal and fixed decisions

Add an opt-in build capability so the existing `autograd` target can be built
as a shared library for CppResist, while preserving today's static library as
the default for all existing users.

- Add `AUTOGRAD_BUILD_SHARED`, default `OFF`.
- With `OFF`, build the current static `autograd` library exactly as today.
- With `ON`, build the same `autograd` target as a shared library.
- Preserve the target name `autograd` and alias `autograd::autograd` in both
  modes. CppResist can therefore select shared autograd before its existing
  `add_subdirectory()` call without a target-name fork.
- Do not use global `BUILD_SHARED_LIBS`: the selection must be local to
  autograd and must not alter an embedding project's unrelated libraries.

This is build-topology support only. No public C++ API, CUDA behavior,
numerical operation, source ownership, or dependency contract changes.

## Scope

Implementation may touch only:

- `CMakeLists.txt`, restricted to the option declaration and the existing
  `add_library(autograd ...)` target-kind selection;
- existing CMake test/configuration wiring only if a focused build-variant
  validation cannot be performed using the present `test_core` and
  `test_cuda_core` targets.

Out of scope: C++ headers/sources, target renaming, a second simultaneous
autograd target, `BUILD_SHARED_LIBS`, install/export layout, RPATH/packaging,
ABI/versioning policy, CppResist source changes, new dependencies, and CUDA
kernels.

## Proposed edit

1. Declare `AUTOGRAD_BUILD_SHARED` next to the other local options with
   default `OFF` and documentation that it changes only the `autograd` target
   kind.
2. Derive one private local library-kind variable from that option and pass it
   to the existing `add_library(autograd ...)` call. Its values are exactly
   `STATIC` and `SHARED`; do not duplicate the source list or the target body.
3. Leave the target alias, PIC setting, C++ standard, include directories,
   Eigen/OpenMP/CUDA links, CUDA source selection, test targets, and existing
   `install(TARGETS autograd EXPORT autogradTargets ...)` unchanged.
4. Do not force the option in minimal_autograd itself. CppResist R6.1 will set
   it in its own top-level build before adding minimal_autograd as a
   subdirectory, after this phase is merged.

## Right-reason gates

### Red gate

At the current baseline, configure a fresh CPU build with
`-DAUTOGRAD_BUILD_SHARED=ON`, build target `autograd`, and inspect the produced
artifact. The option is ignored because it does not yet exist, so the artifact
is the static archive. This proves the missing capability rather than a
numerical defect.

### Green gates

1. In a fresh default configuration, build `autograd`; its artifact remains a
   static archive, install it to a temporary prefix, and run the targeted
   existing CPU tests.
2. In a fresh `-DAUTOGRAD_BUILD_SHARED=ON` configuration, build `autograd`;
   its artifact is a shared library, install it to a temporary prefix, and run
   the same targeted CPU tests.
3. In CUDA-capable validation, repeat both configurations with
   `-DAUTOGRAD_USE_CUDA=ON`; validate static/shared artifact type and run
   `test_cuda_core` in each. CUDA source selection and public CUDA definition
   must remain identical apart from library type.
4. Configure CppResist against the merged capability in a later R6.1 phase;
   that is an integration acceptance gate, not part of this implementation PR.

The artifact inspection uses platform-appropriate tooling and paths; no
platform-specific `.so` filename is made part of the public API.

Stop immediately for a changed default target type, duplicate `autograd`
target, changed alias, unexpected public usage requirement, install/export
failure, CUDA-only regression in CPU configuration, or a test failure. Allow
at most three materially different attempts per failed gate.

## Validation matrix

| Configuration | Required evidence |
| --- | --- |
| CPU default | fresh configure; build `autograd`; static artifact; temporary-prefix install; `test_core` |
| CPU shared | fresh configure with `AUTOGRAD_BUILD_SHARED=ON`; build `autograd`; shared artifact; temporary-prefix install; `test_core` |
| CUDA default | fresh CUDA configure; static artifact; temporary-prefix install; `test_cuda_core` |
| CUDA shared | fresh CUDA/shared configure; shared artifact; temporary-prefix install; `test_cuda_core` |

Use local CMake/CTest for CPU. Route CUDA configure/build/test to the
GPU-capable OpenCode environment. Keep one CUDA validation session for this
PR and report its job/session evidence. No CppResist configure is run before
this PR's own gates are green.

## Delivery, routing, and manual gates

Delivery is one implementation PR: **R6.0a minimal_autograd shared-build
capability**. It is separate from CppResist R6.1 because it changes a sibling
repository's build configuration and has its own default-compatibility,
rollback, and CUDA validation boundary.

Safety L3; implementation difficulty difficult due to public CMake target and
cross-repository use. Before implementation, obtain one bounded Sol-expert
read-only CMake/install-export preflight. Then route implementation to Luna.
Keep the implementation PR draft until the red/green gates, CPU evidence,
CUDA evidence, and required preflight are complete.

Owner approval is required for this plan's exact committed SHA, before R6.0a
implementation, and separately before merge. After its merge, CppResist R6.1
may begin under its already merged shared-autograd plan. R6.2 and R7 remain
outside this phase.

Wait policy: this plan PR waits for exact-SHA owner approval, then separate
owner merge approval. The R6.0a implementation PR waits for its own owner
approval before edits and after publication before merge; it never proceeds to
CppResist R6.1 automatically. CUDA delegation is polled in intervals below 60
seconds and stops on the first declared stop condition or after three attempts.

Rollback is one revert of R6.0a, restoring an unconditional static target;
there is no data migration or source/API rollback.

## Grilled-Me review

Assumptions confirmed: the current `autograd` target is unconditionally
`STATIC`, already has `POSITION_INDEPENDENT_CODE ON`, uses one target alias,
and has existing CPU/CUDA core tests. Risks identified: using
`BUILD_SHARED_LIBS` would alter embedding projects; adding two autograd targets
would introduce link/ABI ambiguity; and a shared artifact's loader resolution
is not tested by a static-only unit test. Simplification applied: one local,
default-off option selects the type of the existing single target, leaving all
target body and export rules untouched. Surviving concern: install/RPATH
policy is intentionally not defined here; the phase validates the in-tree
build only and must not expand into package policy work.
