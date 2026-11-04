# Phase 11d.2 — legacy retirement and API freeze proposal

2026-11-04. Planning only. One provider plan-plus-implementation PR, with
automatic internal packets; no inventory-only, examples-only or test-only PR.
Approval of the recommendations below is required before implementation.

## Goal, prerequisites and authority

Remove the obsolete Eigen/Var graph and duplicated implementation while keeping
the already migrated Tensor/Variable API and CppResist behavior intact. Freeze
the supported source/package surface, not every C++ ABI or deployment setup.

Verified prerequisites:

- minimal_autograd PR117 is merged at
  `c7216e412a1b0716faebe043bcd1e160d73349c2`, tree
  `7669d4119766940a1e028c281a64c914e869336e`; current remote main matches.
  Literal normal-target `AUTOGRAD_CUDA_ENABLED=0/1` is available.
- CppResist PR173 is merged at
  `bc23730f9cc32cd2664dc826f6277ce99afe119b`, tree
  `ead489d700318d3ac6d30982208a3dbce0443fc1`. CUDA configuration no longer
  requests provider expert or reads its legacy macro. CPU/CPU32, CPU/CUDA32
  and visible RTX5060 CUDA/CUDA41 passed for that prior delivery.
- These are prerequisites, NOT qualification of this deletion candidate.
  Exact-base baseline and post-deletion qualification remain mandatory.

This proposal details the 11d.2 successor named in
`PHASE11D_CUDA_CAPABILITY_PREPARATION_PLAN.md`. It preserves the historical
plans. On approval, its explicit routing, packets and allocations replace the
old Phase11 deletion proposal's Luna/attempt-cap/wait suggestions only for
11d.2. All source-breaking, downstream, review and publication gates remain.

## Recommended owner design decisions

Approval means explicitly accepting all of the following recommendations.
None follows automatically from the earlier additive metadata approval.

1. **Legacy source break:** remove Var/VarPtr/Function, Mat/Mats and their
   convenience overloads, public legacy graph/mirror fields and duplicated
   optimizer state. No deprecated adapter or compatibility graph remains.
   Unknown external legacy consumers must migrate and rebuild.
2. **Keep replacement names:** retain `ag::nn`, `ag::optim`, `ag::diffusion`
   and existing `autograd/core/*.h` include paths and symbol ownership.
   Add the flat aliases promised by the architecture plan, without renaming
   classes or introducing a second implementation/state representation:
   nn NamedParameter/Module/Linear/ReLU/Sequential/Conv2d/MaxPool2d/
   DepthwiseConv2d/AvgPool2d/NearestUpsample2d/GroupNorm, optim SGD/Adam/AdamState,
   and diffusion randn/randn_like/sinusoidal_time_embedding/q_sample.
   Preserve function signatures/default arguments. Do not recreate old
   ReLUModule/SiLUModule/SigmoidModule wrappers
   or old constructor/forward signatures. Alias identity is compile-tested.
3. **Umbrella:** add `autograd/autograd.h` containing the canonical normal
   headers, including custom-op construction and complex/FFT. Root
   `autograd.h` contains only pragma-once and that forwarding include.
   Neither umbrella includes Eigen/CUDA extensions. Direct current core
   includes remain supported; no simultaneous header relocation project.
4. **Expert boundary:** keep the existing explicit `expert` component and
   `autograd::autograd_expert` target spelling and opt-in dependency graph.
   Preserve Eigen copying helpers, changing signatures/local types to explicit
   `Eigen::MatrixXf`; remove only legacy aliases/shape/numel helpers there.
   Preserve borrowed CUDA views and their lifetime/synchronization rules.
   Remove expert's PUBLIC `AUTOGRAD_USE_CUDA`; use normal-target metadata.
   Private backend configuration may remain inside the library; backend files
   and runtime device dispatch are not being redesigned.
5. **First versioned surface:** propose API/package `1.0.0`, shared library
   VERSION `1.0.0`, SOVERSION `1`, with a generated installed CMake package
   version file using `SameMajorVersion`. Keep unversioned find_package calls
   working. Test version acceptance/rejection; this is a version-selection
   rule, not a cross-compiler, static/shared, CPU/CUDA or C++-stdlib ABI promise.
   Accept a source-compatible1.x evolution policy; later breaking source
   changes need a new major. Future same-major binary compatibility is not
   established by this initial freeze or by package-version matching.
   Old unversioned binaries must rebuild; do not claim legacy ABI compatibility.
   This delivery does not publish a release/tag or invent a public runtime
   backend/version accessor.

Normal shared consumers remain Eigen/CUDA-header and backend-macro free.
CUDA static consumers retain only the technically required link-only cudart
dependency; existing OpenMP propagation remains. CUDA-enabled libraries still
execute CPU operations on CPU Tensors. Compiled capability is not GPU presence.

## Checkout and artifact binding after approval only

Use the current root checkout, not a new clone/worktree. Preserve original
branch `codex/phase11d-cuda-capability` and head
`bfa45e438f1f90da86928644520e98fe23aca384` (tree equals merged117).
Recheck connected main, fetch exact expected merged117, and create
`codex/phase11d2-legacy-retirement` at c7216e4 in this SAME checkout. Remote
drift stops setup; no silent rebase/base refresh, reset or cleanup.

Preserve owner dirty `.opencode/agents/sol-implementer.md`, untracked `.agents/`,
`.opencode/agents/terra-high-implementer.md`, `.worktrees/`, `HANDOFF.md` and
existing docs reports. Do not inspect private settings, edit profiles/skills,
stage unrelated files or require owner cleanup. Confirm task paths are clean.
Keep the retained old provider checkout and every previous evidence directory
unchanged. An actual access denial stops affected operations, not a workaround.

Create fresh ignored `build-phase11d2-retirement` (absent at planning) for the
frozen approved proposal, baseline snapshots, builds/installs and evidence.
Baseline source may reuse READ-ONLY the already verified c7216e4 tracked-source
archive under the contained consumer's build-phase11d1-cuda-capability directory;
reverify archive identity/bytes/modes first. It has no .git; do not inherit its
parent checkout HEAD as provider identity. No old build is reconfigured.

Downstream qualification uses unchanged source in
`.scratch/cppresist-phase11c4a-self-contained`, local head60d1a11 with tree equal
to merged173. Recheck that binding/clean state. Allow only a fresh ignored
`build-phase11c4-installed-provider/matrix-phase11d2-retirement` child under
the existing phase11c4-owned parent. Existing driver may build the candidate
provider root into its fresh prefixes, with source/dirty-state provenance
recorded. No CppResist tracked edit, branch switch, new pin, commit or sync.

## File envelope and deletion boundary

All candidate source/docs regular100644; no symlinks, executable additions or
renames. Explicit staging after controller review, never a whole-directory add.

Delete exactly these legacy production paths after their final users move:

- `include/autograd/{variable,function,ops,module,loss,optim,conv,norm,diffusion,complex,fft,cuda_core}.h`;
- `src/{variable,ops,module,loss,optim,conv,norm,fft}.cpp`;
- `src/{cuda_core,cuda_fft}.cu`.

Retain `tensor.h`, `shape.h`, `device.h`, every replacement `core/` header and
implementation, split CPU/CUDA dispatch and Tensor kernels/storage, private
detail headers/constants and canonical `core/fft_norm.h`. No algorithm,
reduction tree, FFT normalization, precision, layout, dispatch, state format,
transfer policy or checkpoint change. Names resembling old kernels alone are
not proof that a function is disposable; verify references/linkage.

Permitted edits/additions, only for this retirement:

- This plan; root `CMakeLists.txt`; `cmake/autogradConfig.cmake.in` comments
  and version-file install wiring; README migration/build/example sections;
  `tests/consumer/README.md`; append final coverage/status mapping to
  `docs/PHASE0_BEHAVIOR_CONTRACT.md`, without rewriting its historical rows.
- `include/autograd.h`; new `include/autograd/autograd.h`;
  `core/{variable,module,optim,diffusion,complex}.h` only aliases/transitional
  comments; `extension/eigen.h` only removal of aliases and explicit copy types.
- Existing `test/*.cpp` and `test/grad_check.h` only necessary legacy migration,
  replacement coverage and private selector changes. Add
  `test/test_api_freeze.cpp` for registered umbrella/alias/boundary assertions.
- Existing three `examples/*.cpp` and `examples/mnist_data.h` only migration.
- Existing normal/embedded/Eigen/CUDA consumer CMake/programs for new umbrella,
  macro-absence and version assertions. Delete both legacy_expert fixture files
  and its empty directory once its useful forward/backward proof is replaced.
  Add `tests/consumer/api_freeze/{CMakeLists.txt,main.cpp}` for installed header,
  removed-surface and version-selection controls.

No core numerical .cpp/.cu edit, other deletion, source relocation, workflow,
instruction/profile/skill, dependency addition, generated fixture update,
provider-target split or RPATH redesign. A required replacement feature/API
gap returns to the owner before editing; do not conceal it inside cleanup.

## Internal packets — one PR, automatic local continuation

### P0 — preserve baseline and executable obligations

Before code edits, record base, approved plan/dirty state, complete candidate
source/mode snapshots, registered native inventories and actual baseline
CPUdefault14/CPUadvanced17/CUDAadvanced20 runs. Capture source/header/exported
symbol evidence proving legacy still exists, and metadata/target graphs.
Fresh installed downstream baseline against exact provider base must retain
original ordered32/32/41 and R1 branches; prior results are only cross-checks.

Create a function/contract-level migration table from every legacy test and
Phase0 guaranteed row to a replacement executable assertion. Suggested groups:

| Existing suite | Preserve on replacement API |
| --- | --- |
| core + grad_check | shared graph, finite differences, broadcast/shape, rollback |
| nn | Linear/Sequential, XOR/loss, SGD/Adam trajectories, MNIST loader values |
| conv + extensions | naive spatial references, gradients, validation, normalization |
| fft | complex branches, fixtures, non-square transforms, spectral gradients |
| characterization | parameter order, zero_grad/state isolation, no data mutation |
| diffusion + smoke | seeded noise, embeddings/q_sample gradients, composed inference |
| cuda_core + cuda_fft | CPU/device forward/backward, optimizers, FFT and rejections |

This grouping is not permission to replace a suite with a trivial smoke.
Port the assertions or cite exact equivalent replacement cases. Retain the
existing14/17/20 test names/order; the new API-freeze test adds one per build.
Do not reduce the inventory merely because an OOP sibling overlaps it.

Only facade-specific/unpinned semantics may retire: public graph fields,
legacy intermediate-gradient amplification/Conv3x repeated backward, legacy
alias/coexistence and mirror state. Preserve fresh-seed accumulation, rollback,
supported validation, optimizer isolation and all guaranteed numerical rows.
Where replacement semantics already intentionally differ (e.g. node-local
Variable::zero_grad, rank/layout or newly supported operations), cite its
existing contract/tests and test that contract; never resurrect the old bug.
A remaining guaranteed behavior with no replacement contract is a STOP.

Right-reason RED: current installed normal/root umbrella leaks legacy/Eigen,
Mat/VarPtr compile and removed legacy symbols/paths are present. Positive
control must prove compiler/package validity; a missing toolkit is not RED.

### P1 — migrate consumers while the old implementation still exists

Port legacy test/helper/example call sites using owned Tensor, Variable,
registered parameters, public ops and one optimizer state. Finite differences
perturb/restore Tensor values and build fresh graphs without public graph data.
Keep assertions effective in Release; no tolerance/reference weakening.

Introduce PRIVATE numeric `AUTOGRAD_TEST_CUDA_ENABLED=0/1` only on relevant
in-tree test targets. Replace existing macro-controlled branches in
test_tensor, test_oop_conv and test_oop_fft; verify CUDA branches actually run
and CPU-only rejection checks do not erroneously execute in CUDA builds.
Other normal test targets link normal; explicit extension tests opt in. Tests
that use CUDA runtime APIs own their SDK dependency. No public selector leak.
Replace test_tensor's legacy alias checks with explicit Eigen-copy and
removed-surface checks. In test_oop_diffusion replace legacy coexistence with
replacement coverage, adding alias checks only after P2 removes collisions.
The misleading legacy-named test_cuda_tensor case actually checks Device
descriptors: preserve those assertions, rename/comment it accurately rather
than inventing or deleting Eigen coverage. Preserve all storage/device cases.

Keep linear-regression and MNIST CLI/network/optimizer choices. Loader data
becomes Tensor/host values instead of ag::Mat; retain IDX normalization,
row-major image order, one-hot labels and malformed-file checks. GPU examples
may use example-local device-constructed registered modules/public functional
ops; no new library Module::to or mutable parameter rebinding API. Keep CPU
Adam and GPU SGD, existing batch/evaluation boundaries and NCHW reshape.

Run focused migrated suites and examples before removal. All in-tree active
call sites, including header/coexistence checks and consumer fixtures, must
be accounted for. Review the coverage table before proceeding locally.

### P2 — remove, expose final surface and version package

Delete only the named unused legacy production paths and legacy fixture.
Remove their source-list entries, public expert macro and stale documentation.
Keep one provider target and existing normal/expert dependency semantics.
Add canonical umbrella, root forwarder, identity aliases and version wiring.
Default STATIC selection and explicit BUILD_SHARED_LIBS behavior remain as-is;
the existing unused AUTOGRAD_BUILD_SHARED option is not a cleanup opportunity.

Register the API-freeze test. Installed controls compile each normal header,
canonical/root umbrella and aliases without Eigen/CUDA; old names/removed
headers must fail intentionally, paired with a valid positive compile.
Check both normal and expert interfaces have no public backend/test macro.
Version controls accept a1.0 request, exact1.0.0 and unversioned,
reject unsatisfied1.1/1.0.1 requests and other majors, and inspect shared SONAME.
Test actual supported minimum CMake behavior or report an explicit limitation;
do not silently raise the minimum. No compatibility claim for version ranges
unsupported by the existing CMake minimum.

### P3 — full qualification, tight review and local delivery

1. Fresh CPU/CUDA x STATIC/SHARED Release builds/installations; original
   default/advanced suites plus the new freeze test (15/18/21 total), all
   enabled examples.
   Preserve baseline names/order and all mapped obligations; raw native exits,
   JUnit and actual per-branch output, zero required skips/failures/errors.
2. Installed and embedded normal consumers in all four variants, no inherited
   tests/examples. Installed normal consumption disables Eigen discovery;
   CUDA shared and CPU installed forms disable toolkit discovery, while CUDA
   static retains required cudart discovery. Embedded provider COMPILATION
   still needs Eigen/toolkit; inspect consumer usage requirements separately,
   not an impossible dependency-free source build. Expand the embedded
   forbidden-target check to the full final test/example inventory. Explicit
   Eigen copied-value/gradient and actual CUDA custom borrowed-view work pass.
   Metadata remains literal0/1. Test malformed/wrong-capability/version cases
   for intended reasons, not unrelated discovery/compiler failures.
3. Run linear-regression and bounded CPU/GPU MNIST training/evaluation using
   existing local data, no downloads. Check finite loss/updates and completion,
   preserve exact parser/row-major fixture checks. No accuracy/speedup claim.
4. Run unchanged CppResist173 full installed-provider driver against candidate:
   original orderedCPU/CPU32, CPU/CUDA32 with product toolkit disabled and
   visible-GPUCUDA/CUDA41. Require R1 rejected/available branches, actual
   graph/backward/Adam/exact-resume, all metadata/locator/static/fixture controls,
   sentinels, selected/relocated loader paths and completed benchmark smoke.
   Preserve all216 fixtures/generator and13 architecture-baseline rows.
5. Scan only active/install/source/link surfaces for legacy, not historical
   plans, build artifacts or fixture prose. Compare demangled exported symbols
   and dependencies: no Var/Function/old module/optimizer symbols, one provider,
   no duplicated state or owning CUDA normal API. Borrowed expert views and
   legitimate detail kernel names are not forbidden leftovers. Fresh prefixes
   must contain no deleted header; do not install over stale prefixes.
6. Distinct read-only Claude L4 finish review covers COMPLETE diff, coverage
   table, aliases/package graph, source/symbol closure, raw test/negative/GPU
   evidence and data preservation. Controller independently verifies native
   results and obligations, resolves blockers, explicitly stages allowed paths
   and commits locally. Review advice is not execution qualification.

Visible GPU is mandatory for the claimed device gates. Use absolute
`/usr/local/cuda/bin/nvcc`, nested PATH/CUDACXX, parallel4 and owned TMPDIR.
Preserve safe argv/exits/streams/loader evidence and all old artifacts; no broad
environment dump or secret read. Retain the known successful-nested-Phase11b
stderr capture limitation; do not add a logging framework or claim complete
nested telemetry. Verify source/owner dirty state and data before and after.

## Routing, limits, stops and publication

Risk L4; hard cross-cutting deletion/coverage migration warrants configured
`sol-implementer` rather than the default mechanical route. Omit model/variant
overrides. Implementer edits/tests only; controller stages/commits/publishes.
Proposed shared implementation/qualification/fix/synthesis allowance10800s;
distinct Claude finish-review/follow-up allowance2400s. Checkpoint about2min,
individual waits<=60s. Prior phase budgets stay closed, never borrowed/reset.
No default attempt/token counter and no usage_mcp accounting.

Same-role recovery follows current skills only after terminal result retrieval
and scoped state checks, within unchanged authority/remaining limits. Missing
apply_patch alone does not prove no permitted editor. No blind replay, model
substitution, installs, permission widening or access-denial workaround.
Stop for source/base drift, replacement-contract gap, unresolved behavior or
ABI/version choice, coverage/numerical/dependency regression, required GPU or
review unavailable, exhausted allowance or scope expansion.

Approval authorizes ONLY the explicit recommendations/setup above, named
local edits/deletions, fresh qualification artifacts, read-only downstream
qualification, reviews, explicit staging and controller-local commit. It does
not authorize push/PR/Ready/merge, cross-repository tracked changes, release/tag,
branch deletion, old-evidence cleanup or additional worktrees. One eventual
plan-plus-implementation PR; named Ready PR merge remains a later owner gate.
Rollback preserves original branches/base and restores via a separately
authorized revert of the one published PR; never destructive reset/cleanup.

## Grilled-Me review

Assumptions confirmed: merged173 no longer consumes provider expert metadata,
but upstream tests/examples and expert alias fixture still use legacy. Existing
core paths and nn/optim names are live downstream contracts; deleting or
renaming them would broaden the break. CUDA test branches depend on the old
public macro today, including Conv parity as well as Tensor/FFT.

Risks identified: passing reduced suites after removal; silent CUDA branch
loss; mistaking a needed private kernel/FFT constant or Eigen copy helper for
legacy; replacing promised behavior with a smoke; accidental replacement API
rename; claiming version/SONAME implies universal ABI compatibility. Explicit
mapping, preserved inventories, private selector, precise deletion list, alias
identity and package/source/symbol/device gates address these.

Simplification applied: one PR with four internal packets, no speculative
Module transfer API, extension-target split, header relocation, numerical
optimization or logging project. Existing namespaces and fixtures stay usable.

Surviving owner choices: accept the legacy source break, additive flat aliases
with retained namespaces, new umbrella and1.0.0/SOVERSION1/package compatibility
policy, plus the explicit local setup/deletion envelope and allocations.
No product implementation has begun.

## Astra advice disposition

Read-only OpenCode astra-expert job9100690429974ee8a77cac5db0e2df9c,
session ses_e5ae29009ffenfBfspJUILaGAh, reached done/exit0; full result retrieved.
Configured profile binds Astra/high, but effective model/effort telemetry was
not exposed; native diagnostics do not attest test execution. The expert
performed source analysis only, no builds, downstream inspection or edits.
It relied on supplied merge/test facts, independently checked by controller.

GO for planning, STOP before owner-approved implementation. Applied findings:
retain existing namespaces/core paths with explicit identity aliases; preserve
case-level training/smoke/MNIST obligations and private Tensor/Conv/FFT selectors;
use example-local GPU parameters, not Module::to; distinguish embedded provider
build dependencies from installed consumer isolation; fix misleading Device
test naming without losing assertions; expand embedded inheritance checks.
The expert suggested possible redundant-suite retirement only after mapping;
this narrower proposal instead retains/ports every registered suite and helper.
Version/include/source-break recommendations remain explicit owner choices.
Consultation consumed an observed upper bound263/600s in its separate planning
allowance; no implementation or
finish-review allowance has started. Advice cannot replace distinct Claude
finish review, owner direction or native qualification.

## Execution discovery — proposed compatibility amendment, not yet approved

Owner approved the original proposal (SHA2568a4d756670a891a9a15f1ab544a1dea5e520065ab856bd1d41e6c39488ac9572),
now frozen verbatim in build-phase11d2-retirement/approved-plan-original.md.
Root taskbranch is bound to the approved c7216e4. No product edits/deletions
have begun; P0 baseline work is partial, not a full qualification claim.

Controller found a concrete conflict with the unchanged-downstream-driver
acceptance gate: CppResist173 test/test_installed_provider.cmake checks explicit
and NOTFOUND-recovery provider-info against the literal lib/libautograd.so.
Product provenance intentionally records $<TARGET_FILE:autograd::autograd>.
The approved VERSION1.0.0/SOVERSION1 design instead exports the actual file
lib/libautograd.so.1.0.0 and SONAME libautograd.so.1. A native configure of the
READ-ONLY baseline from a fresh owned wrapper confirms exactly those generated
target/export paths; no source or existing package was changed. This is a
qualification-script compatibility defect, not evidence of a product failure.

Recommended scope amendment for owner approval:

- Keep the approved provider version/source-break design, allocations and one
  provider plan-plus-implementation PR. No algorithm or dependency change.
- Permit a separate narrowly scoped CppResist compatibility delivery because
  the second repository owns the broken qualification assertion. In the SAME
  contained consumer checkout, verify/fetch exact remote masterbc23730 and
  create codex/phase11d2-versioned-provider-qualification there, preserving
  original branch/head/evidence. Remote drift stops setup.
- Only test/test_installed_provider.cmake plus
  plans/PHASE11D2_VERSIONED_PROVIDER_QUALIFICATION_PLAN.md may change there.
  Compare actual and expected canonical shared-library paths for explicit and
  NOTFOUND selection, retaining exact config/provider identity. Do not merely
  accept any libautograd filename or weaken wrong-provider detection. Support
  both the old unversioned and new versioned real installs. No product CMake,
  C++/header, numerical case, fixture/generator or other consumer edit.
- Qualify the correction with real unversioned baseline and versioned provider
  installs, intended wrong-provider rejection and the full existing32/32/41
  matrix/controls. Fresh artifacts only. The actual candidate matrix then uses
  this narrowly corrected driver; it cannot be called the unchanged173 driver.
  Original173 source observations and all baseline results stay recorded.
- Continue approved local provider migration and qualification after this
  bounded correction; require distinct Claude review covering both complete
  diffs and controller verification. Shared10800s execution/2400s review limits
  cover the amendment; no new/reset budget or attempt cap. Separate local
  commits per repository, not separate inventory/example/coverage micro-PRs.
- Push/PR/readiness/merge and cross-repository release coordination remain
  manual owner boundaries. This proposal authorizes none. Consumer correction
  must be merged before final provider deletion publication is treated as
  qualification against a merged downstream contract.

Until this material scope amendment is approved, no P1 production editing or
consumer tracked change is authorized. The original plan remains immutable;
this appended discovery does not silently replace its acceptance gate.

[Grilled-Me Review — amendment]
Assumptions confirmed: native generated export/TARGET_FILE proves the filename
change; both exact literal assertions are in the consumer driver, not product
discovery. Risks identified: a wildcard filename check would hide a wrong
provider; an unrecorded temporary driver overlay would misrepresent unchanged
downstream qualification. Simplification applied: two consumer files only,
canonical-path identity rather than a product workaround, no version deferral
or numerical edit. Surviving concern: this is an actual cross-repository
compatibility/qualification boundary and needs explicit scope approval before
the separate consumer delivery; no publication authority is inferred.

## P1 discovery — proposed DepthwiseConv correctness amendment, not approved

The owner approved the compatibility amendment above. P0 is complete: actual
upstream14/17/20 and corrected downstream32/32/41, scope/data/loader/coverage
checks passed. P1 partially migrated tests/helper/loader and private selectors,
but is NOT complete/buildable as a full delivery. Partial changes are retained.

The preserved test_extensions depthwise finite difference uses the original
N2,C3,H=W5,kernel3,stride1,pad1,eps1e-3/tol.05. It fails weight input with
maxerror4.01588. Existing replacement OOP gradient coverage uses N1. Independent
public-API reproducer (not the new grad_check helper), N2/C1/3x3/kernel1 with
sample values1 and2, gives exact dWeight27, finite difference26.9985, current
backward18. Controller compiled/ran the same reproducer against the untouched
P0 baseline library and confirmed the identical failure. This is pre-existing
replacement behavior, not a new test-helper or migration artifact.

Root cause: src/detail/tensor_kernels.h, function
tensor_depthwise_conv2d_nchw_backward_weight, assigns dw_data[dw_off] for each
batch, losing earlier samples. The vector is initially zero; accumulated
contributions must be added. No production header/source has been modified.
Original no-numerical-edit scope therefore blocks further implementation.

Recommended narrow owner-approved correctness subplan, in the SAME provider
plan-plus-implementation delivery rather than a new feature/inventory PR:

- Extend the file envelope ONLY to src/detail/tensor_kernels.h's depthwise
  backward-weight batch accumulation: assignment becomes accumulation (`=`
  to `+=`). No other helper, layout, precision/reduction policy, CUDA kernel,
  public API, dependency or optimizer/state change. This repairs an existing
  guaranteed derivative; it adds no replacement feature/capability.
- Add focused regression assertions to existing test/test_oop_conv.cpp using
  independent scalar/naive reference and finite differences: N1 remains
  correct, N2/N3 sum distinct sample contributions, include signed/weighted
  upstream and existing stride/pad geometry. The discriminating27-vs18 fixture
  must fail before the correction and pass afterward. Keep original N2
  test_extensions case, epsilon, tolerances and assertions unchanged.
- Run the focused regression and preserved failed extension test first, then
  complete P1's pending ports and exact14/17/20/example execution gates before
  P2/P3. CUDA depthwise remains explicitly unsupported; CUDA-built provider's
  CPU Tensor path gets the same CPU correction. No new device implementation.
- Continue complete source/coverage/fixture and CPU/CUDA static/shared,
  installed/embedded and actual versioned downstream32/32/41 qualification.
  Distinct Claude finish review must explicitly audit this production change
  and regression strength in addition to the full retirement diffs.
- All other original/amended gates and publication boundaries remain. Shared
  execution consumed1988/10800s (P0293+1073, P1622),8812s remains; P1's3600s
  internal allocation has2978s remaining, distinct Claude0/2400s. No reset or
  borrowing. Same-role Sol may implement the approved correction and resume
  existing partial work only after approval, with exact original plan/base/P0
  and stopped-job lineage retained. No push/PR/Ready/merge authority.

[Grilled-Me Review — correctness amendment]
Assumptions confirmed: independent native baseline reproduction and exact
batch-loop assignment identify a real replacement derivative bug. Risks:
weakening the test or reducing N would conceal lost batch contributions;
changing broader accumulation/precision would exceed the needed correction.
Simplification: one accumulation statement plus regression cases in an existing
suite, no new API or PR split. Surviving owner choice: explicitly permit this
production numerical correctness fix despite the original exclusion, then
resume the approved migration. Without that approval, P1/P2 remain stopped.

## Local execution and review outcome — 2026-11-04

The owner approved the original recommendations, the separate consumer
qualification amendment and the narrow depthwise correctness amendment before
their corresponding edits. Proposal/not-approved headings above are preserved
historical checkpoints, not the current approval state. The approved full
revision remains the exact prefix of this file, SHA256
`1b7cd8a7311edc790a8ca8d5555cc08e1752165d4921b99ec3d0228186419192`;
all three frozen proposal revisions and P0 evidence remain unchanged.
This appendix records outcomes only; it changes no design or authority.

Provider base is merged117 `c7216e412a1b0716faebe043bcd1e160d73349c2`,
branch `codex/phase11d2-legacy-retirement`. The contained consumer uses merged173
`bc23730f9cc32cd2664dc826f6277ce99afe119b` on
`codex/phase11d2-versioned-provider-qualification`. No base refresh, new checkout,
profile/skill/workflow change or unrelated owner-file staging was performed.

P0 through P3 and ordinary review corrections are locally complete:

- Exact24 legacy paths retired; existing Tensor/core/CPU/CUDA/private extension
  implementations retained. Canonical/forwarding umbrellas, identity aliases,
  API/package1.0.0, SONAME1 and SameMajorVersion selection are qualified.
- Eight actual Release default/advanced builds: CPU STATIC and SHARED15/18,
  CUDA STATIC and SHARED18/21, total144 tests, zero failures/errors/skips.
  Final suites include successful-callback transactional rollback and both the
  original (2,3)/(1,3) and additional (2,3)/(4,3) sub/div rejection fixtures.
- Installed/embedded normal and explicit Eigen/device consumers passed in all
  four variants; actual RTX5060 CUDA work executed. Strict installed freeze
  controls passed132 positive/256 intended negative compiles,28 version and8
  metadata controls. Eight linear and24 ordinary/valid-zero MNIST runs completed.
- Actual versioned corrected-consumer matrix passed32/32/41, original103-step
  order, R1 branches, identity/sentinel/static/fixture controls, selected and
  relocated loaders and benchmark smoke. Post-review product/public-header/
  package/consumer bytes and44 provider libraries remained identical, binding
  those results without claiming a new downstream run.
- Only numerical production change is the approved depthwise backward-weight
  `=` to `+=`; independent baseline RED18/reference27 and final GREEN27,
  N1/N2/N3 signed-upstream/stride/pad regressions and original extension FD
  remain qualified. No new device implementation or other numerical policy.
- A P1 ledger rationale and first review incorrectly described sub/div as
  broadcasting. Current source always requires identical shapes. The original
  geometry is restored in the registered test, live documentation corrected,
  and immutable ledgers retained with an additive coverage correction.
- Distinct read-only Claude review, completion and final correction review
  reported Sonnet5.5 and final GO with no blocking findings. Controller checked
  native results, source/modes, library hashes and preserved216 fixtures,
  generator/13 architecture rows, IDX data and historical Phase0 prefix.

Recorded delegated execution charge is7103/10800s; review/follow-ups1058/2400s.
Failed attempts remain honestly recorded, including prior WebSocket1001 and
one superseded configure with unavailable exit; neither qualifies as success.
All required qualifying jobs/commands are terminal and results retrieved.
Actual CMake3.14 execution and Clang diagnostic matching remain unqualified;
the host used CMake3.28 and GCC. No universal ABI/performance/accuracy claim.

Evidence and exact local commit state are retained under
`build-phase11d2-retirement/CONTROLLER_EVIDENCE.md`, P0/P1/P2/P3 handoffs,
`CLAUDE_REVIEW_{1,2,3}.md` and `contract-correction/COVERAGE_CORRECTION.md`.
Controller-local commits are authorized; push, PR creation/readiness/merge,
release/tag and upstream coordination remain separate owner boundaries.
Consumer correction is a separate, currently unmerged delivery and must merge
before provider publication is represented against a merged consumer contract.
