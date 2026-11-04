# Phase 11d compatibility preparation — proposal

2026-11-03. Controller-authored proposal, not implementation or publication
authority. This repairs one live compatibility dependency before the separately
approved L4 legacy deletion and API freeze. No inventory-only or plan-only PR.

## Evidence and scope decision

CppResist PR #171 is merged at
`dce2d7862c3169f3c35fdbe9e9a191425e9988ae`. Its tree exactly matches qualified
local head `a1ccb5d392c4e41bd526092d2ef2fb0178a48dd6`. New installed-provider
qualification passed CPU/CPU32, CPU/CUDA32 with product CUDAToolkit discovery
disabled, and visible RTX5060 CUDA/CUDA41, with distinct finish-review GO.
The source provider remained pinned at
`1dfe66d4177a5034b9fd46e068ecec3f2b9561be`; those results are not a new
qualification of the current upstream base or a future deletion candidate.

Current upstream local and connected-GitHub `main` are
`7a8f33db0ba8226a51c6b3af2f37aba1b77a504e`. Compared with that retained provider
pin, CMake/package files, include/src, tests, examples and README are unchanged;
the revision difference is guidance/skills/profiles. Exact revision binding and
fresh candidate qualification still apply. Old R7 is complete, not an unstarted
phase to repeat.

The remaining problem is concrete: CppResist `CMakeLists.txt:41-49` imports the
expert component only to require its exact `AUTOGRAD_USE_CUDA` compile-definition
item. Neither product target links expert. Removing the exported macro first
would nevertheless break CUDA product configuration. The provider's own CUDA
custom consumer likewise checks that macro. Private config-variable parsing or
runtime allocations during configure are not acceptable replacements.

### Proposed sequence — real compatibility/repository boundaries only

1. **11d.0, this detailed proposal:** one backward-compatible provider delivery
   adds documented normal-target CUDA build metadata, migrates the in-tree CUDA
   custom consumer to it and qualifies exports/consumers. Keep legacy API and
   expert macro unchanged so merged CppResist continues to work.
2. **11d.1, later detailed CppResist plan:** consume the new provider revision,
   replace the macro check with normal-target metadata, remove the capability-only
   expert import and run final downstream qualification. Merge before deletion.
3. **11d.2, later detailed provider L4 plan:** migrate remaining in-tree
   tests/examples, replace or explicitly justify retiring legacy-only coverage,
   delete the legacy surface/implementation and freeze API/version in ONE PR
   with internal packets. Do not make separate inventory/example/test-only PRs
   absent a real replacement-contract blocker.

Owner approval of this proposal would add these compatibility boundaries to
the sequence in `PHASE11_API_FREEZE_PLAN.md`; it would authorize ONLY 11d.0
local execution. Existing 11d deletion design/source-break approval, exact-base
downstream qualification and manual publication gates are not waived. The old
plan is not silently rewritten. Later scopes need their own detailed approval.

## 11d.0 public contract requiring approval

Expose this literal custom property on the actual normal library target:

```cmake
get_target_property(_cuda autograd::autograd AUTOGRAD_CUDA_ENABLED)
# Exactly "0" for a CPU provider; exactly "1" for a CUDA-built provider.
```

- Always present for both STATIC and SHARED builds, installed or embedded.
- Set from the provider's effective build option; export with `EXPORT_PROPERTIES`.
  Use neither an `INTERFACE_` property nor a generator expression. Local CMake
  documentation confirms this export mechanism exists since 3.12, below the
  project's minimum 3.14. No minimum-version increase is proposed.
- Means compiled CUDA capability ONLY, not GPU visibility, supported device
  architecture, successful allocation, runtime availability or CPU exclusion.
- Is CMake metadata, not a C++ definition, public backend accessor, owning/raw
  CUDA API, plugin system, ABI flag or generic feature registry.
- A component-free installed normal consumer can query it without creating the
  expert target or discovering Eigen/CUDAToolkit for a shared provider.
- A CUDA static normal consumer retains its existing link-only cudart requirement;
  no stronger dependency-free claim is made. OpenMP behavior stays unchanged.
- Do not export/parse `_AUTOGRAD_HAS_CUDA` or infer support from paths/symbols.
- Missing/malformed metadata is unsupported, not CPU by default. Normal CPU
  consumers accept valid `0` or `1`; the CUDA custom consumer requires `1`.
  Qualifiers pass an independent expected value to verify correct polarity.

Retain the current normal/expert names, component semantics, public headers,
legacy expert definition and dependencies. This is an additive capability
contract; it does not authorize legacy removal or target reorganization.

## Checkout, files and artifacts

After design approval only, create `codex/phase11d-cuda-capability` at exact
upstream base `7a8f33db0ba8226a51c6b3af2f37aba1b77a504e` in the current root
checkout `/home/mhkim90/workspace/minimal_autograd`. Recheck remote `main` first;
drift stops setup rather than refreshing the base automatically. No new clone,
worktree, rebase, reset, cleanup or sibling-repository mutation.

Preserve owner changes: tracked `.opencode/agents/sol-implementer.md` and
untracked `.agents/`, `.opencode/agents/terra-high-implementer.md`, `.worktrees/`,
`HANDOFF.md` and existing docs reports. Record their task-relevant dirty state;
do not stage, edit or demand cleanup. Other task-path drift stops editing.
The retained `.scratch/minimal-autograd-e2-provider` stays clean/read-only at its
old pin. Do not repurpose it as the implementation checkout.

Exactly seven regular100644 candidate files, no deletion/rename/symlink:

- `PHASE11D_CUDA_CAPABILITY_PREPARATION_PLAN.md` — preserve approved proposal;
  append factual local outcome only if needed, never silently change design.
- `CMakeLists.txt` — additive normal-target literal/exported property only.
- `README.md` — capability/build contract and current retirement status only.
- `tests/consumer/README.md` — precise qualification commands/metadata semantics.
- `tests/consumer/find_package/CMakeLists.txt` — normal metadata assertion and
  optional independently supplied expected-value check; retain existing guards.
- `tests/consumer/add_subdir/CMakeLists.txt` — equivalent metadata assertion;
  retain no-inherited-tests/examples and macro-leak controls.
- `tests/consumer/cuda_custom/CMakeLists.txt` — replace the capability selector
  with the public normal-target property; keep expert linkage for the actual
  opt-in extension, and retain the existing source/forward/backward work.

No provider C++ source/header/test numerical-body, workflow, version, package
config dependency logic, examples, algorithms, fixtures/generator, RPATH policy,
instruction/profile/skill or CppResist source change. No new validation framework.
If another path or a different public contract becomes necessary, report first.

Use a fresh root-owned `build-phase11d-capability` directory, absent at entry,
for builds/install prefixes/negative controls and evidence. Keep all old evidence.
For downstream compatibility qualification, allow only a fresh ignored child
`build-phase11c4-installed-provider/matrix-phase11d-capability` in the existing
contained CppResist checkout. Its source remains exact qualified `a1ccb5d`, tree
equal to merged PR171; no checkout switch or provider-pin edit there. Existing
manual installed-provider driver may build the candidate root source into its
owned prefixes, without changing that source. No system install or downloads.

## Automatic internal packets after approval

**P0 — bind and observe.** Verify branch/base, approved dirty state, candidate
plan and source boundaries; record original native inventories and existing
package usage. Right-reason RED: a baseline normal target lacks the proposed
property, while old expert CUDA metadata still satisfies merged CppResist.
Use an actual baseline property query/consumer configure, not a fake GPU error.
Preserve unmodified baseline/source observations separately from candidate runs.

**P1 — implement and qualify the contract.** Add literal/exported metadata and
the three consumer assertions. Keep programs and old dependency guards intact.
Fix ordinary in-scope errors automatically. Test both valid polarities plus
missing, malformed and deliberately wrong expected-value controls in fresh
owned scratch; verify intended diagnostics rather than unrelated compiler,
toolkit or package-discovery failures. Synthetic package negatives are only
configure guards, never substitutes for real installed-provider success.

**P2 — integrated local delivery.** Complete the matrix below, compare complete
diff and preserved files, and obtain distinct Claude read-only L3 finish review.
An advice author cannot finish-review its own advice. Controller checks actual
native results and graph evidence, stages exactly allowed paths and commits
locally. One eventual plan-plus-implementation PR; stop before publication.

## Acceptance matrix

1. Build/install real CPU/CUDA x STATIC/SHARED providers with parallel4. Assert
   exact metadata `0`/`1` on installed and source-tree normal targets, and prove
   exported properties survive installation without source/build paths.
2. Build/run normal installed consumers for all four variants with Eigen3
   package discovery disabled. Both CPU forms and CUDA SHARED also disable
   CUDAToolkit discovery. CUDA STATIC permits its existing runtime-link discovery.
   Verify no normal Eigen/CUDA headers, public backend macro or expert import.
3. Preserve explicit Eigen and legacy expert consumer behavior. The CUDA custom
   consumer must build/run real forward/backward borrowed-view work on visible
   RTX5060; a CPU-provider negative must fail for metadata capability, not tools.
4. Preserve complete CPU default and advanced registered test coverage and
   visible-GPU CUDA advanced coverage, plus all enabled example builds. Record
   names/order and native JUnit/exits; no reduced inventory or falsely skipped
   required CUDA parity. Metadata-only edits must not alter numerical bodies.
5. Run the existing CppResist manual installed-provider qualification against
   candidate source in the new named evidence child: full original ordered
   CPU/CPU32, CPU/CUDA32 with product toolkit discovery disabled, CUDA/CUDA41.
   Preserve R1 rejected/available smoke/checkpoint branch proof, static/CPU
   capability/package/fixture negatives, shared-loader/relocation checks and
   completed-work benchmark smoke. No inherited PR171 success claim.
6. Validate actual selected/imported libraries and compile interfaces: one
   provider per product, existing expert macro polarity unchanged, no new
   dependency/macro leak. Preserve all216 CppResist fixtures and generator,
   prior matrices, root unrelated work and retained provider checkout.

CUDA uses absolute `/usr/local/cuda/bin/nvcc`, PATH prepend and CUDACXX for
nested children, build-owned TMPDIR and parallel4. Record safe loader variables
and actual native exits/outputs, not broad environment dumps. CUDA deployment
still needs runtime libraries. Retain the disclosed pre-existing Phase11b
successful-nested-stderr capture limitation; do not claim complete nested
telemetry or add a logging project. No tolerance or reference-output changes.

## Routing, limits and manual boundaries

11d.0 is L3 package/API metadata work with bounded CMake implementation:
configured `terra-implementer`, no model/variant override, edit/test only.
Proposed allocation60min actual delegated implementation including qualification
and fixes; distinct Claude review20min including follow-ups. Checkpoint about
2min, poll/wait <=60s. These are new proposed task-specific allocations, not
unused time borrowed from the closed installed-provider bundle.

Stop for access denial, missing essential capabilities, role/model contradiction,
repeated blocker without a changed approach, exhausted explicit allocation,
scope/base drift, unavailable required GPU, numerical/coverage/dependency
regression or unresolved review blocker. No automatic session recovery, model
fallback, installs or permission changes. Missing model telemetry alone is not
a mismatch. Native editing capability may use an existing permitted editor;
native apply_patch absence alone proves nothing. All jobs/commands must reach
terminal state and results be retrieved before gates.

Approval would cover only this local setup/implementation/qualification/review/
explicit staging/controller commit. Push, PR creation/update/readiness, merge,
cross-repository source changes, later provider-pin updates and 11d.1/11d.2
remain separate authority. Rollback is preserving the old base and reverting
this additive delivery if later authorized; no destructive cleanup is needed.

## Grilled-Me review and advice disposition

Assumptions confirmed: merged CppResist and provider CUDA custom configuration
still inspect the public expert macro; normal shared consumption itself does
not need it. Local CMake supports exporting a literal normal-target property.
Copied Eigen extension helpers and private spectral constants remain real
replacement functionality, not disposable legacy merely because names overlap.

Risks identified: deleting the expert macro first breaks downstream configure;
mistaking metadata for runtime availability weakens R1; boolean-false `0` must
not be mistaken for missing metadata; static/shared dependency differences can
hide leaks. Removing expert propagation later can also silently deselect CUDA
tests (`test_tensor`, `test_oop_fft`), so that private-test-configuration migration
belongs in the final deletion plan, not this additive change.

Simplification applied: one additive provider contract plus executable consumer
gates; no private-package parsing, runtime feature API, inventory-only PR or
separate examples/tests PR. Cross-repository compatibility boundaries are kept;
all ordinary internal packets continue without per-packet owner prompts.

Fresh Astra expert consultation `dd1e017ddedd4d4ebb161473091cfd48`, session
`ses_e608b8a74ffeDe9OFJXpg6nI47`, terminal exit0/result retrieved, recommends this
three-boundary sequence and says STOP before unapproved deletion. It performed
source analysis only, no qualification. Controller independently corroborated
the concrete source findings. Actual model/effort telemetry was not exposed;
wrapper native-execution diagnostics alone do not attest shell execution.
The consultation consumed an observed upper bound240/600s, separate from
execution allowances. The MCP job/session identifiers above, not the advisor's
unverified prose job label, are the recorded lineage.
It is advice, not an implementation audit/owner approval or finish review.

Surviving owner decision NOW: approve the literal public property and compatible
11d.0 delivery, with the later downstream transition before macro removal.
Final deletion decisions remain later: semantic version/package/SOVERSION,
flat aliases versus retained `ag::nn`/`ag::optim` names, canonical/root umbrella,
explicit Eigen helper signatures without legacy aliases, exact file/coverage
retirement mapping, and the existing L4 source-breaking gate. None is inferred
from approval of this additive preparation.

## Execution binding update — owner-requested refresh, 2026-11-04

The owner approved 11d.0, then explicitly requested latest remote update/rebase
and skill reload before resuming. The first root branch-creation attempt failed
on read-only Git metadata; no implementation, delegate, clone or partial edit
occurred. Subsequently approved controller Git operations fetched and
fast-forwarded the same root checkout to remote main
`2a46173a176e2752f0166f91aa0b06d8e12e4c66`. Only four guidance files differ
from the original7a8f33d base; product/package/test/example sources are unchanged.
Active phase-gate/OpenCode skills were reread. The owner then reported MCP fixed.

Execution branch is now `codex/phase11d-cuda-capability` at exact2a46173 in the
original root checkout; no independent clone was created. This entry replaces
ONLY the original base/setup binding above. The original approved proposal is
preserved byte-for-byte as a prefix and in the owned qualification root, with
SHA256858947fd5c95fd3c200d3940c5bced95406461a7ede87b144265bb49de44f268.
Seven-file envelope, public contract, test gates, existing recovery restrictions,
3600s implementation/1200s review allowances and publication boundaries stay.
No execution time was consumed before this resumed first implementation job.

## Local implementation outcome — 2026-11-04

11d.0 is locally qualified on the bound branch/base. The normal target exports
literal0/1 build capability; installed/embedded checks and the CUDA custom
consumer use it. Existing legacy API, expert macro and dependencies remain.
No C++/header/numerical/fixture/version change or later deletion is included.

Fresh native qualification passes upstream CPUdefault14, CPUadvanced17,
CUDAadvanced20, all four installed and four embedded normal consumers, explicit
Eigen/legacy consumers and actual CUDA borrowed-view forward/backward. New
downstream CPU/CPU32, CPU/CUDA32 and RTX5060 CUDA/CUDA41 retain original order,
R1 branches, negative controls, loader/relocation and benchmark smoke. All216
fixtures are byte-identical. The original HEAD-derived test inventory was
captured after edits, not a pre-edit native full inventory; actual pre-edit
property RED and six source snapshots are retained. No native result is
inherited from the earlier cutover or claimed as reviewer execution.

Controller verification and distinct Claude finish review returned GO with no
blockers. Evidence remains in the owned qualification directories. This local
delivery stops before push/PR/readiness/merge; 11d.1 and 11d.2 remain separately
planned and approved successors.
