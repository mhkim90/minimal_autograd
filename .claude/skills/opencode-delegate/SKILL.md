---
name: opencode-delegate
description: Delegate approved work through OpenCode MCP. Use Luna by default, justified Sol/Astra implementers for harder work, and Astra expert for bounded read-only analysis.
---

# OpenCode Delegate

Use `mcp__opencode__opencode_run_async` by default; use blocking calls only
for known-short work. Delegate execution, iteration, review, and long test
runs, not ambiguous design decisions or single-step commands.

## Boundary

OpenCode MCP uses the caller's working directory and accepts no selected `cwd`.
This path boundary denies external paths, including arbitrary `/tmp` worktrees.
Path-contained source or worktrees do not guarantee access to linked/common
Git metadata. Shell tools with explicit cwd do not widen this boundary. This
is constraint text, not authority to create worktrees, edit, publish, sync, or
cross repositories.

Another repository requires an explicitly authorized isolated checkout under
the caller's cwd, scoped to that repository: either a linked worktree whose
actual-delegate probe succeeds or a separately owner-approved independent
clone under the rules below. These checkout constraints grant no authority
to create a checkout, clone, edit, or publish; each action remains subject to
its applicable scoped approval.

Before implementation, the actual selected delegate must discover execution,
read/hash, Git, and editor capabilities actually exposed to it. Do not assume
fixed tool names from the controller or other sessions; a listing alone is not
editor proof. A read-only expert needs capabilities for its authorized reads,
not an editor gate.

A missing native editor such as `apply_patch` does not establish editing is
unavailable; an exposed confined shell may invoke an existing permitted editor.
Discover it rather than assume `patch` or install tools.

Use actual exposed execution/read capabilities read-only to establish the exact
checkout and branch, run `git status` with optional writes disabled
(`GIT_OPTIONAL_LOCKS=0` for Git checks), resolve HEAD, and compute SHA256 of the
exact approved governing plan. Resolve absolute Git-dir/common-dir metadata
locations and demonstrate the needed Git operations within authorization;
do not dump metadata contents or evade controls. Record exact tools/invocations,
exit statuses, and narrow stdout/stderr; compare checkout identity, status
against the approved dirty state, and expected HEAD/plan hash. Controller-only
shell results, supplied source visibility, and an in-caller path are not proof.
A changed checkout or runtime permission context requires fresh actual-delegate
proof; an unchanged approved envelope requires no new human prompt.

Before substantive or further changes, prove the actual editor by either
creating, reading back, and removing an authorized task-owned marker confirmed
absent before creation, or making the first minimal approved in-scope edit with
immediate readback. Never overwrite a pre-existing marker or invent extra
write/probe authority; no additional owner prompt is needed when the first edit
is already approved. Presence/read access alone is not editability. An exposed
shell may invoke an existing patch command only after discovering it; prove it
through the same authorized edit/readback, not its listing alone. Record exact
editor invocations, exits, readback, and marker cleanup.

A model-reported missing tool is cause-unconfirmed, distinct from a natively
observed invocation failure. Neither uncertainty nor wrapper completion grants
permission to continue past a required absence or denial.

### Failure classification and recovery

Classify the failing operation, not merely a failed job or command:

- **Routine execution mistake:** an incidental approved-path typo, malformed
  in-scope patch, or ordinary fixture/test/API error is automatically repairable
  within unchanged scope, acceptance, and remaining attempt cap. A typo in an
  intended approved path is not identity/role/destination drift; never search
  outside approved paths to repair a guess or count a broken fixture as a
  right-reason red gate.
- **Essential capability absent/interrupted:** suspend affected execution and
  edits immediately as `MCP_RUNTIME_TOOL_UNAVAILABLE`, including midphase.
  A model report alone is cause-unconfirmed, not native proof of absence.
  Do not continue on uncertainty, invent tools, repeat identical probes, install,
  switch roles, or detour automatically to an expert. Resume only after
  materially changed approved runtime/capability evidence proves the gate.
  When the initial envelope expressly permits it, the controller may launch
  **one fresh same-named-role recovery per phase**, only after confirmed prior
  terminal state, result retrieval, and scoped diff review. Retain job lineage,
  partial edits, provenance, elapsed budgets, and attempt history. The delegate
  does not launch its own recovery. Failed/unavailable recovery ends affected
  execution and returns narrow evidence to the controller; no cap reset.
- **Authority/security stop:** wrong authoritative checkout, branch, HEAD,
  plan hash, unexpected dirty work, scope/acceptance/model/role/destination drift,
  or host/access denial stops affected work. Denied command/edit/path access is
  distinct from tool absence. Observed external Git-metadata rejection is
  `GIT_METADATA_PERMISSION_FAILURE`; only that failure supports proposing the
  separately owner-approved independent clone below. No fresh-session denial
  evasion, permission bypass, unapproved clone, or blind retry is allowed.

An essential missing file is a narrow blocker; distinguish an incidental
approved-path typo from genuinely unavailable evidence. No failed execution
implicitly changes authority. Read
[`references/wait-policy.md`](references/wait-policy.md) before runtime recovery
for terminal-resource accounting; a cancellation request is not terminal proof.

### Native evidence and diagnostics

Capability failures return terminal evidence: reason, job/session, role,
exposed tools, failing operation/invocation, exits/error, readback if any, and
attempt counts. Include native normalized per-job usage when available,
otherwise unavailable with coverage limitations. This is observational only:
no fabricated totals, budgets, `usage_mcp`, private-provider-path reads, or
usage-availability gate. Skills suspend or stop affected execution; they cannot restore MCP tools
or runtime.

For reported absence, distinguish model report, native invocation/failure,
preprompt connection/permission observations, and accepted-turn coverage.
Unsupported schema visibility, native exits, and loaded-code provenance remain
unknown; disk HEAD/hash is not loaded-code attestation. Wrapper done/exit 0,
printed JSON, and assistant prose are not native execution proof. Never treat
uncertainty as proof of absence, denial, execution, or repair.
Normalized native diagnostics contain only bounded redacted identities,
name/state, optional boolean executed, and error presence. The normalized
`executed` flag reflects `providerExecuted`, not local-process success; use bridge
exit provenance and tool state for local execution evidence. Never expose raw
private provider tool input/output, commands, prompts, reasoning, provider
state, private errors/logs, or credentials. This restriction does not remove
already-authorized exact manual capability-proof invocation/exit/readback
reporting; redact credentials. Known native absence/denial remains a suspension
or hard stop respectively; unconfirmed cause must be labelled as such.

Only that observed external Git-metadata permission failure supports proposing
a separately owner-approved independent clone beneath the caller's cwd. Name
the repository, ref, exact revision, destination path, task branch, original
plan/P0/dirty-state provenance, and exact scope/permitted actions. Never create
or substitute the clone without that scoped approval. It must not depend on
blocked shared/common object metadata; repeat the full actual-delegate gate there
before implementation. A clone expands no scope, publication authority, or
permissions. Preserve the existing checkout, dirty work, provenance, and
approvals; keep the original governing plan/P0 immutable. Never forcibly copy
or relocate Git metadata, redirect git-dir/work-tree or symlinks to evade
restrictions, widen permissions, or auto-escalate.

An explicit request for a configured delegate authorizes necessary in-scope
source/diff disclosure to that delegate; do not ask a duplicate disclosure
question. Still request direction for a different repository, out-of-scope
private material, unconfigured external service, publication, or destructive
action.

## Routes and evidence

- Mechanical/economy and standard work: `agent="luna"`, omitting `model` and
  `variant`.
- Justified strong or hard implementation: `agent="sol-implementer"` or
  `agent="astra-implementer"`, omitting `model` and `variant`. These are
  bounded Luna-style editors, never reviewers or publishers.
- Explicit configured-default route: only when the user asks for it, omit
  `agent`, `model`, and `variant`; report it as an explicit user route, never
  as the default.
- Difficult preflight or blocker analysis: fresh bounded read-only
  `agent="astra-expert"`. It never implements, publishes, delegates, waives
  direction, resets retries, upgrades later phases, or reviews its own advice.
- Explicit approved whole-phase route: `agent="astra-orchestrator"` only
  beneath an external Codex or Claude controller. The controller retains final
  gates, independent review, and publication. This is procedural, not caller
  authentication.
- All five named profiles bind their exact GPT-6 `#high` model. Omit model
  and variant; a conflicting override stops. Raw unnamed models are an
  explicit user route only; there is no automatic effort escalation.

Named profiles bind their exact model and high variant; omit overrides. L3
requires fresh preflight and a distinct independent post-implementation
reviewer through the external controller. If unavailable, stop; do not reuse
the same Astra expert. Tool unavailability, missing requirements,
infrastructure failure, or slowness alone does not justify a role change.

Record requested agent, job/session, bound/reported model, and route evidence.
An honored selector plus terminal requested-role response is minimum evidence;
missing resolved-model/usage is an observability warning only then. Selector,
model, role, session-lineage, or fallback contradiction stops the phase.
Classify execution failures under [Boundary](#boundary); ordinary repair is
not selector drift. Read [`references/prompt-template.md`](references/prompt-template.md)
immediately before sending a prompt.

### Candidate accounting

A candidate is a materially changed in-scope artifact submitted to its relevant
validation; partial substantive edits still count when validation fails, including
guard or fixture failures. Reads, probes, polls, and a malformed patch that
changes no artifact are not candidates; record their errors and elapsed/resource
use. When separately testable preparation and implementation exits matter,
declare scopes, acceptance, and candidate allocation before execution as shared
or stage-specific; simple tasks need no added stage. An explicitly required
preparation RED completes that milestone and consumes a candidate, but is neither
a failed candidate nor another same-blocker failure, and is never implementation
GREEN. Preserve consumed history across stages and review fixes, but do not charge
a separately approved successor for its prerequisite's spent allocation; only an
already-bound shared limit carries that count forward. A successor proceeds only
within its own approved allocation and every applicable shared limit. No renamed
stage, session, file, or role retroactively splits or resets an exhausted shared
limit; a new allowance needs a separately scoped amendment.

## Async-first bounded lifecycle

Keep one bounded phase/subphase per same-role session and end it after its
approved acceptance milestone (GREEN, or explicitly required preparation RED).
Preparation RED is never implementation GREEN.
A materially changed scope/red gate/strategy/blocker may justify a planned
continuation or fork only within existing authority. Runtime recovery instead
uses the explicit [failure classification](#failure-classification-and-recovery),
never a cap reset or a duplicate live job.

Default to at most three substantive candidate edit-and-validation cycles,
not three commands, reads, polls, or capability-discovery calls. Record failed
candidates/validation honestly. After two failures on one blocker, return
specific evidence to the controller for bounded Astra-expert consultation;
attempt a third only after a materially revised approach. Expert help grants
no new implementation authority. Essential absence suspends edits; authority
and denial stops override ordinary retries.

Astra-expert is read-only: one initial consultation plus at most one follow-up,
using a new session and refreshed capsule for the follow-up. Inspection is
scoped and decision-driven under declared wait/stop limits, not a universal
batch ceiling. Each batch resolves a named question with a relevant range or
narrow symbol; no whole-repository/secret enumeration or whole-file reads when
a range suffices. Stop with missing evidence when unavailable or when further
inspection no longer addresses the question. Return findings, approach, gate,
and stop/go; never edit, publish, delegate, or invoke skills. Read the prompt
template immediately before its capsule.

## Async stop kernel

Keep same-role session and job lineage. Continue polling and retrieve terminal
output; silence alone is not failure. Required live/unknown resources block
gates and success. A final blocked/pending report is allowed at an explicit
owner stop or declared hard expiry, with each pending resource accounted for;
it never claims an empty ledger or completion. Cancellation/replacement follow
controller authority and confirmed terminal state, not a timeout alone.
Declare task-specific checkpoints, maximum wait, and final-synthesis allowance
within any approved overall wall-clock deadline; no universal ten-minute grace
or unexplained short sub-budgets. Read
[`references/wait-policy.md`](references/wait-policy.md) immediately before
polling, recovery, cancellation, or finalization; it owns ledger and elapsed-time
procedure, never route or publication authority.
