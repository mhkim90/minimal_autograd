---
name: opencode-delegate
description: Delegate approved work through OpenCode MCP. Use Luna by default, justified Sol/Astra implementers for harder work, and Astra expert for bounded read-only analysis.
---

# OpenCode Delegate

Use `mcp__opencode.opencode_run_async` by default; use blocking calls only for
known-short work. Delegate execution, iteration, review, and long test runs,
not ambiguous design decisions or single-step commands.

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

If any essential capability is missing whenever discovered, including midphase,
STOP immediately as `MCP_RUNTIME_TOOL_UNAVAILABLE` and name the capability.
Denied command/edit/path access also STOPs immediately with distinct failure
evidence, not `MCP_RUNTIME_TOOL_UNAVAILABLE`. An observed external Git-metadata
permission rejection instead STOPs as `GIT_METADATA_PERMISSION_FAILURE`; only
this supports proposing the separately owner-approved independent clone below.
Other capability failures are not a clone remedy.
Wrong status, HEAD, hash, or path, missing files, and other failures each STOP
with distinct failure evidence. Do not blindly retry or bypass controls.
For capability failures, also do not invent tools, repeat identical attempts,
automatically detour to an expert, substitute roles, or bypass controls. Resume
only with materially changed capability/runtime evidence or a separately
owner-approved revised route, preserving attempt history.

Capability failures return terminal evidence: reason, job/session, role,
exposed tools, failing operation/invocation, exits/error, readback if any, and
attempt counts. Include native normalized per-job usage when available,
otherwise unavailable with coverage limitations. This is observational only:
no fabricated totals, budgets, `usage_mcp`, private-provider-path reads, or
usage-availability gate. Skills detect and stop; they cannot restore MCP tools
or runtime.

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
model, role, session-lineage, or fallback contradiction and failed execution
stop the phase. Read [`references/prompt-template.md`](references/prompt-template.md)
immediately before sending a prompt.

## Bounded lifecycle

Keep one bounded phase/subphase per same-role session; fork only after a
material scope, red-gate, strategy, or blocker change, and end it after green.
The [Boundary](#boundary) immediate capability STOP overrides ordinary retries;
it never triggers an automatic expert detour. After that gate passes, allow
three ordinary implementation/fix attempts; after two same-blocker failures, obtain one
bounded Astra-expert consultation and use a materially revised approach before
a third. The external controller supplies independent L3 review; no named
variant override resets retries.

Astra-expert is read-only: one initial consultation and one follow-up maximum,
four narrow inspection batches maximum, and a new session with refreshed
capsule for follow-up. It returns findings, approach, gate, and stop/go; it
never edits, publishes, delegates, or invokes skills. Read the prompt template
immediately before its capsule.

## Async stop kernel

Keep same-role session and job lineage. Retrieve terminal output before
finalization; do not send a final response while a
required job is live. Cancel only for terminal error, declared maximum wait, or
explicit controller/owner stop with reason and approved replacement route. A
full Astra-expert consultation in post-inspection `step_start` receives at least a
10-minute declared final-synthesis grace. Read
[`references/wait-policy.md`](references/wait-policy.md) immediately before
polling, recovery, cancellation, or finalization; it supplies the ledger and
elapsed-time procedure, never route or publication authority.
