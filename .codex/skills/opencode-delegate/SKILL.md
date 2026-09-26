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
Another repository requires an isolated worktree under the caller's cwd; shell
tools with explicit cwd do not widen this boundary. This is constraint text,
not authority to create worktrees, edit, publish, sync, or cross repositories.
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
Allow three
implementation/fix attempts; after two same-blocker failures, obtain one
bounded Astra-expert consultation and use a materially revised approach before
a third. The external controller supplies independent L3 review; no named
variant override resets retries.

Astra-expert is read-only: one initial consultation and one follow-up maximum,
four narrow inspection batches maximum, and a new session with refreshed
capsule for follow-up. It returns findings, approach, gate, and stop/go; it
never edits, publishes, delegates, or invokes skills. Read the prompt template
immediately before its capsule.

## Async stop kernel

Keep stable workflow and same-role session lineage. Retrieve terminal output
and normalized usage before finalization; do not send a final response while a
required job is live. Cancel only for terminal error, declared maximum wait, or
explicit controller/owner stop with reason and approved replacement route. A
full Astra-expert consultation in post-inspection `step_start` receives at least a
10-minute declared final-synthesis grace. Read
[`references/wait-policy.md`](references/wait-policy.md) immediately before
polling, recovery, cancellation, or finalization; it supplies the ledger and
elapsed-time procedure, never route or publication authority.
