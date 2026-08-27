---
name: opencode-delegate
description: Delegate approved, tedious, or long-running work through async-first OpenCode MCP. Route mechanical/economy and standard work to Luna, and justified difficult preflight to Sol-expert.
---

# OpenCode Delegate

Use `mcp__opencode__opencode_run_async` by default; use blocking calls only for
known-short work. Delegate execution, iteration, review, and long test runs,
not ambiguous design decisions or single-step commands.

## Caller-working-directory boundary

OpenCode MCP execution uses the caller's working directory; its tools do not
accept a caller-selected `cwd` parameter. This boundary is path-based, not
repository-identity-based: external paths, including arbitrary `/tmp`
worktrees, are denied. To delegate work in another repository, the controller
must first create an isolated worktree for that repository under the caller's
cwd, then give the delegate paths scoped to that worktree. Commits and PRs
still belong to that repository's remote. Codex shell tools with an explicit
`cwd` are a separate capability and do not widen OpenCode MCP's delegation
boundary. This description states constraints only; it authorizes no
cross-repository work, worktree operation, editing, publishing, or syncing.

## Routes

- **Mechanical/economy**: use `agent="luna"`; omit `model` and `variant`.
- **Standard**: use `agent="luna"`; omit `model` and `variant`.
- **Explicit configured-default request**: only when the user requests this
  route, omit `agent`, `model`, and `variant`, and report it as an explicit
  user route rather than a default.
- **Difficult**: use `agent="sol-expert"` only for a bounded preflight or
  breakthrough when warranted, then use `agent="luna"` for implementation.
- Use `agent="terra"` only for a separate, fresh-context, read-only review
  with a recorded reason; never use Terra as implementer.
- Use `agent="sol"` only for explicit whole-phase triad compatibility. Do not
  call Luna or Terra separately in that route.
- Use a raw model only for an explicit user override or approved fallback.

## Route evidence

- Record requested agent, job ID, session ID, and bound/reported model.
- Accept an honored named selector plus a terminal role response as minimum
  route evidence. Query normalized usage after terminal state when available.
- Report unresolved normalized model identity or partial accounting as an
  observability warning when no evidence contradicts the selected role.
- Stop on selector rejection, mismatched continuations, explicit model
  contradiction, or silent fallback. Never substitute silently.

## Luna and Sol-expert lifecycle

- Keep one bounded phase or subphase per Luna session and end it after green.
- Start or fork a new same-role session after a material scope, red-gate,
  strategy, or blocker change. Repeat the same named agent on continuations.
- Allow at most three implementation/fix attempts. After two failures on the
  same blocker, request one bounded Sol-expert consultation. Permit a third
  attempt only after a materially revised approach is agreed.
- Give Sol-expert approved scope, one focused question or blocker, and selected
  compact diff/test evidence. Allow one consultation and at most one follow-up.
  Require: findings, proposed approach, acceptance gate, stop/go. Sol-expert
  never implements, edits, publishes, delegates, or invokes repository skills.
  Answer from the capsule when possible. If inspection is needed, each batch
  resolves one named decision using a relevant file range or narrow symbol;
  never use repository-wide enumeration/search or whole-file reads when a
  range will do. Limit it to four inspection batches; if evidence remains
  insufficient, it returns Stop with the missing evidence. A follow-up uses a
  new session and refreshed compact capsule rather than accumulated tool history.

## Prompt template

Read [`references/prompt-template.md`](references/prompt-template.md) immediately
before sending a delegate prompt.

## Async lifecycle invariants

Keep the same session lineage and stable workflow ID for each role. Poll and
fetch the terminal result, then query normalized usage. Do not end the
controller turn or send a final response while a required job is live. Finalize
only after terminal-result retrieval, explicit interruption/stop, or the
declared maximum wait. Cancellation requires terminal error, the phase-defined maximum wait, or
an explicit controller/owner stop decision, with the reason and approved
replacement route recorded. For a full `sol-expert` consultation that has
completed inspection work and then reports `step_start`, keep the same job
through a declared final-synthesis grace of at least 10 minutes, and make the
phase-defined maximum wait no shorter than that grace. Usage is observational
evidence only. Do not add fixed token, price, or provider-private-path limits.

## Async workflow and elapsed-time policy

At implementation-job launch, create the active-job ledger. Read
[`references/wait-policy.md`](references/wait-policy.md) immediately before
polling, recovering, or cancelling; it governs ledger fields, polling,
terminal handling, step-cap continuations, and ledger removal.
