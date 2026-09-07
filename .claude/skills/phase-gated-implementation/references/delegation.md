# Delegation

Read this immediately before an implementation delegation or a Sol-expert
consultation. Return route evidence to the phase controller; do not start a
new phase, pass a gate, publish, or delegate again from this reference.

## Route and evidence

Select the difficulty route named by the core: use `agent="luna"` for
mechanical/economy and standard work; use the configured default only when the
user explicitly requests that route, omitting `agent`, `model`, and `variant`;
or use a bounded `agent="sol-expert"` consultation before Luna for difficult
work. Preserve an explicitly approved whole-phase `agent="sol"` route: do not
call Luna or Terra separately. Keep user-selected models as explicit overrides;
use a raw model only for an explicit user override, an approved degraded
fallback, or the approved Astra escalation below.

### Named effort variants

Normal Luna delegation remains `agent="luna"` with `model` and `variant`
omitted; do not escalate automatically. A named Luna exception may request only
`variant="xhigh"` or `variant="max"`, while retaining the same named agent,
role, model/default, and session lineage. Luna `xhigh` requires a named
complex code phase, documented reasoning bottleneck, why clarification or
decomposition is insufficient, and a fixed acceptance gate. Luna `max` also
requires relevant failure evidence or a bounded Sol-expert recommendation.
Tool unavailability, missing requirements, infrastructure failure, and slow
execution alone are not bottleneck evidence.

Sol, Sol-expert, and Terra remain `high` by default; their `xhigh` is limited
to named difficult analysis or review, and their `max` is explicit-owner only.
Claude remains at configured `high` with no automatic override. Record the
requested agent and variant, evidence, gate, job/session identity, and
resolved-model or usage warnings. A variant does not reset retries or count as
a revised approach; selector, model, or role contradiction still stops.

After Luna `xhigh` or `max`, require exactly one existing independent review:
Codex Terra for Claude-controlled work. Whole-phase Sol keeps its mandatory
Terra review; do not stack reviews or permit preflight-author self-review.

### Optional Astra expert escalation

After bounded Sol consultation, escalate only a named unresolved material
correctness/safety question, or use Astra when multiple coupled boundaries have
severe or irreversible consequences and require integrated analysis, or when the
owner explicitly requests it. Start a fresh read-only session exactly as
`agent="terra", model="openai/gpt-6-astra", variant="high"`. Record the role,
escalation reason and named question, why Luna/Sol or another cheaper route is
insufficient, requested and resolved model, effort, job/session identity,
expected evidence, and stop condition. Astra does not implement, publish,
delegate, waive L4 owner direction, reset retry caps, or persistently upgrade a
later phase. Permit one initial consultation and at most one follow-up; use
`variant="max"` only for a separately named unresolved question supported by
sufficient evidence, otherwise use `high`.

Astra may replace an optional isolated Terra review of the same fresh
read-only concern, but never a required cross-provider Claude/Codex review. A
preflight author cannot independently review its own recommendation. An
unavailable selector or selected model, failed execution, explicit model
contradiction, role mismatch, or silent fallback stops the phase. Missing
resolved-model or usage/observability metadata is only an observability warning
when the requested selector was honored and a terminal response has the
requested role. Retain route and usage observability reporting.

Record requested agent, job ID, session ID, and bound/reported model. An
accepted named-agent selector plus a terminal role response is minimum route
evidence; query normalized provider usage when available. Missing normalized
resolution or partial usage is an observability warning, not a correctness
blocker, when the selector was accepted and a terminal response has the
requested role, with no contradiction. Stop on a rejected or unavailable
selector/model, failed execution, mismatched continuation, explicit model
contradiction, role mismatch, or silent fallback. Do not treat an unavailable
selected model as missing observability metadata. For Astra,
also retain the escalation reason, effort, expected evidence, and stop
condition as separate fields.

## OpenCode caller-working-directory boundary

OpenCode MCP execution uses the caller's working directory; its tools do not
accept a caller-selected `cwd` parameter. The boundary is path-based, not
repository-identity-based, so external paths—including arbitrary `/tmp`
worktrees—are denied. Another repository is reachable only when the
controller first creates an isolated worktree for it under the caller's cwd and
gives the delegate paths scoped to that worktree. Commits and PRs still belong
to that repository's remote. Codex shell tools with an explicit `cwd` are a
separate capability and do not widen OpenCode MCP's delegation boundary. This
is descriptive constraint text; it authorizes no cross-repository work,
worktree operation, editing, publishing, or syncing.

## Bounded sessions and retries

Use one bounded phase/subphase per Luna session; end it after green/completion.
Start or fork a session when scope, red gate, strategy, or blocker changes
materially. Allow at most three implementation/fix attempts. After two failures
on the same blocker, stop blind retries and request one bounded Sol-expert
consultation; allow a third only with a materially revised approach.

Give Luna the compact implementation prompt in [templates.md](templates.md).
Give Sol-expert only the compact capsule, prior diff/test evidence, and one
focused question. It is read-only: it never implements, edits, publishes,
delegates, or invokes repository skills. Permit one initial consultation and at
most one follow-up. Require findings, proposed approach, acceptance gate, then
stop/go. Answer from the capsule when possible. If inspection is necessary,
each batch must resolve one named decision using a relevant range or narrow
symbol—never repository-wide enumeration/search or whole-file reads when a
range suffices. Limit it to four batches; return Stop and missing evidence if
still insufficient. Use a new session and refreshed capsule for a follow-up.

## Whole-phase Sol exception

Use `agent="sol"` only for the explicit approved whole phase. The outer
controller retains final stop/go and publication authority. Sol returns its
diff and evidence without committing, pushing, or creating a PR. Only a
repeated blocker permits one bounded Sol-expert consultation. Report controller
model, requested agent, job/session identity, bound/reported model, and route.
