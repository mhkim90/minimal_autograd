# Delegation

Read this immediately before an implementation delegation or a Sol-expert
consultation. Return route evidence to the phase controller; do not start a
new phase, pass a gate, publish, or delegate again from this reference.

## Route and evidence

Apply the core's [route and selector rules](../SKILL.md#usage-correlation).
This reference adds delegation evidence: record the requested agent and
variant, named evidence, acceptance gate, job/session identity, bound/reported
model, and warnings. A rejected or unavailable selector/model, failed
execution, mismatched continuation, explicit model contradiction, role
mismatch, or silent fallback is a stop; missing resolved-model or usage
metadata is only an observability warning after the selected route is honored
and the requested role returns a terminal response.

### Named effort variants

The core's [effort-variant policy](../SKILL.md#effort-variant-exceptions) is
authoritative. For a named exception, retain its requested agent/variant,
bottleneck evidence, fixed gate, retry accounting, and observability warnings
in the delegation record.

### Optional Astra expert escalation

Follow the core's [Astra eligibility and limits](../SKILL.md#usage-correlation).
For a selected consultation, record the escalation reason, named question, why
the cheaper route is insufficient, requested/resolved model, effort, job/session
identity, expected evidence, and stop condition. Preserve the read-only role
and do not silently substitute a selector, model, role, or continuation.

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
