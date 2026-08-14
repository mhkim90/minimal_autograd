# Delegation

Read this immediately before an implementation delegation or a Sol-expert
consultation. Return route evidence to the phase controller; do not start a
new phase, pass a gate, publish, or delegate again from this reference.

## Route and evidence

Select the difficulty route named by the core: omit `agent`, `model`, and
`variant` for configured-default mechanical/economy work; use `agent="luna"`
for standard work; or use a bounded `agent="sol-expert"` consultation before
Luna for difficult work. Preserve an explicitly approved whole-phase
`agent="sol"` route: do not call Luna or Terra separately. Keep user-selected
models as explicit overrides; use a raw model only for an approved degraded
fallback.

Record requested agent, job ID, session ID, and bound/reported model. An
accepted named-agent selector plus a terminal role response is minimum route
evidence; query normalized provider usage when available. Missing normalized
resolution or partial usage is an observability warning, not a correctness
blocker, when the selector was accepted and nothing contradicts its returned
role/model. Stop on a rejected/unavailable selector, mismatched continuation,
explicit model contradiction, or silent fallback.

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
