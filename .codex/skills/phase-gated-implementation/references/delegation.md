# Delegation

Read this immediately before an implementation delegation or an Astra-expert
consultation. Return route evidence to the phase controller; do not start a
new phase, pass a gate, publish, or delegate again from this reference.

## Route and evidence

Apply the core's [route and selector rules](../SKILL.md#routes-and-selector-stops).
This reference adds delegation evidence: record the requested agent and
variant (omitted for named routes), acceptance gate, job/session identity,
bound/reported model, and warnings. A rejected or unavailable selector/model,
mismatched continuation, explicit model contradiction, role mismatch, or silent
fallback is a stop; missing resolved-model or usage
metadata is only an observability warning after the selected route is honored
and the requested role returns a terminal response.

### Named model binding

The core's [effort and review policy](../SKILL.md#effort-and-review) is
authoritative. Named agents bind exact GPT-6 `#high` profiles; omit per-call
model and variant. Record requested agent, bound/reported model, retry
accounting, and observability warnings.

### Astra expert consultation

Follow the core's [Astra eligibility and limits](../SKILL.md#routes-and-selector-stops).
For a selected consultation, record the escalation reason, named question, why
the cheaper route is insufficient, requested/resolved model, effort, job/session
identity, expected evidence, and stop condition. Preserve the read-only role
and do not silently substitute a selector, model, role, or continuation.

## OpenCode caller-working-directory boundary

OpenCode MCP execution uses the caller's working directory; its tools do not
accept a caller-selected `cwd` parameter. The boundary is path-based, not
repository-identity-based, so external paths—including arbitrary `/tmp`
worktrees—are denied. Path-contained source/worktrees do not prove access to
linked/common Git metadata. Follow the
[OpenCode Delegate boundary](../../opencode-delegate/SKILL.md#boundary) for
actual-delegate technical proof, runtime/tool failure classification, and
scoped cross-repository checkout/clone authority. Commits and PRs belong to
the verified source repository's configured remote, not the caller repository
or an arbitrary clone destination. Shell tools with an explicit `cwd` are a
separate capability and do not widen OpenCode MCP's delegation boundary.
This descriptive constraint text authorizes no cross-repository work,
worktree operation, editing, publishing, or syncing.

## Bounded sessions and retries

Load the [delegate capability/recovery contract](../../opencode-delegate/SKILL.md#boundary)
before delegation and its [bounded lifecycle](../../opencode-delegate/SKILL.md#bounded-lifecycle)
for candidate accounting and expert limits. Return failure evidence to the
controller, not an automatic expert detour or a delegate-owned recovery launch.
Preserve prior jobs, scoped partial diff, caps, and original plan/P0 provenance
in any expressly preauthorized controller recovery.

Give the implementer the [canonical prompt](../../opencode-delegate/references/prompt-template.md);
[templates.md](templates.md#implementation-prompt) retains its loader. Give
Astra-expert only the compact capsule, relevant prior diff/test evidence, and
one focused question. Record question, cheaper-route insufficiency, expected
evidence, and stop condition; return findings, proposed approach, acceptance
gate, then stop/go. Its inspection is scoped and decision-driven within the
declared wait/stop limits; stop when evidence is unavailable or further work no
longer addresses the question. The delegate contract owns its read-only role
and one-consultation/one-follow-up limit.

## Controller-bound Astra orchestration exception

Use `agent="astra-orchestrator"` only for the explicit approved whole phase
beneath an external Codex or Claude controller. The outer controller retains
final stop/go, independent review, and publication authority. Astra returns its
diff and evidence without committing, pushing, or creating a PR. Only a
repeated blocker permits one bounded Astra-expert consultation. Report controller
model, requested agent, job/session identity, bound/reported model, and route.
