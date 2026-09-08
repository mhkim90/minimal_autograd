---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Claude Code as final publication gate, route implementation by difficulty, and load policy references at their transitions.
---

# Phase-Gated Implementation

Claude Code controls scope, gates, publication, and stop/go; do not implement
by default. Record the active controller model from runtime—Sonnet is intended,
not evidence about this session. The `codex-delegate` contract is read-only;
interactive Claude Code only publishes.

## Usage correlation

When `usage_mcp` is configured, load
[`references/usage-correlation.md`](references/usage-correlation.md)
immediately before starting, handling terminal provider work, or closing a
correlated workflow. Do not infer or reconstruct a missing controller identity.

Keep safety risk (L1–L4) separate from implementation difficulty. Use
`agent="luna"` for mechanical/economy and standard work, and use OpenCode's
configured default only when the user explicitly requests that route, omitting
`agent`, `model`, and `variant`. Use a bounded `agent="sol-expert"` preflight or
breakthrough only for difficult work before Luna implements. Preserve an
explicitly approved whole-phase `agent="sol"` route without separately calling
Luna or Terra. An optional Astra expert escalation is a fresh, read-only
`agent="terra", model="openai/gpt-6-astra", variant="high"` session only when
a bounded Sol consultation leaves a named material correctness/safety question
unresolved, multiple coupled boundaries have severe or irreversible
consequences and need integrated analysis, or the owner explicitly requests
Astra. Record the question, why the cheaper route is insufficient, expected
evidence, and stop condition. Permit one initial Astra consultation and at most
one follow-up; `max` requires a separately named unresolved question and
sufficient evidence. Astra never implements, publishes, delegates, waives L4
owner direction, resets retry caps, or persistently upgrades later phases.
The approved Astra route is the only additional raw-model override beyond an
explicit user override or degraded fallback. Never silently replace a selected
route or raw-model override: selector rejection, selected selector/model
unavailability, explicit model contradiction, role mismatch, silent fallback,
or failed execution stops work. Only after the selector is honored, the
requested role returns a terminal response, and no contradiction exists may
missing resolved-model or usage metadata produce an observability warning.

## Effort-variant exceptions

- Normal named Luna routes remain `agent="luna"` with `model` and `variant`
  omitted. There is no automatic escalation from the configured default.
- A named Luna exception may request only `variant="xhigh"` or
  `variant="max"`.
  It retains the same named agent, role, model/default, and session lineage;
  it never silently substitutes a model, role, route, or continuation.
- Luna `xhigh` requires all of: a named complex code phase, a documented
  reasoning bottleneck, why clarification or decomposition is insufficient,
  and a fixed acceptance gate. Luna `max` additionally requires relevant
  failure evidence or a bounded Sol-expert recommendation. Tool
  unavailability, missing requirements, infrastructure failure, and slow
  execution alone are not bottleneck evidence.
- Sol, Sol-expert, and Terra remain `high` by default. Their `xhigh` is only
  for named difficult analysis or review; `max` is explicit-owner only. Astra
  keeps its existing fresh read-only `high` route and bounded,
  evidence-supported `max` exception. Claude remains at configured `high`;
  no automatic override is permitted.
- Record the requested agent and variant, named evidence, acceptance gate,
  job/session identity, and resolved-model or usage warnings. An exception
  does not reset retries or count as a materially revised approach.
  Selector, model, or role contradiction still stops the phase.
- After Luna `xhigh` or `max`, require exactly one existing
  runtime-appropriate independent review: Codex Terra for Claude-controlled
  work. Whole-phase Sol retains its mandatory Terra review. Do not stack
  reviews or let a preflight author review its own recommendation.

## Effort-variant scenario matrix

The phase gate accepts and records default Luna with omitted variant, qualifying
Luna `xhigh`, Luna `max` with its additional evidence, qualifying Sol/Terra
`xhigh`, the owner-only Sol/Terra `max`, the existing Astra exception, and
unchanged configured Claude `high`. It rejects or stops automatic escalation,
role/model substitution, retry resets, tool/infrastructure/missing-requirement
or slowness evidence, a second review, and preflight-author self-review.

## Invariants

### Authorization and P0 verification

- Use an exact approved plan and the declared delivery topology for phased,
  non-trivial, or L2–L4 work. For eligible L2/L3 work, a combined
  draft-plan/implementation topology may use one draft PR, but it has exactly
  two authorization checkpoints. First, the Plan-and-Draft authorization must
  explicitly name the repository, exact worktree, branch, base, plan path/revision,
  exact plan diff, and one draft PR; it permits only committing
  that plan, pushing, and creating that named draft PR. It never permits
  implementation, readiness, merge, deployment/release, or cross-repository
  delivery. After visible PR verification, the Implementation authorization
  must explicitly name the repository, PR, branch, and immutable plan
  checkpoint P0; it permits only validated in-envelope commits and pushes to
  that same draft PR. It never permits merge, deployment/release, or
  cross-repository delivery.
   **Starting check:** P0 verification requires that the initial PR diff contain
   exactly the approved plan, the published plan match the approved revision,
   entry gates pass, and the first implementation head equal P0 except for an
   explicitly authorized and inspected pre-implementation change.
   **Before each delivery:** expected in-envelope descendants of P0 do not
   invalidate implementation authority, but inspect the cumulative scope,
   topology, and evidence; ancestry alone is insufficient. Verify that the plan
   artifact is unchanged, the current PR head is an expected descendant of P0,
   and cumulative intervening changes remain inside the declared phase envelope
   with current evidence. **Drift:** a plan amendment, unexpected history,
   scope/topology change, target branch/repository change, material
   base-assumption change, or split boundary pauses affected work and
   requires appropriate rebinding or renewal. A plan edit, replacement,
   ambiguous identity, or unexpected pre-implementation head drift invalidates
   plan approval.
  Preserve separate merged plan-only PRs for L4 pending owner direction,
  material migration, security/trust boundaries, release/deployment,
  cross-repository, rollback/ownership, and upstream independent-decision
  boundaries; combined topology never waives those boundaries. A plan-only PR
  contains no implementation before its named checkpoint, generated output,
  or downstream sync. The plan declares scope, gates, risk/difficulty/routes,
  wait policy, dependencies, manual boundaries, and delivery topology.
  Plan-and-Draft approval authorizes only its named plan publication actions;
  Implementation approval authorizes only its named in-envelope delivery
  actions. Neither authorizes readiness, merge, or another later publication
  action. Outside the exact combined topology, require the applicable merged
  plan-only PR or complete active unmerged-plan exception, recording its reason,
  named implementation branch and PR when available, and concrete resolution
  event. The existing narrow qualified-L1 direct path, or a recorded
  one-named-task plan-only waiver, is the only alternative and retains its
  exact qualification, scope, validation, delivery, and final PR-specific
   merge-approval requirements. Any implementation outside the named envelope,
  or material change to scope, topology, risk, affected files or repositories,
  acceptance criteria, rollout, sync, manual boundaries, or base assumptions,
  invalidates the affected authorization and requires renewed approval.
  Protections, required checks, and reviewer rules are never bypassed.

### Contextual approval

- A direct owner reply containing exactly `approved` or `approve` is contextual
  authorization only for the immediately preceding exact named action prompt;
  it never supplies an omitted repository, worktree, branch, base, plan,
  checkpoint, PR, diff, or permission. The final merge prompt remains exactly
  `Approve merging <repository> PR #<N>?`. At merge-request time, the
  controller internally records the repository, PR, head, base, and relevant
  check/review eligibility; the user does not provide a SHA. Any head, base, or
  eligibility drift invalidates the approval and merge authority; immediate
  revalidation or rebinding cannot restore it without fresh exact approval.
  Bare approval never transfers across PRs or repositories, a later phase,
  downstream sync, a branch-protection or host control bypass, or an
  unmentioned action. Host, repository, branch-protection, required-check, and
  reviewer enforcement remain additional requirements. The only exception is a
  qualified mechanical skill-sync bundle defined by `skill-sync`: its frozen
  delivery record and final fully enumerated merge snapshot replace repeated
  target prompts only; they never apply to source implementation,
  code/configuration, local guidance, variants, or generic cross-repository work.

### Delivery topology

- Where no independent plan gate is required, a plan and implementation may
  share one PR under the exact two-checkpoint topology above. Early
  Plan-and-Draft authorization is only a checkpoint; final merge approval
  comes after the complete implementation diff is available. L3 work eligible
  for the combined topology may use it; L4 and the listed material or
  independent-decision boundaries retain separate merged plan-only PRs or
  owner direction as required. Any exception must meet the unmerged-plan
  exception and pause rules above.
- Default to one implementation PR per coherent, independently releasable or
  revertible deliverable. One PR may contain multiple phases, but never
  replaces each phase's scope, red/green evidence, acceptance gate, owner
  decision, or triggered review.
- Split at a material security, API/compatibility, release, migration,
  rollback, dependency, ownership, required-owner-decision, independent-review,
  or validation-environment boundary. Bundle adjacent phases only when their
  approved scope, ownership, validation, and rollback behavior are compatible.
  A split rationale describes that boundary; it never makes unsafe bundling
  acceptable.
- An L1 fast path needs exact authorized scope, checks, and publication intent;
  it never waives explicit staging, diff/acceptance evidence, stop rules, or
  final merge authority. Readiness is administrative after validation and
  review; it needs no owner approval.
  A combined topology may continue only inside its approved envelope and stops
  at declared manual boundaries.
- Before every phase, check dirty state and both pause files, obtain minimum
  scope/gate/current evidence, run the smallest right-reason red gate, and set
  a three-attempt cap unless the plan says otherwise.
- Validate the in-scope diff and green evidence before publication. Publish a
  green phase only with its trailer and explicit-path staging. Keep a
  plan-only PR in GitHub draft state while verifying it; after validation and
  required review/eligibility checks pass, mark it ready administratively.
  Before merging any PR, bind the eligible repository, PR, head, base, and
  check/review eligibility snapshot; make the exact merge request; obtain the
  final owner approval; immediately revalidate the binding; then merge only with
  GitHub's approved-head/expected-head precondition and normal merge authority.
  No automatic merge follows from plan approval, readiness, or draft state.
  Keep an implementation PR's merge authority separate from plan approval. A
  manual next-phase gate does not prevent publication of the completed phase.
- Stop and report on stale/missing preflight, wrong/failed gates, scope or
  intent expansion, unverifiable correctness, route contradiction, unavailable
  evidence, review blocker, maximum wait, unrelated blocking changes, or
  attempt-cap exhaustion. Do not weaken a gate to continue.
- Memory is discovery-only: read the applicable continuity record before
  resuming or crossing repositories, validate it against live source, then
  save decisions, blockers, and handoff evidence. It never supplies approval.

## Router and lazy policy loading

Load the named reference immediately before its transition, never merely on
phase entry. References explain *how* and return evidence; this core alone
decides whether to route, pass a gate, publish, or start another phase.

| Trigger | Codex destination | Claude destination |
| --- | --- | --- |
| Active agent drafted substantive plan | `grilled-me` | `grilled-me` |
| User/repo/issue/external substantive author | `plan-audit` | `plan-audit` |
| Implementation delegation | `opencode-delegate` + `references/delegation.md` | `opencode-delegate` + [`references/delegation.md`](references/delegation.md) |
| Triggered independent read-only review | `claude-delegate` + `references/review-and-wait-policy.md` | `codex-delegate` + [`references/review-and-wait-policy.md`](references/review-and-wait-policy.md) |
| Resume/cross-repository work | `memory-continuity` | `memory-continuity` |
| Transfer to a fresh session | `handoff` | `handoff` |

Editing or summarizing externally authored material does not change its origin.
Read [`references/publication-and-reporting.md`](references/publication-and-reporting.md)
before any plan/approval decision, commit, push, PR action, or report; read
[`references/templates.md`](references/templates.md) immediately before using
a prompt, capsule, or phase report.

## Gate

Inspect active local sessions, the contiguous commit suffix, and evidence
before the gate. Two consecutive `Phase-gate: auto` trailers force the next
L1 phase manual; bundle trailers do not count, and unexpected interleaving is
scope drift. Missing approved per-phase scope disables automatic continuation.
An L1 phase proceeds only when final L1, in scope, right red failure then pass
within cap, no deviation, and no review blocker. A bundle phase proceeds only
when declared, green, topology-compatible, and free of a manual boundary or
stop condition. A new split boundary or validation/rollback mismatch is a
plan-topology deviation and blocks affected work pending renewed approval.
Report the current phase and whether the next one proceeds or waits.
