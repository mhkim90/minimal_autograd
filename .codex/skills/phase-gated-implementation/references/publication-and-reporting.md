# Publication and Reporting

Read this before a plan/approval decision, commit, push, PR action, or phase
report. Return publication evidence to the core; do not override its gate.

## Context-bound direct approval

A direct owner reply containing exactly `approved` or `approve` is valid only
when it immediately follows a prompt naming one concrete pending publication
action. The prompt must identify the exact repository and worktree, branch and
PR when applicable, verified head SHA or exact changed-path scope, and whether
the action is delivery, readiness, merge, or qualified readiness-and-merge. One delivery action may group
commit, push, and creation or update of one named draft PR only when all use
the same verified branch and diff. The reply authorizes only that action and
expires on any target, verified-head, changed-path/scope, or action change.
Stop for clarification when the evidence or action is missing or ambiguous.

## Qualified combined readiness-and-merge action

One direct owner reply may authorize marking one PR ready and immediately
merging that same PR only when its prompt explicitly names both actions and the
repository/worktree, branch, PR, and verified head or exact changed-path scope.
It is available only for an unchanged, source-local documentation or
skill-instruction deliverable. It excludes application/runtime code,
dependencies, generated output, workflow or release configuration,
security-sensitive changes, migrations, deployment, and downstream
synchronization.

Before asking, verify that validation, triggered independent review, and
host-required checks are green and that no review thread or required owner
decision remains. Immediately before merging, verify the PR head again and use
it as GitHub's expected merge head. Any head, scope, target, check/review state,
or action drift invalidates the reply. The action never bypasses branch
protection, reviewer requirements, merge permissions, or host controls; a host
refusal stops the action rather than authorizing a retry under a different form.

This is final owner authorization, not automatic merge authority. It is never
inferred from plan approval, draft state, a prior delivery reply, or a generic
reply that did not expressly name both readiness and merge. It does not apply
to target-sync or other cross-repository delivery.

Owner approval of the exact current verified head of a genuine plan-only PR
continues to authorize readiness and merge of that same unchanged plan-only PR
after required checks/reviews. At execution, the controller may present the
unchanged PR/head and already-authorized readiness-and-merge sequence without
requiring a new direct reply. This exception never applies to an
implementation PR.

It never authorizes another repository or PR, a later phase, downstream sync,
a branch-protection or other host-control bypass, or a merge not explicitly
named. Plan-only authorization remains limited to the same unchanged plan-only
PR/head and never merges an implementation PR. An implementation merge requires
its own exact pending merge action and normal merge authority. Host, repository,
branch-protection, required-check, and reviewer enforcement remain additional
requirements.

## Plan-first workflow

Use a verified plan-only PR for phased, non-trivial, or L2–L4 work when an
independent plan gate is required and publication is permitted. Its committed
plan must name scope globs, phases and dependencies, risk and difficulty,
routes, right-reason red/green gates, acceptance criteria, manual gates, wait
policy, publication policy, and delivery topology: implementation-PR count,
phase-to-PR mapping, and every split boundary. A verified plan-only PR
contains no source-policy implementation, executable/configuration/runtime
change, generated output, or downstream sync. Draft is a GitHub PR state, not a
separate delivery artifact. Record its path and current plan-only HEAD SHA as
controller evidence. Request owner approval of an unambiguous visible identity,
normally the plan title plus draft PR; a unique commit title or named ingredients
also suffice. The owner need not quote a SHA. Codex resolves that identity to
the current plan artifact and records its HEAD internally. Before implementation,
verify the binding still covers the current scope, topology, manual boundaries,
and named ingredients. A plan edit, replacement, ambiguous identity, or
unexpected pre-implementation head drift invalidates approval. Plan approval is
not readiness or merge approval; required protections, checks, and reviewers
remain in force.

An exact L1 mechanical/economy task may use a direct implementation PR only
when it is demonstrably behavior-preserving and has authorized exact
paths/scope, acceptance checks, and publication intent. It excludes policy,
architecture, API, security, runtime/release configuration, dependencies,
migrations, generated outputs, workflows, approval behavior, and operational
documentation, as well as any new cross-repository rollout unless it uses an
already merged source revision and exact manifest. When uncertain, use the
plan-first workflow. An owner may waive the plan-only gate for one named
bounded task only; record it and do not generalize it. If publication is
prohibited, retain all local scope and acceptance gates and report why no PR
exists.

Where no independent plan gate is required, a plan and implementation may share
one PR. Early plan approval is a checkpoint only; final approval occurs after
the complete implementation diff is available. L3 and other independent-plan-
gate work retain a merged plan-only PR followed by a separate implementation PR.
Adding implementation content or materially changing scope, risk, affected
files or repositories, acceptance criteria, rollout, or sync invalidates the
plan-only authorization envelope. Editorial clarification is not a material
scope change, but every changed PR head—including editorial-only changes—must
be verified and owner-approved before ready or merge.

Default to one implementation PR per coherent, independently releasable or
revertible deliverable. A PR can contain multiple phases; its boundary never
replaces phase scope, red/green evidence, acceptance gates, owner decisions,
or triggered review. Split before crossing a material security,
API/compatibility, release, migration, rollback, dependency, ownership,
required-owner-decision, independent-review, or validation-environment
boundary. Adjacent phases may share a PR only when scope, ownership,
validation, and rollback behavior are compatible. A split rationale describes
the boundary; it never pressures unsafe consolidation.

An initiative bundle has one approved plan, branch, and draft PR. Its plan
names full scope, every phase's scope/gate/dependency, manual boundaries, and
phase-to-PR mapping. Publish each green internal phase to that PR and continue
only in the approved envelope. A topology change, including a new split
boundary or incompatible validation/rollback behavior, requires a revised
committed plan and renewed owner approval before affected work.

Expected green bundle commits do not invalidate the approved plan. Before each
delivery, verify that the plan artifact is unchanged, the current PR head is
the expected descendant of the verified plan-only head, and intervening changes
remain inside the declared phase envelope. Unexpected history, plan, scope,
topology, or manual-boundary drift blocks delivery and requires correction or
renewed approval.

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests, formatting,
lint, triggered review, and active-local-session list. Stage only explicit
intended paths; never broad stage. Use `Phase-gate: auto (L1)`,
`Phase-gate: bundle (P<N>)`, or `Phase-gate: manual`. When current phase
authorization and evidence are complete, present one exact pending publication
action and apply only its contextual direct approval. A grouped delivery may
commit, push, and create or update one named draft PR only under the rule above.
A manual next-phase gate blocks only entry to the next phase, not this green
phase's publication. An approved initiative bundle may commit, push, and update
phase evidence on its same draft PR after each declared green phase without a
fresh owner reply, subject to the expected-ancestry verification above. This is
delivery-only: it cannot replace a PR, mark ready, merge, bypass host controls,
release, deploy, publish elsewhere, or authorize downstream sync. Keep a plan-only PR in GitHub draft state while verifying
it; after its current verified head is owner-approved and required checks/
reviews pass, mark it ready and merge it under that existing exact-head
authorization; at execution, present the unchanged PR/head and
already-authorized readiness-and-merge sequence without requesting a new direct
reply. Keep an implementation PR draft through final validation and triggered
review; then present readiness, merge, or—only when the qualification above
holds—one explicit readiness-and-merge action with final owner approval and
normal merge authority. Do not use plan-only
authorization to merge an implementation PR.

For the L1 fast path, record qualification and acceptance evidence in the
normal PR before implementation and wait for separate final merge approval
unless the qualified combined action above applies. For a
bundle, publish each green phase on the same draft PR and continue without new
plan approval unless it reaches a declared manual boundary or a stop rule. A
topology deviation blocks affected phases; do not use a shared PR to bypass
that stop condition.

## Minimum phase report

Use [templates.md](templates.md) immediately before reporting. Include plan and
approval state, scope/changed files, controller/model and route evidence, risk
and difficulty, sessions/retries/wait status, reviewer state, red/green and
validation evidence, usage warnings, deviations, publication state, and gate.
