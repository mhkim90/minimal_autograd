# Publication and Reporting

Read this before a plan/approval decision, commit, push, PR action, or phase
report. Return publication evidence to the core; do not override its gate.

## Context-bound authorization

A direct owner reply containing exactly `approved` or `approve` is contextual
only for the immediately preceding exact named action prompt. It never fills an
omitted field or expands the named permission. The final merge prompt remains:

```text
Approve merging <repository> PR #<N>?
```

At merge-request time, the controller internally records the repository, PR,
head, base, and relevant check/review eligibility; the user does not provide a
SHA. Bare approval never authorizes an unmentioned action, PR, repository,
phase, readiness, merge, branch-protection or host-control bypass, target
delivery, or downstream sync. Stop for clarification when identity or evidence
is missing or ambiguous. Host, repository, branch-protection, required-check,
and reviewer enforcement remain additional requirements.

For eligible L2/L3 work, use exactly two authorization checkpoints for the
combined draft-plan/implementation topology. The Plan-and-Draft authorization
must explicitly name the repository, exact worktree, branch, base, plan
path/revision, exact plan diff, and one draft PR. It permits only committing the
named plan, pushing the named branch, and creating that named draft PR. It never
permits implementation, readiness, merge, deployment/release, or
cross-repository delivery. After visible PR verification, the Implementation
authorization must explicitly name the repository, PR, branch, and immutable
plan checkpoint P0. It permits only validated in-envelope commits and pushes
to that same draft PR. It never permits merge, deployment/release, or
cross-repository delivery. These are separate from merge authorization.

P0 verification records that the initial PR diff contains exactly the approved
plan, the published plan matches the approved revision, entry gates pass, and
the first implementation head equals P0 except for an explicitly authorized
and inspected pre-implementation change. Expected in-envelope descendants of
P0 do not invalidate implementation authority. Inspect cumulative scope,
topology, and evidence; ancestry alone is insufficient. A plan amendment,
unexpected history, scope/topology change, target branch/repository change,
material base-assumption change, or split boundary pauses affected work and
requires appropriate rebinding or renewal.

Preserve separate merged plan-only PRs for L4 pending owner direction, material
migration, security/trust boundaries, release/deployment, cross-repository,
rollback/ownership, and upstream independent-decision boundaries. Combined
topology does not waive these boundaries. Separately explicit target-specific
authority must name the target repository, exact worktree and branch, exact
paths or diff, and permitted target delivery actions; it authorizes only those
named actions and never a merge.

### Qualified mechanical skill-sync bundle

`skill-sync` alone may replace repeated target approvals, and only after its
complete frozen record establishes one already-merged source revision,
exhaustive portable manifest with content digests, target bases and evaluated
revisions, task-owned isolated branches/worktrees, expected copy-only diffs,
downstream applicability, trusted validation, exclusions, and permitted
operations. Missing/extra files, target-only drift, variants, semantic edits,
untrusted validation, or code/configuration/generated/local-guidance changes
disqualify the bundle.

One explicit delivery authorization may permit only listed isolated copies,
trusted validation, explicit staging, commit, push, draft-PR creation/update,
and administrative readiness. It expires on any frozen-record change and never
permits force-push, adaptation, settings/protection change, auto-merge, bypass,
retry, merge, release, deployment, or an unlisted target. One final merge
request identifies a stable bundle ID and every eligible repository, PR,
expected head, base plus evaluated revision, checks/reviews, and merge method.
After direct approval, revalidate each entry immediately before sequential
expected-head merge. On any drift, eligibility regression, host failure, or
uncertain outcome, stop all remaining entries and report partial completion;
retry, rollback, or any remainder requires fresh exact authority.

Marking a PR ready is administrative after validation and review eligibility
pass; it needs no owner approval. Readiness is not merge authority. No
automatic merge follows from plan approval, draft state, or readiness.

## Plan-first workflow

For eligible L2/L3 work, use the combined two-checkpoint topology when
publication is permitted. Its committed
plan must name scope globs, phases and dependencies, risk and difficulty,
routes, right-reason red/green gates, acceptance criteria, manual gates, wait
policy, publication policy, and delivery topology: implementation-PR count,
phase-to-PR mapping, and every split boundary. A verified plan-only PR
contains no source-policy implementation, executable/configuration/runtime
change, generated output, or downstream sync. Draft is a GitHub PR state, not a
separate delivery artifact. Record its path and current plan-only HEAD SHA as
controller evidence. The initial PR diff must contain exactly the approved plan,
the published plan must match its approved revision, and entry gates must pass.
Record immutable plan checkpoint P0. Plan-and-Draft authorization permits only
the named plan commit, push, and one named draft-PR creation. After visible PR
verification, Implementation authorization is required; it permits only
validated in-envelope commits and pushes to that same draft PR. The first
implementation head must equal P0 except for an explicitly authorized and
inspected pre-implementation change. Expected descendants of P0 do not
invalidate authority, but cumulative scope, topology, and evidence must be
inspected; ancestry alone is insufficient. A plan amendment, unexpected
history, scope/topology change, target change, material base-assumption change,
or split boundary pauses affected work and requires rebinding or renewal.
Preserve separate merged plan-only PRs for L4 pending owner direction, material
migration, security/trust, release/deployment, cross-repository,
rollback/ownership, and upstream independent-decision boundaries. Outside the
combined topology, require that plan-only PR or a complete active unmerged-plan
exception recording its reason, named implementation branch and PR when
available, and concrete resolution event. The existing narrow qualified-L1
direct path or one-named-task waiver retains its exact requirements. At the
resolution event, pause advancement until the prerequisite is resolved or the
exception is explicitly renewed.
Request owner approval of an unambiguous visible plan identity and verify that
the binding still covers the current scope, topology, manual boundaries, and
named ingredients. A plan edit, replacement, ambiguous identity, or unexpected
pre-implementation head drift invalidates plan approval. Required protections,
checks, and reviewers remain in force.

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
one PR only under the exact two-checkpoint topology. Early Plan-and-Draft
authorization is a checkpoint only; final approval occurs after the complete
implementation diff is available. L3 work eligible for that topology may use
it. The listed L4, material, security/trust, release/deployment,
cross-repository, rollback/ownership, and upstream independent-decision
boundaries retain separate plan-only delivery or owner direction.
Adding implementation content outside the named Implementation authorization,
or materially changing scope, topology, risk, affected files or repositories,
acceptance criteria, rollout, sync, manual boundaries, or base assumptions,
invalidates the affected authorization and requires renewed approval.
Protections, required checks, and reviewer rules are never bypassed.

Default to one implementation PR per coherent, independently releasable or
revertible deliverable. A PR can contain multiple phases; its boundary never
replaces phase scope, red/green evidence, acceptance gates, owner decisions,
or triggered review. Split before crossing a material security,
API/compatibility, release, migration, rollback, dependency, ownership,
required-owner-decision, independent-review, or validation-environment
boundary. Adjacent phases may share a PR only when scope, ownership,
validation, and rollback behavior are compatible. A split rationale describes
the boundary; it never pressures unsafe consolidation.

The combined topology has one approved plan, branch, and draft PR. Its plan
names full scope, every phase's scope/gate/dependency, manual boundaries, and
phase-to-PR mapping. Publish each green internal phase to that PR and continue
only in the approved envelope after the two authorization checkpoints. A
topology change, including a new split boundary or incompatible
validation/rollback behavior, requires a revised committed plan and renewed
owner approval before affected work.

Expected green descendants of P0 do not invalidate implementation authority.
Before each delivery, verify that the plan artifact is unchanged, the current
PR head is an expected descendant of P0, and cumulative intervening changes
remain inside the declared phase envelope with current evidence. Ancestry alone
is insufficient: unexpected history, plan amendment, scope, topology, target,
base-assumption, or manual-boundary drift blocks delivery and requires
correction or renewed/rebound approval.

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests, formatting,
lint, triggered review, and active-local-session list. Stage only explicit
intended paths; never broad stage. Use `Phase-gate: auto (L1)`,
`Phase-gate: bundle (P<N>)`, or `Phase-gate: manual`. When current phase
authorization and evidence are complete, present one exact pending delivery
action and apply only the current checkpoint's bounded authorization. A grouped
delivery may commit, push, and create or update one named draft PR only under
the Plan-and-Draft checkpoint; implementation delivery requires the separate
named Implementation checkpoint above.
A manual next-phase gate blocks only entry to the next phase, not this green
phase's publication. The combined topology may commit, push, and update phase
evidence on its same draft PR after each declared green phase, subject to the
expected-descendant and cumulative-scope verification above. This is
delivery-only: it cannot replace a PR, mark ready, merge, bypass host controls,
release, deploy, or publish elsewhere. Source-side authority does not transfer
to target delivery or downstream sync. Separately explicit
target-specific authority may authorize only its named target delivery actions;
it never authorizes a merge. Keep the
plan-only PR in GitHub draft state while verifying it; after validation and
required review/eligibility checks pass, mark it ready administratively. For
every PR, bind the eligible repository, PR, head, base, and check/review
eligibility snapshot, then make one exact merge request and obtain its final
owner approval. Immediately revalidate the bound values, and merge only with
GitHub's approved-head/expected-head precondition and normal merge authority.
If the head, base, or eligibility changes, or eligibility regresses, invalidate
the approval and request fresh approval; revalidation cannot revive invalid
authority. Keep an implementation PR's merge authority separate from plan
approval. Do not use plan-only authorization to merge an implementation PR.

For the L1 fast path, record qualification and acceptance evidence in the
normal PR before implementation and wait for separate final merge approval.
For the combined topology, publish each green phase on the same draft PR and
continue without a new plan approval unless it reaches a declared manual
boundary or a stop rule. A topology deviation blocks affected phases; do not
use a shared PR to bypass that stop condition.

## Minimum phase report

Use [templates.md](templates.md) immediately before reporting. Include plan and
approval state, scope/changed files, controller/model and route evidence, risk
and difficulty, sessions/retries/wait status, reviewer state, red/green and
validation evidence, usage warnings, deviations, publication state, and gate.
