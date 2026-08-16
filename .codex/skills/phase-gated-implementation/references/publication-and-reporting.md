# Publication and Reporting

Read this before a plan/approval decision, commit, push, PR action, or phase
report. Return publication evidence to the core; do not override its gate.

## Plan-first workflow

Use a plan-only draft PR for phased, non-trivial, or L2–L4 work when
publication is permitted. The committed plan must name scope globs, phases and
dependencies, risk and difficulty, routes, right-reason red/green gates,
acceptance criteria, manual gates, wait policy, publication policy, and
delivery topology: implementation-PR count, phase-to-PR mapping, and every
split boundary. Record its path and SHA in the PR; wait for explicit owner
approval of that SHA in a PR comment before implementation. Plan approval is
not merge approval.

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

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests, formatting,
lint, triggered review, and active-local-session list. Stage only explicit
intended paths; never broad stage. Use `Phase-gate: auto (L1)`,
`Phase-gate: bundle (P<N>)`, or `Phase-gate: manual`. When current phase
authorization and evidence are complete, commit, push, and update its PR.
A manual next-phase gate blocks only entry to the next phase, not this green
phase's publication. Keep a plan PR draft through the final phase and triggered
review; then mark it ready for separate owner merge approval.

For the L1 fast path, record qualification and acceptance evidence in the
normal PR before implementation and wait for separate merge approval. For a
bundle, publish each green phase on the same draft PR and continue without new
plan approval unless it reaches a declared manual boundary or a stop rule. A
topology deviation blocks affected phases; do not use a shared PR to bypass
that stop condition.

## Minimum phase report

Use [templates.md](templates.md) immediately before reporting. Include plan and
approval state, scope/changed files, controller/model and route evidence, risk
and difficulty, sessions/retries/wait status, reviewer state, red/green and
validation evidence, usage warnings, deviations, publication state, and gate.
