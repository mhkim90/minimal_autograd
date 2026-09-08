# Publication

Read this before a plan/approval decision, commit, push, delivery-draft
action, or phase report. Return publication evidence to the core; do not
override its gate.

## Ticket identification

A Jira ticket ID is mandatory before any commit or delivery draft for an
initiative. If the owner hasn't given one yet, ask for it once at
preflight (Sequence step 1) before the first commit — do not guess,
infer, or omit it. Prefix every commit message with `TICKET-ID: ` and the
delivery draft's title the same way. Record the ticket ID in the plan
artifact and in the phase report. A missing ticket ID is a stop
condition, same as a missing plan or gate.

## Commit-first, PR-free default

The closed-network default is commit-based publication with no real PR,
ever: land green work as local commits on the working branch. **Never
open, update, mark ready, or merge a real PR — there is no exception, not
even on explicit request.** The closed side always batches actual
submission by hand from a human, outside this tool. What this skill
produces instead, when a submission is due, is a markdown delivery draft
for the owner to approve — see "Delivery draft" below.

This changes only the publication vehicle. Every other gate still applies
in full: an approved plan for phased/non-trivial/L2-L4 work, per-phase
red/green evidence, acceptance gates, `isolated-review`, escalation
outcomes, scope limits, and stop rules. Commit-first is not a fast path.

## Plan approval without a PR

For phased, non-trivial, or L2-L4 work, write the plan as a reviewable
artifact at `delivery/<TICKET-ID>-<YYYYMMDD>-<topic-slug>/plan.md`
before implementation. It is never staged or committed — the closed
side's git policy excludes plan files from the repository, and a plan
commit would pollute the history the human later batches into a
submission. It must name scope globs, phases and dependencies, risk and
difficulty, right-reason red/green gates, acceptance criteria, manual
gates, wait policy, and delivery topology (see below). Since there is no
commit SHA to tie approval to, record the file's content hash (e.g.
`sha256sum`) and wait for explicit owner approval tied to that exact
hash before implementation. Record the approval where it is durable —
an approval note appended to the plan artifact (the note names the
approved pre-note hash) or the phase report. Material changes to scope,
phase order, risk, gate, or delivery topology change the hash and
invalidate that approval; re-approve against the new hash.

An exact L1 mechanical/economy task may skip the plan artifact only when
it is demonstrably behavior-preserving and has authorized exact
paths/scope, acceptance checks, and publication intent (see the L1 fast
path exclusions in the core). When uncertain, write the plan.

## Owner approval is per-action

A direct owner reply of `approved`/`approve` authorizes exactly one
concrete pending publication action that was named immediately before it.
Name the target explicitly when asking: repository/worktree, branch, the
exact changed-path scope or verified head, and whether the action is a
commit, a push, or approval of a delivery draft's content. One approval
may group commit and push of the same verified diff.

The reply expires on any change of target, scope, verified head, or
action type. It never authorizes a later phase, another repository, or a
delivery draft covering a different commit range than what was named.
Plan approval is not publication approval, and neither is a prior
approval of a different action.

## Delivery topology

Topology still matters even when publication is commit-first, because it
determines where phases may be batched into one delivery draft later and
where they must not be.

Default to one coherent, independently releasable or revertible
deliverable per eventual delivery draft. Multiple phases may share one;
that boundary never replaces phase scope, red/green evidence, acceptance
gates, owner decisions, or a triggered `isolated-review`/`airgap-export`.
Split before crossing a material security, API/compatibility, release,
migration, rollback, dependency, ownership, required-owner-decision,
independent-review, or validation-environment boundary. Adjacent phases
may be batched together only when scope, ownership, validation, and
rollback behavior are compatible. A split rationale describes the
boundary; it never pressures unsafe consolidation.

An initiative bundle has one approved plan and one branch. Its plan names
full scope, every phase's scope/gate/dependency, manual boundaries, and
the intended phase-to-delivery-draft mapping for whenever a draft is due.
Commit each green internal phase to that branch and continue only inside
the approved envelope. A topology change, including a new split boundary
or incompatible validation/rollback behavior, requires a revised plan
artifact (new hash) and renewed owner approval before affected work.

## Commit granularity

Default commit shape per initiative is activity-based, not one commit per
phase: one implementation commit, one test commit, and one fix commit per
review round (`fix-after-review-1`, `fix-after-review-2`, ...). As phase
commits land, consolidate (squash/amend) them into that shape before they
accumulate — this matters because the closed side batches many commits
into a submission by hand later, and per-phase granularity does not scale
to that. Consolidate only local, not-yet-pushed commits; never rewrite
history that has already been pushed.

When folding multiple phases into one commit, keep one `Phase-gate:`
trailer per folded phase, in original order, in the resulting commit
message — see the core Gate section for why (the auto-continuation count
scans trailer occurrences, not commits).

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests,
formatting, lint, `isolated-review`/`airgap-export` outcome, and
active-local-session list. Stage only explicit intended paths; never
broad stage. Use `Phase-gate: auto (L1)`, `Phase-gate: bundle (P<N>)`, or
`Phase-gate: manual`. Prefix the commit message with the ticket ID (see
"Ticket identification" above). When current phase authorization and
evidence are complete, commit. Push only when the user has asked for it
or the plan declares it; a refusal by the host (branch protection,
required checks) stops the action rather than authorizing a different
form of it. A manual next-phase gate blocks only entry to the next
phase, not this green phase's commit.

A topology deviation blocks affected phases; do not batch across it to
work around that stop condition.

## Delivery draft (markdown, no real PR)

There is no real-PR path, not even on explicit request — see "Commit-first,
PR-free default" above. When an initiative/bundle's planned phases are
all green and committed, or whenever the owner asks for one covering an
explicit commit range (whichever comes first), write a markdown delivery
draft instead:

- **Location**: `delivery/<TICKET-ID>-<YYYYMMDD>-<topic-slug>/PR.md` at
  the repository root — the same ticket folder that holds the plan
  artifact (`plan.md`, see "Plan approval without a PR" above). If the
  owner has supplied real-test evidence (not unit tests — those stay in
  the repo's normal test suite), place it under
  `delivery/<TICKET-ID>-<YYYYMMDD>-<topic-slug>/tests/`. Never stage or
  commit anything under `delivery/`; it is local working material, not a
  repository artifact.
- **Content**: title `TICKET-ID: <summary>`, branch, base, exact commit
  range/SHAs, phase-to-commit mapping, a change summary, a test plan
  (unit coverage plus a pointer to `tests/` evidence if present),
  delivery-topology notes (see above), and known limitations.
- **Approval**: the owner approves the draft's content under the
  per-action rule above — name the exact draft path and commit range
  covered. Approval is terminal: take no further action on it. Do not
  open, ready, or merge anything; do not push beyond what commits already
  had their own approval. The human batches the actual submission later,
  out of band, using this file as source material. Any change to the
  covered commit range or the draft's scope invalidates a prior approval.

## Phase report

```text
Ticket: <TICKET-ID>
Phase <N> complete: <commit or uncommitted state>
Publication: <committed | committed+pushed | stopped before prohibited action>
Plan preflight: <artifact/revision, audit, freshness, manual gates | L1 fast path: behavior-preserving qualification | initiative bundle: phase envelope and manual boundaries>
Plan artifact: <delivery/<path>/plan.md@sha256:<short-hash> | none: L1 fast path>; owner approval: <hash-tied evidence | L1 qualification | pending/invalidated>
Delivery draft: <none | delivery/<path>/PR.md@<version>, owner-approved | pending approval>
Delivery topology: <intended phase-to-delivery-draft mapping; included phases; split boundary/rationale | direct L1 qualification>; topology deviation: <none or renewed-approval state>
Scope: <approved globs>; changed files: <list>
Safety risk level: <L1-L4>; implementation difficulty: <mechanical/economy | standard | difficult>
Implementation attempts: <count>
Isolated review: <verdict, or skipped-and-why>
Airgap escalation: <case id + consultation/follow-up count + outcome, or none>
Red gate: <right failure then pass>; attempts: <used>/<cap>
Validation: <commands/results>
Plan deviations: <none or rationale>
Gate: <committed; auto-proceeding | committed; proceeding-within-approved-initiative-bundle | committed; waiting-for-next-phase-approval | blocked>
```
