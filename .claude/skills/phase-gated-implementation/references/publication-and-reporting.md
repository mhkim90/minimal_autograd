# Publication and Reporting

Read this before a plan/approval decision, commit, push, PR action, or phase
report. Return publication evidence to the core; do not override its gate.

## Context-bound authorization

The core's [authorization and P0 rules](../SKILL.md#authorization-and-p0-verification)
define authority, contextual approval, protected boundaries, and the complete
invalidation set. This reference supplies the publication order below; it does
not add authority or override the core gate.

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

## PR-lifecycle ledger gate

Create or update the non-authoritative ledger for every workflow-touched PR,
including created, adopted, updated, readied, reopened, closed, and merged
events. Record host/repository/PR, source-or-target role, branch, head, base,
complete file set and file modes, classification, lifecycle state, owner
disposition, authority evidence, and last validation. The ledger grants no
authority. Before implementation entry, successful phase completion, sync, or
success reporting, every entry must be merged, explicitly retained by
PR-specific boundary-scoped owner direction, or explicitly closed. Retained is
not an unmerged-plan exception. Stop reports may list unresolved entries but
cannot claim success.

For the only combined content-and-merge paths, bind and revalidate the full
repository/PR/role/branch/head/base/eligibility/file-set/file-mode/classification
snapshot before the prompt and immediately after approval. Exact plan-only and
verified non-operational documentation-only PRs use their named prompts;
plan-plus-implementation, operational or uncertain documentation, profiles, skills,
agents, local guidance, policy, workflow, configuration, generated, renamed,
symlinked, executable, and code content use ordinary merge control. Drift
requires fresh approval.

## Plan-first workflow

Follow the core's [combined topology and boundary rules](../SKILL.md#delivery-topology).
When publication is permitted, execute the combined L2/L3 workflow in this
order:

1. Verify the approved plan identity, declared scope, phases, gates, manual
   boundaries, and delivery topology; keep the plan-only PR free of
   implementation, generated output, and downstream sync.
2. Confirm the initial diff is exactly the approved plan, the published plan
   matches its approved revision, and entry gates pass. Record the plan path,
   current plan-only HEAD, and immutable P0.
3. After visible PR verification, obtain the named Implementation authorization
   and apply its permitted in-envelope actions only.
4. Before each delivery, perform the core's continuing P0 check and inspect
   cumulative scope, topology, and evidence. Request approval only for the
   exact visible plan identity and current binding.
5. On plan, target, authorization, scope, topology, manual-boundary, or other
   core-defined drift, stop and obtain the required rebinding or renewal.

The core's [separate-boundary and L1 rules](../SKILL.md#delivery-topology) apply
outside the combined topology. If publication is prohibited, retain all local
scope and acceptance gates and report why no PR exists.

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests, formatting,
lint, triggered review, and active-local-session list. Stage only explicit
intended paths; never broad stage. Use `Phase-gate: auto (L1)`,
`Phase-gate: bundle (P<N>)`, or `Phase-gate: manual`; present one exact pending
delivery action and apply only the current checkpoint's bounded authorization.

For the combined topology, publish each green phase on the same draft PR. A
manual next-phase gate blocks only entry to the next phase, not this green
phase's publication; a topology deviation or stop rule blocks the affected
phase. Keep the plan-only PR in draft while verifying it and mark it ready only
after validation and review eligibility pass.

Update the PR-lifecycle ledger for each publication transition, including
external PR changes observed during the gate. Do not enter implementation,
complete the phase, or sync downstream while an entry is unresolved.

Follow the core's [publication and merge boundaries](../SKILL.md#contextual-approval):
bind the repository, PR, head, base, and check/review eligibility snapshot;
request the exact final merge approval; revalidate immediately; and merge only
with the approved-head/expected-head precondition. If a bound head, base, or
eligibility changes or regresses, invalidate the approval and request fresh
approval; revalidation cannot revive it. Source authority never transfers to
target delivery or downstream sync. Separately explicit target authority must
name the target repository, exact worktree and branch, exact paths or diff, and
permitted target actions; it authorizes only those actions and never a merge.
For the L1 fast path, record qualification and acceptance evidence in the
normal PR and wait for separate final merge approval.

## Minimum phase report

Use [templates.md](templates.md) immediately before reporting. Include plan and
approval state, scope/changed files, controller/model and route evidence, risk
and difficulty, sessions/retries/wait status, reviewer state, red/green and
validation evidence, usage warnings, deviations, publication state, and gate.
