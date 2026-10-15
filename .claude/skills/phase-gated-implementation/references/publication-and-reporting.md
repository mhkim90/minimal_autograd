# Publication and Reporting

Read this before a plan/approval decision, commit, push, PR action, or phase
report. Return publication evidence to the core; do not override its gate.

## Context-bound authorization

The core's [authorization and P0 rules](../SKILL.md#authorization-and-p0-verification)
define authority, contextual approval, protected boundaries, and the complete
invalidation set. This reference supplies the publication order below; it does
not add authority or override the core gate.

### Exact publication-transfer binding

Immediately before a publication transfer, bind the source repository, named
configured GitHub destination, base, branch, exact approved payload paths/diff
or manifest, minimum PR metadata, and permitted actions in one visible prompt;
read [templates.md](templates.md) immediately before presenting that prompt.
An immediately following `approve`/`approved` authorizes only the unchanged
enumerated transfer, without a duplicate disclosure prompt, and never supplies
an omitted field. A changed repository, destination, base, branch, payload,
manifest, metadata, or eligibility; an unconfigured service, out-of-scope
private material, destructive action, readiness, merge, an unbound later
action or phase, or host rejection stops the transfer and requires fresh exact
authority. Already-bound local successors or in-envelope same-PR work do not
require fresh approval merely because the phase advances; this never expands
the exact publication payload/actions or grants merge authority. Host policy
and safety enforcement are never bypassed.

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

Mark a PR ready only after validation and review eligibility pass. A prior
explicit binding (such as qualified bundle delivery) may already authorize
readiness. Otherwise, for an eligible draft, include readiness and merge in one
direct PR-specific request; do not infer readiness from a merge-only approval.
Readiness is not merge authority, and no automatic merge follows from plan
approval, draft state, or readiness.

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

For the optional early-published route, a still-active draft requires explicit
PR-specific retention wherever this gate demands a resolved disposition.
Include that boundary-scoped disposition and declared concrete resolution event
in the already-required named Implementation binding, not a separate approval
prompt; never infer retention from draft state or green phases. Retention does
not waive entry/P0 checks or authorize success or merge of an unmerged Plan PR.

If no task Plan PR exists, report unpublished local plan/P0 evidence and
PR `not-created`/`not-applicable`, without inventing a ledger entry. An existing
or duplicate task Plan PR is an actual touched PR and topology deviation:
record its identity and resolve its ledger disposition under the rules above,
applying the core's renewal rules before a route switch. Never report that PR
absent or infer implementation permission from retained/reopened status;
an unmerged Plan PR neither satisfies a merged-plan gate nor supplies
implementation permission.

For the only combined content-and-merge paths, bind and revalidate the full
repository/PR/role/branch/head/base/eligibility/file-set/file-mode/classification
snapshot before the prompt and immediately after approval. Exact plan-only and
verified non-operational documentation-only PRs use their named prompts;
plan-plus-implementation, operational or uncertain documentation, profiles, skills,
agents, local guidance, policy, workflow, configuration, generated, renamed,
symlinked, executable, and code content use ordinary merge control. Drift
requires fresh approval.

## Plan-first workflow

Apply the core's [authorization and P0 rules](../SKILL.md#authorization-and-p0-verification)
for route eligibility, protected boundaries, checkpoint identity, renewal, and
exceptions. Record the rationale in the existing binding/report fields; use
[templates.md](templates.md) for the complete authorization fields.

**Local plan and one eventual PR**

1. Complete the provenance-correct planning review for the exact revision.
   Obtain the complete named Local Plan-and-Implementation binding, including
   any authorized external route destinations and source/diff payload.
2. Record the approved plan, create plan-only local P0, and verify the core's
   entry gate before implementation. Save the durable controller evidence
   location and original checkpoint record in the handoff.
3. Perform bounded implementation, tests, required independent review, explicit
   staging, and validated controller-local commits within that envelope. No initial draft,
   published-plan verification, per-phase push, or per-file/per-test/per-review/
   per-phase owner confirmation is required for already authorized actions.
4. After the local gates pass, inspect cumulative changes and original-P0
   descent, then prepare one PR containing the plan, implementation, and tests
   when applicable. Obtain the separate exact publication-transfer binding
   above; local approval is not GitHub publication authority.

**Optional early-published same-PR route**

1. Verify the approved plan identity, scope, phases, gates, manual boundaries,
   and topology. Obtain named Plan-and-Draft authorization for only that
   plan's commit/push/draft PR; keep the initial plan-only diff free of
   implementation, generated output, and downstream sync.
2. Verify the visible PR's initial diff is exactly the approved plan, its
   published revision matches approval, and entry gates pass. Record the plan
   path, current plan-only HEAD, and immutable published P0.
3. After visible PR verification, obtain named Implementation authorization
   for validated in-envelope same-PR commits/pushes under the core's entry and
   authority limits, including explicit PR-specific draft retention and its
   declared resolution event when required by the ledger gate.
4. Before each delivery, perform the continuing P0 check and inspect cumulative
   scope, topology, and evidence under the current exact binding.

For amendments, base refresh, route switches, or other drift, apply the linked
core review/renewal and lineage rules before affected work. Other topologies
use the core's applicable merged-plan/exception rules and
[delivery boundaries](../SKILL.md#delivery-topology).
If publication is prohibited, retain local scope and acceptance gates and
report actual local and PR state.

## Per-phase publication

Before a commit, inspect the diff, scope, red/green evidence, tests, formatting,
lint, triggered review, and active-local-session list. Stage only explicit
intended paths; never broad stage. Use `Phase-gate: auto (L1)`,
`Phase-gate: bundle (P<N>)`, or `Phase-gate: manual`; present one exact pending
delivery action and apply only the current checkpoint's bounded authorization.

For the local route, follow the [local workflow](#plan-first-workflow) for
validated local phases and eventual authorized delivery; no phase push is
required. Apply the core's
[classification rules](../SKILL.md#non-authoritative-ledger-and-classifications)
to the plan-plus-implementation PR.

For the early-published combined topology, publish each green phase on the same
draft PR under its existing authorization, with explicit PR-specific retention
from the named Implementation binding when the ledger gate requires it. A
manual next-phase gate blocks
only entry to the next phase, not authorized publication of this green phase;
a topology deviation or stop rule blocks the affected phase. Keep the initial
plan-only PR in draft while verifying it. Mark a PR ready only after
validation, review eligibility, and explicit readiness authority.

Update the PR-lifecycle ledger for each publication transition, including
external PR changes observed during the gate. Do not enter implementation,
complete the phase, or sync downstream while an entry is unresolved.
After authorized publication, verify remote branch/head, PR state, base, and
payload against the binding; stop on rejection, drift, or uncertain outcome.

Follow the core's [publication and merge boundaries](../SKILL.md#contextual-approval):
bind the repository, PR, draft state, head, base, and check/review eligibility
snapshot; request the exact ready-and-merge or merge-only approval as
applicable; revalidate immediately and again after readiness; merge only with
the approved-head/expected-head precondition. Except for the authorized draft-
to-ready transition, any bound state/head/base drift or eligibility regression
invalidates approval and requires fresh authority; revalidation cannot revive
it. Source authority never transfers to target delivery or downstream sync.
Separately explicit target authority must
name the target repository, exact worktree and branch, exact paths or diff, and
permitted target actions; it authorizes only those actions and never a merge.
For the L1 fast path, record qualification and acceptance evidence in the
normal PR and wait for the applicable final PR-specific approval.

Authorized branch/worktree housekeeping alone needs no PR. Separate code
optimization warrants a PR only when independently justified and authorized;
do not invent implementation work merely to accompany cleanup.

### Post-action successor evaluation

After a successful authorized action or verified merge, update the ledger,
refresh continuity and live state, and evaluate the ordered successor under the
core's [automatic successor transition](../SKILL.md#automatic-successor-transition)
guard. Only an eligible uniquely declared `advance: auto` successor with its
existing named worktree may enter automatic preflight or bounded in-scope work;
do not create a worktree merely to continue, and delegate only where the
approved route and existing authority already permit it. This transition does
not change the scopes of plan approval, delivery approval, administrative
readiness, or bound merge approval.

An eligible successor may include already-authorized validated local commits
by the controller, never by delegates. If it requires an unbound local action,
push, PR creation/update, readiness change, merge, release, deployment,
downstream or cross-repository sync, stop before that action and present one
precise request for its independently required authority. Automatic advance
never supplies new authority or a new worktree. Record the successor, guard evidence, and transition
disposition in the phase report.

## Minimum phase report

Use [templates.md](templates.md#phase-report) for a concise owner-facing report:
actual local/publication state, changed files, red/green and validation,
candidate attempts, route, reviewer/resources, time, blockers, and gate/successor.
Point to durable evidence for applicable detailed bindings rather than requiring
a giant chat report. Pending/unknown resources block success; a hard-expiry stop
report must account for them, not claim an empty ledger.

## Durable phase evidence

Keep the following applicable fields discoverable in controller evidence/handoff
outside the immutable plan. `Not-applicable`, `not-created`, and `unknown` are
honest states, not permission. These records grant no authority and do not add
mandatory token/price accounting. Use the
[delegate native-evidence contract](../../opencode-delegate/SKILL.md#native-evidence-and-diagnostics)
for provenance, bounded/redacted normalized diagnostics, and exact authorized
manual invocation/exit/readback evidence.

```text
Phase <N>: <complete | in progress | blocked>; local state: <commit or uncommitted state>
Publication: <local/unpublished | published | stopped before prohibited action>
Plan preflight: <artifact/revision, provenance-correct review, freshness, selected route and applicable manual-gate evidence | L1 fast path: behavior-preserving qualification and waiver>
Plan PR disposition: <merged | unmerged-exception | explicitly-retained | closed | not-created | not-applicable | unknown>; plan identity: <local path/revision plus P0 | title plus actual PR | unique commit title | named ingredients | not-applicable | unknown>; owner approval: <plan-scope/content/route evidence | L1 waiver | pending | invalidated | not-applicable | unknown>; controller binding: <plan path, provenance, approved revision/amendments, original and current bound base, original plan-only P0, durable non-authoritative evidence/handoff location; base-update review/renewal and authorized history-preserving action or none; current-head/expected-ancestry verification: verified | stale | ambiguous | not-applicable | unknown>
Local L2/L3 binding: <Local Plan-and-Implementation fields: repository, worktree, branch, base, exact plan revision, scope, topology, eligibility/boundary rationale and applicable gate/direction, gates, permitted local actions, configured external destinations/routes/roles and source/diff payload or none; status | not-applicable>; task Plan PRs: <none: not-created/not-applicable | existing/duplicate: actual identities and ledger links | unknown>
Early-published L2/L3 binding: <Plan-and-Draft fields: repository, worktree, branch, base, plan path/revision, exact plan diff, one draft PR; status | not-applicable>
Implementation authorization: <local: exact local binding and permitted actions, no GitHub publication authority | early-published: repository, PR, branch, immutable published P0, visible PR verification, permitted same-PR commits/pushes, explicit PR-specific retention disposition/declared resolution event when required by the ledger; status>
P0 verification: <local: original-base delta exactly approved plan artifact(s), no implementation already staged/present, approved revision/checkpoint verified before implementation | early-published: initial PR diff exactly approved plan, published revision matches, visible entry gates pass>; recorded first implementation HEAD: <P0 or named authorized inspected exception>; original P0/provenance/revision/amendments/expected descent and cumulative scope/topology/evidence: <verified | stale | ambiguous>
Publication-transfer binding: <separately bound repository, configured destination, base, branch, reviewed payload, minimum metadata, permitted actions; authority/status and verified remote state | not-authorized | not-created | not-applicable>
Unmerged-plan exception: <applicable original route and authorization; reason; implementation branch: <name or not-created/not-applicable/unknown>; implementation PR: <number/URL or not-created/not-applicable/unknown>; resolution event: <event or not-created/not-applicable/unknown>; resolution state: <resolved | pending | renewed | not-applicable | unknown>>; local route: <not an exception>
Implementation branch/PR: <branch and PR or not-created | not-applicable | unknown>; readiness: <draft | ready | not-created | not-applicable | unknown>; merge state: <merged | not-merged | not-created | not-applicable | unknown>
Delivery topology: <local one eventual plan-plus-implementation PR | early-published same-PR | separate Design/Plan route | direct L1 PR qualification>; implementation-PR count/current PR/included phases/phase-to-PR mapping: <actual state, including existing/duplicate task PRs>; split boundary/rationale: <evidence or none>; topology deviation: <none | existing/duplicate Plan PR | route switch | other; existing renewal evidence or blocked>; actual touched-PR ledger/dispositions: <resolved evidence or blockers>
Scope: <approved globs>; changed files: <list>
Controller: Claude Code; active model: <runtime evidence>
Safety risk level: <L1-L4>; implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <luna | justified sol-implementer | justified astra-implementer | astra-expert then implementer | controller-bound astra-orchestrator | explicit user configured default>
Implementation sessions / retries: <count> / <count>; route evidence: <requested agent, job/session IDs, bound/reported model, warnings>
Capability proof / STOP diagnostics: <actual tools/invocations/exits/Git/editor readback or terminal failure record per linked Boundary>
Expert escalation: <role, reason/question, why cheaper route is insufficient, requested/resolved model, effort, job/session, expected evidence, stop condition, or none>
Elapsed time: <per-job actual launch-to-terminal or runtime-to-date, controller work and owner wait separately; overall deadline>; checkpoints: <count>; wait-policy reviews / task-specific synthesis allowance: <none or list>
Active resources at gate: <none unresolved, or blocked/pending: each job/command ID, live/unknown status and result-retrieval state>
Reviewer trigger reason: <reason or none>; independent reviewer/state/verdict: <details or pending>
Red gate: <right failure then pass>; attempts: <used>/<cap>
Validation: <commands/results>; metrics: <values>
Plan deviations: <none or rationale>
Successor: <uniquely declared immediate successor and existing named worktree | not declared | ambiguous | none>
Successor evidence: <advance: auto declaration, scope/route/dependency/acceptance/topology, ancestry, blocker, and live-state/preflight evidence>
Transition disposition: <auto-advanced | waiting-for-explicit-delivery | waiting-for-manual-boundary | blocked>
Gate: <local validated; proceeding-within-approved-local-envelope | local validated; waiting-for-explicit-delivery | published; auto-proceeding | published; proceeding-within-approved-combined-topology | published; waiting-for-next-phase-approval | blocked>
```

Only when requested or required for a capability-failure record, add native
normalized per-job usage diagnostics when available, otherwise unavailable
with coverage limitations. They are observational, not total-session accounting
or a quality gate; do not fabricate totals, use `usage_mcp`, read private provider
paths, or introduce fixed token, price, or provider-private-path budgets.
