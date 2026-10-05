# Templates

Read this immediately before sending a delegate prompt, Astra-expert capsule, or
phase report. Fill every applicable field; return the resulting evidence to the
core.

For OpenCode implementation, apply the [actual-delegate capability gate](../../opencode-delegate/SKILL.md#boundary)
and its scoped proof/STOP diagnostics; read-only experts need no editor gate.

## Implementation prompt

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <luna | justified sol-implementer | justified astra-implementer | astra-expert then implementer | controller-bound astra-orchestrator | explicit user configured default>; omit named model/variant
Expert escalation evidence: <role or none>; escalation reason/question: <...>; why cheaper route is insufficient: <...>; requested/resolved model: <...>; effort: <...>; job/session: <...>; expected evidence: <...>; stop condition: <...>
Model binding: <named GPT-6 #high profile or explicit unnamed route; reported match/warnings>
Capability proof / STOP diagnostics: <actual exposed capabilities; authorized Git/editor proof with tools/invocations/exits/readback, or terminal failure record per linked Boundary>
Elapsed-time checkpoint / final-synthesis grace (full Astra expert only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; do not commit or edit outside scope
- immediate capability STOP overrides ordinary retries, including midphase
- after capability gate passes, max attempts: <approved cap>; stop after two ordinary same-blocker failures
- use a materially revised approach before any third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, route,
and capability proof or STOP diagnostics under the linked Boundary
```

## Local Plan-and-Implementation authorization checkpoint

For the local route, apply the core's
[authorization and P0 rules](../SKILL.md#authorization-and-p0-verification)
after provenance-correct planning review. Resolve fields internally; do not
request a user-supplied SHA.

```text
Authorize Local Plan-and-Implementation: repository=<repository>;
exact worktree=<worktree>; branch=<branch>; recorded base=<base and revision>;
approved plan=<path, exact revision, authorship provenance, planning-review
evidence>; scope=<exact local paths/actions>; topology=<one eventual
plan-plus-implementation PR, included phases and split boundaries>;
eligibility/boundary rationale=<ordinary eligibility | important early
API/architecture decision | material boundary; evidence and applicable merged
Design/Plan gate or exact scoped owner direction>;
gates=<risk/difficulty/routes, tests, required independent review, wait policy,
acceptance, dependencies, and manual boundaries>.
External route binding=<configured delegate/reviewer destinations, authorized
implementation/review routes and roles, necessary in-scope read-only source/diff
payload for each, or none>.
Permitted local actions and bound route use only: record the approved plan,
create its immutable plan-only local P0, bounded implementation/tests/required
review, explicit staging, and validated in-envelope local commits.
P0 entry gate=<original-base delta exactly approved plan artifact(s), no
implementation already staged or present; approved revision/checkpoint verified
before implementation; first implementation HEAD recorded as P0 or an existing
explicitly authorized, inspected pre-implementation exception>.
P0 record after creation=<durable non-authoritative controller evidence/handoff
location outside the immutable plan; original P0, original base, approved
revision/provenance, entry and ancestry evidence>.
Not permitted: push, PR creation/update, readiness, merge,
release/deployment, downstream or cross-repository sync, unbound disclosures,
or out-of-envelope work.
```

An immediately following `approve`/`approved` binds only this unchanged local
envelope. Use the linked core rules for P0 preservation and renewal. After
local implementation, tests, and required review pass, use the separate
transfer binding below:

```text
Authorize one-PR publication transfer: source repository=<repository>;
configured GitHub destination=<configured host/owner/repository>;
exact worktree=<worktree>; branch=<branch>; base=<base>;
approved payload paths or diff/manifest=<reviewed plan plus implementation and
tests when applicable; exact paths/diff or manifest>;
minimum PR metadata=<title, draft status, and only enumerated body/labels>;
allowed actions only=<push the named validated branch and create/update the
one named draft PR>; local P0/cumulative scope/topology/evidence/ancestry
verification=<current controller evidence>.
Not permitted: readiness, merge, release/deployment, target sync,
cross-repository delivery, or any other PR.
```

This is mixed operational content, not plan-only or documentation-only.
Revalidate the exact transfer binding and verify remote state after authorized
publication; retain the ordinary final ready-and-merge or merge-only prompt.

## Plan-and-Draft authorization checkpoint

For the optional early-published same-PR L2/L3 route, present this exact visible
publication-transfer action only after required planning review and the plan's
red gate. It must enumerate the source repository, configured GitHub
destination, base, branch, approved payload paths or diff/manifest, minimum PR
metadata, and allowed actions. The controller resolves the fields; do not
request a user-supplied SHA. This is not required for the local route.

```text
Authorize Plan-and-Draft publication transfer: source repository=<repository>;
configured GitHub destination=<configured host/owner/repository>; exact
worktree=<worktree>; branch=<branch>; base=<base>; approved payload paths or
diff/manifest=<plan path and approved revision plus exact approved diff paths,
diff, or manifest>; minimum PR metadata=<title, draft status, and only the
enumerated body/labels>; allowed actions only=<commit the named plan, push the
named branch, and create the named draft PR>.
Not permitted: implementation, readiness, merge, deployment/release,
cross-repository delivery, target sync, or any other PR.
```

This checkpoint is not implementation authority. An immediately following bare
`approve` or `approved` authorizes only the unchanged transfer enumerated above;
do not ask a duplicate disclosure prompt, and never infer a missing field or
authorize a changed destination, payload, or action.

## Visible-draft Implementation authorization checkpoint

For the early-published same-PR route, after visible PR verification present
this exact named action. P0 is the controller-recorded immutable published
plan checkpoint; do not ask the user to provide a SHA. The local route uses
its Local Plan-and-Implementation binding instead.

```text
Authorize Implementation for repository=<repository>; PR=<number/URL>;
branch=<branch>; immutable plan checkpoint P0=<controller-recorded P0>.
Visible PR verification=<initial diff is exactly the approved plan;
published plan matches approved revision; entry gates pass>.
First implementation head=<P0, or explicitly authorized and inspected
pre-implementation change>.
Permitted actions only: validated in-envelope commits and pushes to this same
draft PR. Not permitted: a new PR, readiness, merge, deployment/release,
cross-repository delivery, target sync, or any out-of-envelope change.
```

Expected in-envelope descendants of P0 remain valid only after cumulative
scope, topology, and evidence inspection. Plan amendment, unexpected history,
scope/topology or target change, material base-assumption change, or split
boundary pauses affected work for rebinding or renewal.

## Final merge authorization prompt

At the final request, record the repository, PR, draft state, head, base, and
relevant check/review eligibility internally. Fill those fields before asking
for the direct owner reply, but do not ask the user for a SHA. For an already-
ready PR (mark a draft ready first when earlier explicit authority permits it):

```text
Approve merging <repository> PR #<N>?
```

For an eligible draft without earlier explicit readiness authority, ask once:

```text
Approve marking <repository> PR #<N> ready for review and merging it?
```

The only valid reply is exactly `approved` or `approve`, and only for the
immediately preceding unambiguous single-PR request. Revalidate before the
readiness action, then again before expected-head merge; the authorized draft-
to-ready transition is expected, but other state drift, head/base drift, or
eligibility-regressing checks/reviews invalidate approval. Host rejection
stops the sequence. Neither prompt authorizes delivery, target-sync delivery,
or another PR; the merge-only prompt does not authorize readiness. One
independently authorized target PR may use either prompt under its own bound
snapshot, eligibility, and normal merge controls.

## Narrow combined content-and-merge prompts

Use these only for an exact plan-only PR or a verified non-operational
documentation-only PR. Before prompting and immediately after approval, bind
and revalidate repository, PR, source/target role, branch, head, base,
eligibility, complete file set, file modes, and classification. Any drift,
mixed or uncertain classification, operational documentation, generated
output, rename, symlink, executable mode, configuration-like documentation,
profiles, skills, agents, policy, workflows, configuration, code, or a plan-plus-
implementation PR falls back to the ordinary final merge prompt.

```text
Approve and merge the plan in <repository> PR #<N>?
```

```text
Approve and merge the documentation-only PR in <repository> PR #<N>?
```

For a qualifying draft without earlier readiness authority, use the matching
explicit variant:

```text
Approve the plan in <repository> PR #<N>, mark it ready for review, and merge it?
```

```text
Approve the documentation-only PR in <repository> PR #<N>, mark it ready for review, and merge it?
```

These variants bind all three actions to the same PR; qualification and drift
rules are unchanged.

The ledger entry must be resolved before successful completion; these prompts
never grant implementation, sync, or host-control bypass authority.

## Qualified mechanical skill-sync bundle prompts

Use these only after `skill-sync` has frozen and verified the complete
qualified-bundle record. They never authorize a non-mechanical sync.

```text
Authorize mechanical skill-sync bundle delivery: bundle=<stable ID>; source
revision and manifest digests=<frozen record>; targets=<repository/base and
evaluated revision/task-owned branch/worktree/expected copy-only diff for each>;
trusted validation=<commands>; exclusions=<paths and operations>.
Permitted only: isolated copy, trusted validation, explicit staging, commit,
push, draft-PR create/update, and administrative readiness for every listed
entry. Not permitted: variants/adaptation, force-push, settings/protection
change, auto-merge, bypass, retry, merge, release, deployment, or other targets.
```

After readiness and fresh eligibility collection, request the final snapshot:

```text
Approve merging mechanical skill-sync bundle <stable ID>: <repository / PR /
expected head / base plus evaluated revision / checks and reviews / merge method
for every eligible entry>?
```

`approve` is valid only as the direct response to this unchanged prompt. Before
each sequential expected-head merge, revalidate every listed binding. Stop on
any drift, failure, regression, or uncertain result; report partial completion
and require fresh authority for any remainder.

## Astra-expert capsule

```text
Bounded read-only consultation; do not implement, edit, publish, or delegate.
Scope: <approved scope>
Question: <difficult preflight question or repeated blocker>
Evidence: <compact diff/test evidence>
Answer from this capsule when possible. If inspection is needed, each batch
resolves one named decision using a relevant file range or narrow symbol; never
use repository-wide enumeration/search or whole-file reads when a range will do.
Use at most four inspection batches; if evidence remains insufficient, return
Stop and name the missing evidence. A follow-up starts a new session with a
refreshed compact capsule.
Return: findings; proposed approach; acceptance gate; stop/go.
```

## Phase report

```text
Phase <N>: <complete | in progress | blocked>; local state: <commit or uncommitted state>
Publication: <local/unpublished | published | stopped before prohibited action>
Plan preflight: <artifact/revision, provenance-correct review, freshness, selected route and applicable manual-gate evidence | L1 fast path: behavior-preserving qualification and waiver>
Plan PR disposition: <merged | unmerged-exception | explicitly-retained | closed | not-created | not-applicable | unknown>; plan identity: <local path/revision plus P0 | title plus actual PR | unique commit title | named ingredients | not-applicable | unknown>; owner approval: <plan-scope/content/route evidence | L1 waiver | pending | invalidated | not-applicable | unknown>; controller binding: <plan path, provenance, approved revision/amendments, original and current bound base, original plan-only P0, durable non-authoritative evidence/handoff location; base-update review/renewal and authorized history-preserving action or none; current-head/expected-ancestry verification: verified | stale | ambiguous | not-applicable | unknown>
Local L2/L3 binding: <Local Plan-and-Implementation fields: repository, worktree, branch, base, exact plan revision, scope, topology, eligibility/boundary rationale and applicable gate/direction, gates, permitted local actions, configured external destinations/routes/roles and source/diff payload or none; status | not-applicable>; task Plan PRs: <none: not-created/not-applicable | existing/duplicate: actual identities and ledger links | unknown>
Early-published L2/L3 binding: <Plan-and-Draft fields: repository, worktree, branch, base, plan path/revision, exact plan diff, one draft PR; status | not-applicable>
Implementation authorization: <local: exact local binding and permitted actions, no GitHub publication authority | early-published: repository, PR, branch, immutable published P0, visible PR verification, permitted same-PR commits/pushes; status>
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
Elapsed time: <per role>; checkpoints: <count>; wait-policy reviews / final-synthesis grace: <none or list>
Active local sessions at gate: <none or blocked: IDs/status>
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
