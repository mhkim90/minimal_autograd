# Templates

Read this immediately before sending a delegate prompt, Astra-expert capsule, or
phase report. Fill every applicable field; return the resulting evidence to the
core.

## Implementation prompt

Use the [canonical implementation prompt](../../opencode-delegate/references/prompt-template.md)
with the [actual-delegate gate and recovery contract](../../opencode-delegate/SKILL.md#boundary).
Fill the approved checkout/plan, route, scope, acceptance, candidate cap,
wait/deadline, and expressly permitted recovery fields; do not add unapproved
policy or arbitrary budgets. Read-only experts need no editor gate.

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
gates=<risk/difficulty/routes, tests, required independent review, task-specific
wait/synthesis allowance and overall deadline, acceptance, dependencies, and
manual boundaries>; recovery=<one controller-launched fresh same-role session
per phase after terminal/result/diff review, retaining caps/lineage, or none>.
External route binding=<configured delegate/reviewer destinations, authorized
implementation/review routes and roles, necessary in-scope read-only source/diff
payload for each, or none>.
Permitted local actions and bound route use only: record the approved plan,
create its immutable plan-only local P0, bounded implementation/tests/required
review, explicit staging, and validated in-envelope controller-local commits.
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
its Local Plan-and-Implementation binding instead. When the ledger gate
requires retaining the still-active draft, include explicit PR-specific owner
retention and its declared resolution event here, not in a separate prompt.
Retention does not waive entry/P0 checks or authorize success or merge of an
unmerged Plan PR.

```text
Authorize Implementation for repository=<repository>; PR=<number/URL>;
branch=<branch>; immutable plan checkpoint P0=<controller-recorded P0>.
Visible PR verification=<initial diff is exactly the approved plan;
published plan matches approved revision; entry gates pass>.
First implementation head=<P0, or explicitly authorized and inspected
pre-implementation change>.
PR-specific retention disposition=<when required by the ledger: explicitly
retain this named draft within the implementation boundary; declared concrete
resolution event; otherwise not-applicable>.
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
Inspect within declared wait/stop limits, not a universal batch ceiling; stop
when evidence is unavailable or further inspection no longer addresses the
question, naming missing evidence. A follow-up starts a new session with a
refreshed compact capsule.
Return: findings; proposed approach; acceptance gate; stop/go.
```

## Phase report

Keep chat brief; retain applicable detailed binding and evidence fields in
[durable phase evidence](publication-and-reporting.md#durable-phase-evidence).
Never omit a blocker or substitute a summary for the exact evidence.

```text
Phase <N>: <complete | in progress | blocked/pending>; local/publication state: <actual>
Scope/changes: <files>; red/green and validation: <commands/results>
Route/candidates: <role, job/session, used/cap, warnings, recovery lineage or none>
Review/resources: <verdict or pending; every unresolved job/command and result state>
Time: <per-job launch-to-terminal or runtime-to-date, controller/owner time,
overall deadline and wait expiry if applicable>
Evidence: <durable binding/plan/P0/ancestry, capability proof or narrow STOP>
Unresolved issues/deviations: <none or exact blockers>
Gate/successor: <controller verdict; proceeding within existing local authority,
waiting for unbound action/manual boundary, or blocked>
```
