# Templates

Read this immediately before sending a delegate prompt, Sol-expert capsule, or
phase report. Fill every applicable field; return the resulting evidence to the
core.

## Implementation prompt

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <agent="luna" with omitted model/variant | named Luna xhigh/max exception | explicit user request: configured default | sol-expert then Luna>
Expert escalation evidence: <role or none>; escalation reason/question: <...>; why cheaper route is insufficient: <...>; requested/resolved model: <...>; effort: <...>; job/session: <...>; expected evidence: <...>; stop condition: <...>
Effort-variant evidence: <requested agent and variant; named bottleneck/evidence; why clarification/decomposition is insufficient; fixed acceptance gate; additional max evidence or Sol-expert recommendation; warnings>
Elapsed-time checkpoint / final-synthesis grace (full Sol only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; do not commit or edit outside scope
- max attempts: <approved cap>; stop after two same-blocker failures
- use a materially revised approach before any third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```

## Plan-and-Draft authorization checkpoint

For eligible L2/L3 work, present this exact named action only after the plan
has passed its red gate. The controller resolves the fields; do not request a
user-supplied SHA.

```text
Authorize Plan-and-Draft for repository=<repository>; exact worktree=<worktree>;
branch=<branch>; base=<base>; plan path/revision=<path and approved revision>;
exact plan diff=<controller-rendered exact approved plan diff>; one draft
PR=<named draft PR/title and target>.
Permitted actions only: commit the named plan, push the named branch, and
create the named draft PR. Not permitted: implementation, readiness, merge,
deployment/release, cross-repository delivery, target sync, or any other PR.
```

This checkpoint is not implementation authority. A bare `approve` or
`approved` reply is valid only for this immediately preceding exact prompt and
never supplies a missing field.

## Visible-draft Implementation authorization checkpoint

After visible PR verification, present this exact named action. P0 is the
controller-recorded immutable plan checkpoint; do not ask the user to provide a
SHA.

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

Marking ready is administrative after validation and review; do not request
owner approval for readiness. At merge-request time, the controller records the
repository, PR, head, base, and relevant check/review eligibility internally.
Fill those fields before requesting the direct owner reply, but do not ask the
user for a SHA:

```text
Approve merging <repository> PR #<N>?
```

The only valid reply is exactly `approved` or `approve`, and only for the
immediately preceding unambiguous single PR merge request. Any later head or
base change, or eligibility-regressing check/review change, invalidates it and
requires fresh approval. Do not use this prompt as authority for delivery,
readiness, or target-sync delivery, or for another PR. It may be used to merge
one independently authorized target PR; that target merge still requires its
own bound snapshot, eligibility, and normal merge controls.

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

## Sol-expert capsule

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
Phase <N> complete: <commit or uncommitted state>
Publication: <published or stopped before prohibited action>
Plan preflight: <artifact/revision, audit, freshness, manual gates | L1 fast path: behavior-preserving qualification and waiver | combined L2/L3 topology: phase envelope and manual boundaries>
Plan PR disposition: <merged | unmerged-exception | not-created | not-applicable | unknown>; plan identity: <title plus PR | unique commit title | named ingredients | not-applicable | unknown>; owner approval: <plan-scope/content/route evidence | L1 waiver | pending | invalidated | not-applicable | unknown>; controller binding: <plan path and internally resolved plan-only HEAD; current-head/ancestry verification: verified | stale | ambiguous | not-applicable | unknown>
Combined L2/L3 binding: <Plan-and-Draft authorization fields: repository, worktree, branch, base, plan path/revision, exact plan diff, one draft PR; status>
Implementation authorization: <repository, PR, branch, immutable P0; visible PR verification; permitted action; status>
P0 verification: <initial diff exactly approved plan; published revision matches; entry gates pass; first implementation head equals P0 or named inspected exception; cumulative scope/topology/evidence: verified | stale | ambiguous>
Unmerged-plan exception: <reason; implementation branch: <name or not-created/not-applicable/unknown>; implementation PR: <number/URL or not-created/not-applicable/unknown>; resolution event: <event or not-created/not-applicable/unknown>; resolution state: <resolved | pending | renewed | not-applicable | unknown>>
Implementation branch/PR: <branch and PR or not-created | not-applicable | unknown>; readiness: <draft | ready | not-created | not-applicable | unknown>; merge state: <merged | not-merged | not-created | not-applicable | unknown>
Delivery topology: <implementation-PR count; current PR; included phases; phase-to-PR mapping; split boundary/rationale | direct L1 PR qualification | none: publication prohibited>; topology deviation: <none or renewed-approval state>
Scope: <approved globs>; changed files: <list>
Controller: Claude Code; active model: <runtime evidence>
Safety risk level: <L1-L4>; implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <agent="luna" with omitted model/variant | named Luna xhigh/max exception | explicit user request: configured default | sol-expert then Luna | agent="sol">
Implementation sessions / retries: <count> / <count>; route evidence: <requested agent, job/session IDs, bound/reported model, warnings>
Expert escalation: <role, reason/question, why cheaper route is insufficient, requested/resolved model, effort, job/session, expected evidence, stop condition, or none>
Elapsed time: <per role>; checkpoints: <count>; wait-policy reviews / final-synthesis grace: <none or list>
Active local sessions at gate: <none or blocked: IDs/status>
Reviewer trigger reason: <reason or none>; independent reviewer/state/verdict: <details or pending>
Red gate: <right failure then pass>; attempts: <used>/<cap>
Validation: <commands/results>; metrics: <values>; usage: <accounting and completeness warnings, if available>
Plan deviations: <none or rationale>
Gate: <published; auto-proceeding | published; proceeding-within-approved-combined-topology | published; waiting-for-next-phase-approval | blocked>
```

Usage is observational evidence only; do not introduce fixed token, price, or
provider-private-path budgets or semantics.
