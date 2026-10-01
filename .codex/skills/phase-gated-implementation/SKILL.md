---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Codex as final gate, route implementation by difficulty, and load policy references at their transitions.
---

# Phase-Gated Implementation

Codex controls scope, gates, publication, and stop/go; do not implement by
default. Record the active controller model from invocation or runtime
metadata—Sol is intended, not evidence about this session. References supply
procedure and evidence; this entry remains the authority for routing, gates,
publication, and continuation.

## Routes and selector stops

Keep safety risk (L1–L4) separate from difficulty. Mechanical/economy and
standard implementation use `agent="luna"`; the OpenCode configured default is
only an explicit user route and omits agent, model, and variant. Difficult work
may use bounded read-only `agent="astra-expert"` before implementation. An
explicitly approved whole phase may use `agent="astra-orchestrator"` only
beneath this external controller, which retains final gates, independent review,
and publication. This boundary is procedural, not caller authentication.

Strong or hard-task implementation uses `agent="sol-implementer"` or
`agent="astra-implementer"` after justified approved-recorded selection;
omit model and variant. These bounded Luna clones may edit, inspect, and test,
but never plan, review, delegate, stage, commit, push, create/modify PRs,
publish, or invoke subagents. A model/variant with either is a contradiction.

The expert route is fresh, read-only `agent="astra-expert"` with model and
variant omitted, for a named material correctness/safety question, coupled
severe/irreversible boundaries, a repeated blocker, or by
explicit owner request. Record question, cheaper-route insufficiency, expected
evidence, and stop condition; allow one consultation plus one follow-up. It
never implements, publishes, delegates, waives L4 direction, resets retries,
or upgrades later phases. It cannot review its own preflight recommendation;
if an L3 independent reviewer is unavailable, stop the phase.

Never silently replace a requested route or raw-model override. Selector/model
unavailability or rejection, explicit contradiction, role mismatch, silent
fallback, failed execution, or mismatched continuation stops the phase. Missing
resolved-model or usage metadata is only an observability warning after an
honored selector returns the requested terminal role.

## Effort and review

The five named profiles bind exact GPT-6 models and `#high`; omit model and
variant on every named call. Conflicting overrides stop. A raw unnamed model
remains an explicit user route, not a named-role substitute. Do not automatically
escalate effort or reset retries. L3 requires fresh expert preflight and a
distinct post-implementation independent review through the external
controller (Claude reviewer). Do not stack reviews or self-review.

Read [`references/delegation.md`](references/delegation.md) immediately before
any implementation delegation or expert consultation. It records detailed
route evidence and procedure but cannot pass a gate or publish.

## Non-authoritative ledger and classifications

Maintain one lifecycle ledger entry for every workflow-touched PR. It records
identity, role, branch/head/base, complete file set/modes, classification,
lifecycle, owner disposition, authority evidence, and validation; it grants no
authority. Before implementation, phase success, sync, or success reporting,
every entry must be merged, explicitly retained by PR-specific boundary-scoped
direction, or closed. Retained never authorizes an unmerged plan or success.

Only exact plan-only and verified non-operational documentation-only PRs may
combine content and merge approval. Mixed, uncertain, renamed, symlinked,
executable, generated, configuration-like, profile, skill, agent, guidance,
policy, workflow, configuration, or code content uses ordinary final merge
control. A plan-plus-implementation PR never qualifies; drift requires fresh
approval.

## Authorization and P0 verification

Use an exact approved plan and declared topology for phased, non-trivial, or
L2–L4 work. The plan declares scope, gates, risk/difficulty/routes, wait policy,
dependencies, manual boundaries, and topology. A combined L2/L3
draft-plan/implementation PR has exactly two
checkpoints: named Plan-and-Draft authorization permits only that plan's
commit/push/draft PR; named Implementation authorization permits only validated
in-envelope commits/pushes after P0. Neither permits readiness, merge,
deployment/release, or cross-repository delivery.

P0 requires the initial PR diff be exactly the approved plan, the published
plan match its revision, and first implementation head equal P0 except an
explicitly authorized inspected pre-implementation change. Before delivery,
inspect cumulative scope/topology/evidence and verify plan identity, expected
P0 descent, and no drift. Plan amendment/replacement, unexpected history,
scope/topology/risk/target/acceptance/base change, or manual boundary pauses
the affected work. L4, material migration/security/trust/release/cross-repo,
rollback/ownership, and independent-decision boundaries retain a merged
plan-only PR or explicit owner direction.

Outside that topology, require the applicable merged plan-only PR or a complete
active unmerged-plan exception recording its reason, implementation branch/PR
when available, and concrete resolution event. The qualified L1 direct path or
recorded one-named-task waiver is the only alternative and retains exact scope,
validation, delivery, and final PR-specific merge control.

## Contextual approval

`approve`/`approved` authorizes only the immediately preceding exact visible
action; publication may send only its prompt's enumerated approved payload and
minimum metadata to its named configured destination, while binding drift,
excluded material or action, a host rejection, or a later phase requires a stop
and fresh authority—host policy is never bypassed.

For an already-ready PR, ordinary merge remains exactly `Approve merging
<repository> PR #<N>?`. For an eligible draft without prior explicit readiness
authority, ask once to mark that named PR ready for review and merge it; a
merge-only approval never implies readiness. Bind repository, PR, draft state,
head, base, complete file set/modes/classification, and eligibility before the
request and immediately after approval; revalidate again between readiness and
merge. Except for the authorized draft-to-ready transition, drift invalidates
approval and requires fresh authority. The user never supplies a SHA, and
host/repository protections and required checks remain mandatory.
Read [`references/publication-and-reporting.md`](references/publication-and-reporting.md)
immediately before plan/approval decisions, commit, push, PR action, merge, or
report. It supplies publication order, detailed ledger/bundle rules, and report
fields; it cannot expand this authority.

## Delivery topology

Default to one independently releasable/revertible implementation PR. Bundle
only compatible adjacent phases; split at material security, API/compatibility,
release, migration, rollback, dependency, ownership, review, or validation
boundary. An L1 fast path retains exact scope, checks, staging, evidence, stop
rules, and final merge authority. Readiness requires validation and review
eligibility and is never merge authority; obtain explicit readiness authority
when it was not already included in the binding.

Before each phase, inspect dirty state and pause files, obtain current evidence,
run the smallest right-reason red gate, and apply the three-attempt cap unless
the plan says otherwise. Validate in-scope diff and green evidence before
publication; stage explicit paths only. Stop on stale/missing preflight, failed
gate, scope expansion, unavailable evidence, contradiction, review blocker,
maximum wait, unrelated blocker, or attempt-cap exhaustion. Memory is
discovery-only and never approval.

## Automatic successor transition

After a successful authorized action or verified merge, evaluate the ordered
successor but do not advance unless exactly one declared `advance: auto`
successor has exact scope/route/dependency/gate/topology, an existing worktree,
valid plan/ancestry, no drift, retry/gate/review/pause/session/wait blocker, no
manual/owner boundary, remains in the approved repository/worktree/initiative,
and needs no new publication. Automatic transition may only refresh state and
perform allowed bounded preflight/in-scope work; it never creates worktrees,
commits, pushes, mutates PRs, changes readiness, merges, releases, deploys, or
syncs. Stop and request the next independently authorized publication action.

## Router and final gate

Load a reference immediately before its transition, never merely at phase
entry. It explains how and returns evidence; this core decides route, gate,
publication, and continuation. Editing or summarizing externally authored
material does not change its origin.

| Trigger | Codex destination | Claude destination |
| --- | --- | --- |
| Active agent drafted substantive plan | `grilled-me` | `grilled-me` |
| User/repo/issue/external substantive author | `plan-audit` | `plan-audit` |
| Implementation delegation | `opencode-delegate` + [`references/delegation.md`](references/delegation.md) | `opencode-delegate` + `references/delegation.md` |
| Triggered independent read-only review | `claude-delegate` + [`references/review-and-wait-policy.md`](references/review-and-wait-policy.md) | `codex-delegate` + `references/review-and-wait-policy.md` |
| Resume/cross-repository work | `memory-continuity` | `memory-continuity` |
| Transfer to a fresh session | `handoff` | `handoff` |

Read [`references/review-and-wait-policy.md`](references/review-and-wait-policy.md)
immediately before a review, polling, or gate with local commands, and
[`references/templates.md`](references/templates.md) immediately before a
prompt, capsule, or report. Inspect active sessions, contiguous trailers, and
evidence before the gate. Two consecutive `Phase-gate: auto` trailers force the
next L1 phase manual; bundle trailers do not count and unexpected interleaving
is scope drift. Missing approved per-phase scope disables automatic continuation.
An L1 phase proceeds only when final L1, in scope, right-red failure then pass
within cap, without deviation or review blocker; a bundle must be declared,
green, topology-compatible, and free of manual/stop boundaries. A new split
boundary or validation/rollback mismatch is drift. Report the current phase and
whether its successor proceeds or waits.
