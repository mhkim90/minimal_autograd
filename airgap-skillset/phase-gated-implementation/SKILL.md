---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist in a single-model environment. Keep the controller as final publication gate, size ceremony by difficulty, and load policy references at their transitions.
---

# Phase-Gated Implementation (Single-Model / Closed Network)

The controlling agent (whichever harness is active — Claude Code, OpenCode,
or Copilot) controls scope, gates, publication, and stop/go; it does not
implement by default.

This variant assumes exactly one backend model (e.g. DeepSeek V4 Flash)
reachable through the local harness. There is no multi-model routing: the
open-network `opencode-delegate` / `codex-delegate` named-agent selection
(Luna / Sol-expert / Codex Terra) does not apply here and those skills are
intentionally excluded from this skillset. Do not reintroduce route or
bound-model evidence bookkeeping — it has no meaning when every role
resolves to the same weights.

## Invariants

- Publication is commit-first and PR-free: land each green phase as a
  commit; never open, update, mark ready, or merge a real PR — the closed
  side batches actual submission by hand. Instead produce a markdown
  delivery draft for the owner to approve. See
  [`references/publication.md`](references/publication.md).
- Every commit message and the delivery draft's title starts with the Jira
  ticket ID for this work (`TICKET-ID: ...`). Ask the owner for it before
  the first commit of an initiative if not already given — do not guess or
  omit it. A missing ticket ID is a stop condition (see the
  stop-conditions invariant below).
- Consolidate commits by activity, not by phase: default to one
  implementation commit, one test commit, and one fix commit per review
  round (fix-after-review-1, fix-after-review-2, ...) per initiative.
  Squash/amend intermediate phase commits into that shape before they
  accumulate — only ever local, not-yet-pushed commits. See
  [`references/publication.md`](references/publication.md).
- Use an exact approved plan, written as a reviewable artifact under the
  untracked `delivery/` ticket folder — never committed, since the closed
  side's git policy excludes plan files from the repository — for phased,
  non-trivial, or L2-L4 work. The plan declares scope, gates,
  risk/difficulty, wait policy, dependencies, manual boundaries, and
  delivery topology. Approval names its content hash; it authorizes
  implementation only. Material changes invalidate it.
- Owner approval is per-action: `approved` covers exactly the one concrete
  publication action named immediately before it, and expires on any
  target, scope, head, or action change.
- Keep delivery topology even without a real PR: it decides which phases
  may be batched into one delivery draft later and which must not. Split or
  batch per [`references/publication.md`](references/publication.md).
- An L1 fast path needs exact authorized scope, checks, and publication
  intent; it never waives staging, diff/acceptance evidence, or stop
  rules. It excludes policy, architecture, API, security,
  runtime/release configuration, dependencies, migrations, generated
  outputs, workflows, approval behavior, and operational documentation, and
  any new cross-repository rollout unless it uses an already-merged source
  revision and exact manifest. When uncertain, use the plan-first workflow.
- Before every phase, check dirty state and any pause markers, obtain
  minimum scope/gate/current evidence, run the smallest right-reason red
  gate, and set a three-attempt cap unless the plan says otherwise.
- Ceremony scales with difficulty (see Ceremony level below), not backend
  selection — there is only one backend.
- Validate the in-scope diff and green evidence before publication. Commit
  a green phase only with its trailer and explicit-path staging, after any
  triggered `isolated-review` or `airgap-export` escalation passes. A
  manual next-phase gate does not prevent the completed phase's commit.
- Stop and report on stale/missing preflight, a missing ticket ID, wrong/
  failed gates, scope or intent expansion, unverifiable correctness,
  unavailable evidence, an unresolved review or escalation blocker,
  maximum wait, unrelated blocking changes, or attempt-cap exhaustion. Do
  not weaken a gate to continue.
- Local continuity is discovery-only; no MCP memory service is assumed.
  Before resuming or crossing repositories, check current Git/plan/HANDOFF
  state first. If a local continuity note (e.g. `HANDOFF.md`) exists, read
  it only as a discovery lead and validate it against current evidence —
  it never grants approval, overrides a red/green gate, or replaces
  current evidence.
- No usage/token accounting. The closed-network backend runs locally with
  no meaningful budget constraint — do not add token, price, or usage
  gates or fields.

## Sequence

The Invariants above are unordered; follow this order per phase so a
single controller doesn't drop a step under load. This restores the old
draft's explicit step list — the difference is that each step now points
at the reference holding its detail instead of inlining it.

1. Preflight: read [`references/publication.md`](references/publication.md);
   confirm the ticket ID (ask the owner if not yet given this initiative),
   plan/audit currency, the plan artifact's approved content hash, dirty
   state, pause markers, approved scope, and ceremony tier.
2. Red gate: run the smallest right-reason failing check.
3. Implement per the ceremony tier.
4. Review/escalate per
   [`references/escalation.md`](references/escalation.md) when the tier
   requires it.
5. Gate and publish per
   [`references/publication.md`](references/publication.md).
6. Report the phase; wait for next-phase approval unless auto-continuation
   conditions are met.

## Ceremony level

- **Mechanical/economy**: repetitive, well-understood edits or routine
  commands, within the L1 fast path exclusions above. Implement directly.
  `isolated-review` is optional if automated gates (lint/type/test) are
  already green and sufficient on their own.
- **Standard**: locally understood non-trivial work. Implement, then run
  `isolated-review` before publish.
- **Difficult**: architecture-sensitive, unclear, numerical, or
  repeated-blocker work. Write a short approach note first (what you're
  about to try and why, one paragraph) before touching code, implement,
  then run `isolated-review` before publish.

## Router and lazy policy loading

Load the named reference immediately before its transition, never merely
on phase entry. References explain *how* and return evidence; this core
alone decides whether to route, pass a gate, publish, or start another
phase.

| Trigger | Destination |
| --- | --- |
| Active agent drafted substantive plan | `grilled-me` |
| User/repo/issue/external substantive author | `plan-audit` |
| Review before publish | `isolated-review` |
| Repeated blocker / high-risk step / unresolved review blocker | `airgap-export` |
| Transfer to a fresh session | `handoff` |

Read [`references/publication.md`](references/publication.md) before any
plan/approval decision, commit, push, delivery-draft action, or phase
report. Read [`references/escalation.md`](references/escalation.md)
immediately before triggering `isolated-review` or `airgap-export`.

Editing or summarizing externally authored material does not change its
origin.

## Gate

Before this gate, confirm the applicable reference(s) were actually read
this phase, not merely assumed — the router only names when to load them;
nothing else enforces it. Treat an unread required reference as a stop
condition, not something to skip under time pressure.

Inspect active local sessions, the contiguous commit suffix, and evidence
before the gate. A consolidated commit (see the commit-granularity
Invariant) may carry multiple `Phase-gate:` trailer lines, one per
folded-in phase, in original order — count trailer occurrences across the
suffix, not commits. Two consecutive `Phase-gate: auto` trailers force the
next L1 phase manual; bundle trailers do not count, and unexpected
interleaving is scope drift. Missing approved per-phase scope disables
automatic continuation. An L1 phase proceeds only when final L1, in scope,
right red failure then pass within cap, no deviation, and no
review/escalation blocker. A bundle phase proceeds only when declared,
green, topology-compatible, and free of a manual boundary or stop
condition. A new split boundary or validation/rollback mismatch is a
plan-topology deviation and blocks affected work pending renewed approval.
Report the current phase and whether the next one proceeds or waits.
