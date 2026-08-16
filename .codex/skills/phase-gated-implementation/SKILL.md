---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Codex as final gate, route implementation by difficulty, and load policy references at their transitions.
---

# Phase-Gated Implementation

Codex controls scope, gates, publication, and stop/go; do not implement by
default. Record the active controller model from invocation or runtime
metadata—Terra is intended, not evidence about this session.

Keep safety risk (L1–L4) separate from implementation difficulty. Use the
configured OpenCode default for mechanical/economy work, `agent="luna"` for
standard work, and a bounded `agent="sol-expert"` preflight or breakthrough
only for difficult work before Luna implements. Preserve an explicitly approved
whole-phase `agent="sol"` route without separately calling Luna or Terra.
Never silently replace a selected route or raw-model override.

## Invariants

- Use an exact approved plan and plan-only draft PR for phased, non-trivial, or
  L2–L4 work. The plan declares scope, gates, risk/difficulty/routes, wait
  policy, dependencies, manual boundaries, and delivery topology. Delivery
  topology names implementation-PR count, phase-to-PR mapping, and each split
  boundary. Approval names its SHA;
  it is never merge approval. Material changes invalidate it.
- Default to one implementation PR per coherent, independently releasable or
  revertible deliverable. One PR may contain multiple phases, but never
  replaces each phase's scope, red/green evidence, acceptance gate, owner
  decision, or triggered review.
- Split at a material security, API/compatibility, release, migration,
  rollback, dependency, ownership, required-owner-decision, independent-review,
  or validation-environment boundary. Bundle adjacent phases only when their
  approved scope, ownership, validation, and rollback behavior are compatible.
  A split rationale describes that boundary; it never makes unsafe bundling
  acceptable.
- An L1 fast path needs exact authorized scope, checks, and publication intent;
  it never waives explicit staging, diff/acceptance evidence, stop rules, or
  separate merge approval. An initiative bundle may continue only inside its
  approved envelope and stops at declared manual boundaries.
- Before every phase, check dirty state and both pause files, obtain minimum
  scope/gate/current evidence, run the smallest right-reason red gate, and set
  a three-attempt cap unless the plan says otherwise.
- Validate the in-scope diff and green evidence before publication. Publish a
  green phase only with its trailer and explicit-path staging. Keep a plan PR
  draft until final validation and triggered review pass. A manual next-phase
  gate does not prevent publication of the completed phase.
- Stop and report on stale/missing preflight, wrong/failed gates, scope or
  intent expansion, unverifiable correctness, route contradiction, unavailable
  evidence, review blocker, maximum wait, unrelated blocking changes, or
  attempt-cap exhaustion. Do not weaken a gate to continue.
- Memory is discovery-only: read the applicable continuity record before
  resuming or crossing repositories, validate it against live source, then
  save decisions, blockers, and handoff evidence. It never supplies approval.

## Router and lazy policy loading

Load the named reference immediately before its transition, never merely on
phase entry. References explain *how* and return evidence; this core alone
decides whether to route, pass a gate, publish, or start another phase.

| Trigger | Codex destination | Claude destination |
| --- | --- | --- |
| Active agent drafted substantive plan | `grilled-me` | `grilled-me` |
| User/repo/issue/external substantive author | `plan-audit` | `plan-audit` |
| Implementation delegation | `opencode-delegate` + [`references/delegation.md`](references/delegation.md) | `opencode-delegate` + `references/delegation.md` |
| Triggered independent read-only review | `claude-delegate` + [`references/review-and-wait-policy.md`](references/review-and-wait-policy.md) | `codex-delegate` + `references/review-and-wait-policy.md` |
| Resume/cross-repository work | `memory-continuity` | `memory-continuity` |
| Transfer to a fresh session | `handoff` | `handoff` |

Editing or summarizing externally authored material does not change its origin.
Read [`references/publication-and-reporting.md`](references/publication-and-reporting.md)
before any plan/approval decision, commit, push, PR action, or report; read
[`references/templates.md`](references/templates.md) immediately before using
a prompt, capsule, or phase report.

## Gate

Inspect active local sessions, the contiguous commit suffix, and evidence
before the gate. Two consecutive `Phase-gate: auto` trailers force the next
L1 phase manual; bundle trailers do not count, and unexpected interleaving is
scope drift. Missing approved per-phase scope disables automatic continuation.
An L1 phase proceeds only when final L1, in scope, right red failure then pass
within cap, no deviation, and no review blocker. A bundle phase proceeds only
when declared, green, topology-compatible, and free of a manual boundary or
stop condition. A new split boundary or validation/rollback mismatch is a
plan-topology deviation and blocks affected work pending renewed approval.
Report the current phase and whether the next one proceeds or waits.
