---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Claude Code as final publication gate, route implementation by difficulty, and load policy references at their transitions.
---

# Phase-Gated Implementation

Claude Code controls scope, gates, publication, and stop/go; do not implement
by default. Record the active controller model from runtime—Sonnet is intended,
not evidence about this session. The `codex-delegate` contract is read-only;
interactive Claude Code only publishes.

## Usage correlation

When `usage_mcp` is configured, load
[`references/usage-correlation.md`](references/usage-correlation.md)
immediately before starting, handling terminal provider work, or closing a
correlated workflow. Do not infer or reconstruct a missing controller identity.

Keep safety risk (L1–L4) separate from implementation difficulty. Use
`agent="luna"` for mechanical/economy and standard work, and use OpenCode's
configured default only when the user explicitly requests that route, omitting
`agent`, `model`, and `variant`. Use a bounded `agent="sol-expert"` preflight or
breakthrough only for difficult work before Luna implements. Preserve an
explicitly approved whole-phase `agent="sol"` route without separately calling
Luna or Terra. Never silently replace a selected route or raw-model override.

## Invariants

- Use an exact approved plan and, where an independent plan gate is required,
  a verified plan-only PR for phased, non-trivial, or L2–L4 work. A verified
  plan-only PR contains no source-policy implementation,
  executable/configuration/runtime change, generated output, or downstream
  sync. Draft is a GitHub PR state, not a separate delivery artifact. The plan
  declares scope, gates, risk/difficulty/routes, wait policy, dependencies,
  manual boundaries, and delivery topology. Delivery topology names
  implementation-PR count, phase-to-PR mapping, and each split boundary.
  Owner approval of the current verified plan-only head authorizes marking it
  ready and merging it after required checks and reviews pass; no extra merge
  approval is required. This does not authorize automatic merging of an
  implementation PR merely because it is draft. Adding implementation content
  or materially changing scope, risk, affected files or repositories,
  acceptance criteria, rollout, or sync invalidates the plan-only authorization
  envelope. Editorial clarification is not a material scope change, but every
  changed PR head—including editorial-only changes—must be verified and
  owner-approved before ready or merge. Protections, required checks, and
  reviewer rules are never bypassed.
- A direct owner reply containing exactly `approved` or `approve` is contextual
  authorization for only one immediately preceding, concrete pending
  publication action. The prompt must identify the exact repository/worktree,
  branch and PR when applicable, verified head or exact changed-path scope,
  and whether the action is delivery, readiness, merge, or a qualified combined
  readiness-and-merge action. One delivery action
  may group commit, push, and creation or update of one named draft PR only
  when all use the same verified branch and diff. The reply expires on any
  target, verified-head, changed-path/scope, or action change; it never
  authorizes another repository or PR, a later phase, downstream sync, a
  branch-protection or host-control bypass, or an unmentioned merge.
  Owner approval of the exact current verified head of a genuine plan-only PR
  continues to authorize readiness and merge of that same unchanged plan-only
  PR after required checks/reviews; at execution, the controller may present
  that unchanged PR/head and already-authorized readiness-and-merge sequence
  without requiring a new direct reply. This exception never applies to an
  implementation PR.
  Plan-only authorization never merges an implementation PR; an
  implementation merge requires its own exact pending merge action, except for
  a qualified source-local documentation/skill PR whose final prompt explicitly
  names both readiness and merge and meets the publication reference's gates.
  Host,
  repository, branch-protection, required-check, and reviewer enforcement
  remain additional requirements.
- Where no independent plan gate is required, a plan and implementation may
  share one PR. Early plan approval is only a checkpoint; final approval comes
  after the complete implementation diff is available. L3 work and any work
  requiring an independent plan gate retain a merged plan-only PR followed by
  a separate implementation PR.
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
  final merge authority. A qualifying source-local documentation/skill PR may
  use the explicit combined final action defined in the publication reference.
  An initiative bundle may continue only inside its
  approved envelope and stops at declared manual boundaries.
- Before every phase, check dirty state and both pause files, obtain minimum
  scope/gate/current evidence, run the smallest right-reason red gate, and set
  a three-attempt cap unless the plan says otherwise.
- Validate the in-scope diff and green evidence before publication. Publish a
  green phase only with its trailer and explicit-path staging. Keep a
  plan-only PR in GitHub draft state while verifying it; after owner approval
  of its current verified head and required checks/reviews pass, mark it ready
  and merge it. Keep an implementation PR draft until final validation and
  triggered review pass, then require final owner approval and normal merge
  authority; only a qualified source-local documentation/skill PR may request
  readiness and merge together. A manual next-phase gate does not prevent publication of the
  completed phase.
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
| Implementation delegation | `opencode-delegate` + `references/delegation.md` | `opencode-delegate` + [`references/delegation.md`](references/delegation.md) |
| Triggered independent read-only review | `claude-delegate` + `references/review-and-wait-policy.md` | `codex-delegate` + [`references/review-and-wait-policy.md`](references/review-and-wait-policy.md) |
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
