---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Claude Code as final publication gate, route implementation by difficulty, and stop only at real boundaries.
---

# Phase-Gated Implementation

Claude Code controls scope, gates and publication, delegating implementation by default. Interactive Claude Code alone publishes; the Codex review route is read-only. Use judgment, not policy counters; report observed model evidence honestly. A route label or intended model is not proof of the active model.

## Owner boundaries

Push, PR create/update/ready, merge, release, deploy, cross-repository sync, and destructive or irreversible operations require explicit owner authority. One request may group clearly enumerated actions; approval covers only the unchanged actions and scope presented. Plan approval never authorizes publication.

Task, plan and PR names suffice; do not demand SHA-only approval or a magic reply. Resolve repository, branch, head/base, scope, destination and eligibility; recheck them and required checks/reviews before acting and verify remote state afterward. Drift or an uncertain outcome stops the affected sequence; never bypass host protections.

Request merge approval for a named ready PR. For an eligible draft lacking readiness authority, ask once for readiness and merge. Recheck before readiness and before expected-head merge. Merge-only does not authorize readiness, nor readiness merge.

An exact plan-only or verified genuinely nonoperational documentation-only PR may combine content and merge approval, explicitly including readiness if needed. Verify the complete file set, modes, classification, head/base and eligibility before and after approval. Mixed or uncertain content, operational documentation, skills, profiles, agents, guidance, policy, workflows, configuration, code, generated content, renames, symlinks, executable files and plan-plus-implementation PRs retain ordinary final merge control.

Qualified mechanical-sync bundles may explicitly include administrative readiness in delivery approval and use one final bundle merge request under [skill-sync](../skill-sync/SKILL.md). Claude sync remains approved-manifest based; Codex skill-sync is `opencode_mcp`-source-local and must never be distributed. Source approval never transfers to targets; target-only drift/deletion, runtime-variant and dirty-checkout protections remain.

Necessary in-scope source/diff disclosure to an already-bound configured delegate or reviewer needs no duplicate prompt. Changed destinations, out-of-scope private material and unconfigured services need authority; never bypass host rejection.

## Plan

Non-trivial or L2–L4 work starts with a written plan covering goal, scope, phases, risk, verification, done criteria, dependencies and manual boundaries. Planning review is recommended: `grilled-me` for your own draft, `plan-audit` for external authorship; summarizing an external plan does not change its provenance. Resolve blockers before execution.

For L4 or consequential API, security, migration, trust or release decisions, get owner design direction before implementing. Default to one branch and one plan-plus-implementation PR; split only at a real security, compatibility, migration, release, rollback, ownership or independent-decision boundary.

Plan approval covers local edits, tests, reviews, explicit staging and controller-local commits without per-step prompts. Preserve the approved revision, original base and any task-bound immutable plan/P0 and ancestry; do not silently rewrite checkpoints, refresh the base or change topology. Renew approval before material scope, risk, design, acceptance or authority changes affect work.

## Routes

Separate safety risk from difficulty. Use [opencode-delegate](../opencode-delegate/SKILL.md):

- Standard and mechanical implementation: `agent="terra-implementer"`.
- Hard implementation, with a stated reason: `agent="sol-implementer"` or `agent="astra-implementer"`.
- Explicitly requested Luna implementation only: `agent="luna"`; never an implicit fallback.
- Material hard question or blocker: fresh read-only `agent="astra-expert"`. Preflight is risk-driven, not universal; it never edits, publishes, delegates or resets allocations.
- Explicitly approved whole phase: `agent="astra-orchestrator"` beneath this controller, which retains final gates, independent review and publication. This procedural boundary is not caller authentication.
- Configured OpenCode default only when the owner asks; omit `agent`, `model` and `variant`.

Named profiles bind model and effort; omit per-call model and variant. Rejected selectors, conflicting overrides, reported role/model mismatches, silent fallback or mismatched continuation stop the phase without substitution. Missing telemetry alone is not proof of mismatch. Implementers edit/test only; the controller alone stages, commits or publishes.

## Each phase

1. Verify expected checkout, approved dirty state and prerequisites; stop on unexpected checkout or dirty work. When meaningful, show a right-reason red failure or pre-change observation; a broken fixture is not the intended red.
2. Implement or delegate within scope; repair ordinary in-scope mistakes automatically. Repeated same-blocker failure returns evidence to the controller for a materially changed approach, useful expert help or owner direction, not a blind loop.
3. Preserve explicit task-specific allocations, shared limits, consumed history and deadlines across reloads, sessions, recovery and review fixes. Stop on exhausted explicit limits; reloads or renamed stages never reopen them. Invent no default numeric candidate or consultation caps.
4. Validate focused checks and scoped diff; stage explicit paths only. Keep comments short and about code, not tickets or gates.
5. After L3 implementation, independent read-only finish review through [codex-delegate](../codex-delegate/SKILL.md) is required. Seek a second opinion when security, concurrency, numerics, API/release behavior, weak tests or disagreement warrants it. An advice author cannot independently review that advice. Stop if required review is unavailable; an OpenCode expert is no substitute.
6. The controller commits locally and continues approved local successors without another prompt, stopping at manual boundaries, changed authority/scope or real blockers. Continuation grants no new checkout, remote action or publication authority.

Stop for contradictory evidence, missing essential capability, access denial or unresolved review blockers. Discover capabilities pragmatically under the delegate contract: missing native apply_patch does not prove no permitted editor exists. Memory and handoff notes are leads, not approval.

Use the repository's authorized GitHub route. A workflow specifically requiring GitHub MCP/app does not permit silent `gh` fallback; neither route expands a read-only reviewer's authority.

## Report

Briefly report changes, commands/results, actual local and remote state, applicable allocations and lineage, review disposition, blockers and next successor or boundary. Keep useful authority evidence without a universal PR-lifecycle ledger or proof ritual.

Wait for required jobs and commands to reach terminal state and retrieve output before success, commits, publication or phase advancement. Never duplicate live or unknown work; wrapper completion or prose is not tool-execution evidence. At owner stop or a declared deadline, honestly report unresolved resources as pending without claiming success or inferring cause.
