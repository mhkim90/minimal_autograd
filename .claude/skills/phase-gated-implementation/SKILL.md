---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Claude Code as final publication gate, route implementation by difficulty, and stop only at real boundaries.
---

# Phase-Gated Implementation

Claude Code is the controller: it owns scope, gates, and publication, and delegates implementation by default. Interactive Claude Code alone publishes; the Codex review route is read-only. Use judgment rather than policy counters. Report observed model evidence honestly; an intended model or route label is not proof of the active model.

## Owner boundaries

Push, PR create/update/ready, merge, release, deploy, cross-repository sync, and destructive or irreversible operations need explicit owner authority. Grouped requests may authorize several clearly enumerated actions together. Approval covers the unchanged actions and scope presented, not an omitted or later action. Plan approval never authorizes publication.

Owner-facing task, plan, and PR names suffice; do not demand SHA-only approvals or a magic reply. The controller resolves the concrete repository, branch, head/base, scope, destination, and eligibility. Recheck these and required checks/reviews before acting; verify remote state afterward. Drift or an uncertain outcome stops the affected sequence. Host protections are never bypassed.

For an already-ready PR, request approval to merge that named PR. For an eligible draft without earlier readiness authority, ask once to mark it ready and merge it. Recheck before readiness and again before expected-head merge. Merge-only approval never implies readiness; readiness never implies merge authority.

An exact plan-only or verified genuinely nonoperational documentation-only PR may combine content and merge approval. Explicitly include readiness if needed. Verify its complete file set, modes, classification, head/base, and eligibility before and after approval. Mixed or uncertain content, operational documentation, skills, profiles, agents, guidance, policy, workflows, configuration, code, generated content, renames, symlinks, executable files, and plan-plus-implementation PRs retain ordinary final merge control.

Qualified mechanical-sync bundles may explicitly include administrative readiness in delivery approval and use one final bundle merge request under [skill-sync](../skill-sync/SKILL.md). Claude sync remains approved-manifest based; the Codex skill-sync is `opencode_mcp`-source-local and must never be distributed. Source approval never transfers to targets; that skill retains target-only drift/deletion, runtime-variant, and dirty-checkout protections.

Necessary in-scope source/diff disclosure to an already-bound configured delegate or reviewer needs no duplicate permission prompt. Changed destinations, out-of-scope private material, and unconfigured services need authority; host rejection is never bypassed.

## Plan

Non-trivial or L2–L4 work starts from a written plan: goal, scope, phases, risk, verification, done criteria, dependencies, and manual boundaries. Planning review is recommended: `grilled-me` for your own draft, `plan-audit` for external authorship. Summarizing an external plan does not change its provenance. Resolve blocking findings before execution.

For L4 or consequential API, security, migration, trust, or release decisions, get owner design direction before implementing. Default delivery is one branch and one PR containing the plan and implementation. Split only at a real security, compatibility, migration, release, rollback, ownership, or independent-decision boundary.

One plan approval authorizes its local edits, tests, reviews, explicit staging, and controller-local commits without per-step prompts. Preserve the approved revision, original base, and any task-bound immutable plan/P0 and ancestry. Do not silently rewrite checkpoints, refresh the base, or change topology. Material scope, risk, design, acceptance, or authority changes need renewed approval before affected work.

## Routes

Keep safety risk separate from difficulty. Use [opencode-delegate](../opencode-delegate/SKILL.md):

- Standard and mechanical implementation: `agent="terra-implementer"`.
- Hard implementation, with a stated reason: `agent="sol-implementer"` or `agent="astra-implementer"`.
- Explicitly requested Luna implementation only: `agent="luna"`; never an implicit fallback.
- A material hard question or blocker: fresh read-only `agent="astra-expert"`. Preflight is risk-driven, not universally required. It never edits, publishes, delegates, resets allocations, or reviews its own advice.
- An explicitly approved whole phase: `agent="astra-orchestrator"` beneath this controller, which retains final gates, independent review, and publication. This is a procedural boundary, not caller authentication.
- The configured OpenCode default only when the owner asks; omit `agent`, `model`, and `variant`.

Named profiles bind their model and effort: omit per-call model and variant. Rejected selectors, conflicting overrides, reported role/model mismatches, silent fallback, or mismatched continuation stop the phase; never substitute. Missing telemetry alone is not proof of a mismatch. Implementers edit/test only; only the controller stages, commits, or publishes.

## Each phase

1. Verify the expected checkout, approved dirty state, and prerequisite evidence. Show a right-reason red failure or pre-change observation when meaningful; a broken fixture is not the intended red signal.
2. Implement or delegate within scope. Repair ordinary in-scope mistakes automatically. Repeated same-blocker failure returns evidence to the controller for a materially changed approach, useful expert help, or owner direction—not a blind loop.
3. Preserve explicit task-specific allocations, shared limits, consumed history, and deadlines across reloads, sessions, recovery, and review fixes. Reload or renamed stages never reopen an exhausted allocation. Do not invent default numeric candidate or consultation caps.
4. Validate focused checks and the scoped diff. Stage explicit paths only. Keep code comments short and about the code, not tickets or gates.
5. Independent finish review is required after L3 implementation through [codex-delegate](../codex-delegate/SKILL.md), read-only. Also seek a second opinion when security, concurrency, numerics, API/release behavior, weak tests, or disagreement warrants it. An advice author cannot independently review that advice. If required review is unavailable, stop; an OpenCode expert is not a substitute.
6. The controller commits locally and continues approved local successors without another prompt, stopping at manual boundaries, changed authority/scope, or real blockers. Continuation grants no new checkout, remote action, or publication authority.

Stop for contradictory evidence, unexpected checkout or dirty work, missing essential capability, access denial, exhausted explicit limits, or unresolved review blockers. Discover capabilities pragmatically under the delegate contract; missing native apply_patch does not establish that no permitted editor exists. Memory and handoff notes are leads, never approval.

Use the repository's authorized GitHub route. A workflow that specifically requires GitHub MCP/app does not permit silent `gh` fallback; neither route expands a read-only reviewer's authority.

## Report

Keep it short: changes, commands/results, actual local and remote state, applicable allocations and lineage, review disposition, blockers, and next successor or boundary. Retain useful authority evidence without a universal PR-lifecycle ledger or proof ritual.

Wait required jobs and commands to terminal and retrieve their output before success, commits, publication, or phase advancement. Never duplicate live or unknown work. Wrapper completion or prose is not tool-execution evidence. At owner stop or a declared deadline, report unresolved resources honestly as pending; do not claim success or infer their cause.
