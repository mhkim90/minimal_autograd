---
name: phase-gated-implementation
description: Control approved phased work from a plan, issue, design, or checklist. Keep Claude Code as final publication gate, route implementation by difficulty, and stop only at real boundaries.
---

# Phase-Gated Implementation

Interactive Claude Code owns scope, final gates and publication; delegate implementation by default. Codex review is read-only. Route by difficulty, gate by risk. Report observed models, not route labels.

## Owner boundaries

Get explicit authority for push, PR create/update/ready, merge, release, deploy, cross-repo sync and destructive or irreversible actions. Group named actions; approval binds only the unchanged scope and actions. Task/plan/PR names suffice, not SHAs or magic replies. Memory and handoffs are leads, not approval.

Bind repo, branch, head/base, scope and destination. Recheck eligibility, checks and reviews before each authorized action; verify remote outcomes. Drift, rejection or uncertainty stops the affected sequence; never bypass protections.

- Ready PR: ask for merge. Eligible draft: ask once for ready **and** merge if readiness is not already authorized. Neither implies the other; recheck before ready and expected-head merge.
- Only exact plan-only or genuinely nonoperational docs-only PRs may combine content and merge approval (explicit readiness if needed). Verify all paths, modes, classification, head/base and eligibility before and after approval. Mixed/uncertain changes, operational guidance (including skills, profiles and agents), policy, workflows, config, code, generated files, renames, symlinks and executables use ordinary final merge control.
- Qualified mechanical sync: [skill-sync](../skill-sync/SKILL.md) permits explicit administrative readiness in delivery approval and one final bundle merge request. Source approval never covers targets; preserve local drift, deletion boundaries, variants and dirty checkouts. Never distribute Codex's source-local sync skill. Claude sync follows the approved manifest.

Already-bound configured delegates/reviewers need no second approval for necessary in-scope source/diffs. Changed destinations, unconfigured services or out-of-scope private material need authority. Use the repository-authorized GitHub route. Where MCP/app is required, never silently use `gh`; reviewers remain read-only.

## Plan

For nontrivial or L2–L4 work, write the goal, scope, phases, risks, dependencies, verification, acceptance and manual gates. Recommend `grilled-me` for your plan, `plan-audit` for externally authored plans (including summaries). Resolve blockers first. L4 or consequential API, security, migration, trust or release decisions need owner design direction before implementation.

Default to one branch and one plan-plus-implementation PR; split only at a security, compatibility, migration, release, rollback, ownership or independent-decision boundary. Plan approval covers local edits, tests, reviews, explicit staging and controller commits, never publication. Preserve approved revision, base, immutable plan/P0, ancestry and topology. Renew approval before material scope, risk, design, acceptance or authority changes.

## Routes

Read [opencode-delegate](../opencode-delegate/SKILL.md) before selecting or using an OpenCode route. Expert consultation is for material questions or blockers. Implementers edit/test only.

## Each phase

1. Verify checkout, approved dirty state and prerequisites. Stop on unexpected checkout/changes. When meaningful, establish a pre-change observation or right-reason failure, not a broken fixture.
2. Fix ordinary in-scope mistakes automatically. For a recurring blocker, return evidence for a changed approach, expert help or owner direction; no blind retry. Carry explicit budgets, shared consumption and deadlines across sessions, reloads, recovery and review fixes. Stop at exhaustion; no reset by renaming, no invented candidate/consultation caps.
3. Validate focused checks and scoped diff. Keep code comments short and code-specific. Stop on contradictions, denial or unresolved review blockers. For missing capability suspend affected edits and follow delegate recovery; missing native `apply_patch` alone is not proof.
4. After L3 implementation obtain independent read-only [codex-delegate](../codex-delegate/SKILL.md) review. Stop if unavailable; neither an OpenCode expert nor the advice author qualifies. Seek a second opinion for warranted security, concurrency, numerical, API/release, weak-test or disagreement risks.
5. Before success, commits, publication or advancement, retrieve terminal output for all required jobs/commands. Never duplicate live/unknown work; wrapper completion or prose is not execution proof. At owner stop/deadline report pending resources without guessing cause.
6. Only the controller stages explicit paths and commits locally. Continue approved local successors without re-asking until a manual gate, material change or blocker. No new checkout or remote authority.

## Report

Briefly give changes, commands/results, local/remote state, allocation and lineage, review disposition, blockers and next step. Preserve authority evidence; no universal PR ledger.
