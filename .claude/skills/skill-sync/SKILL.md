---
name: skill-sync
description: Safely prepare, validate, and deliver an approved shared-skill sync across explicitly named repositories. Use when Claude must add, update, compare, or reconcile selected skills in multiple repositories while preserving repository-local guidance and active worktrees.
---

# Skill Sync

Synchronize only an approved shared-skill manifest across explicitly named repositories. Repository-local guidance and every active worktree are protected.

Claude applicability is approved-manifest based. The Codex coordination skill remains `opencode_mcp`-source-local: never copy `.codex/skills/skill-sync/` to targets or include it in a distribution manifest. Target-only drift and deletion safeguards are the same across runtimes.

## Inputs

Before writing, have the authoritative source repository and immutable revision, exact selected source manifest, target repositories and base branches, expected changed paths, validation, and explicit authority for the requested actions. Never assume the current repository is the source or reuse an earlier target list.

An owner discovery scope such as `../` permits direct-child read-only discovery only. Show each child as usable Git, non-Git, or inaccessible/broken, with its canonical path and available cached remote/base information. No recursion, fetch, active-worktree inspection, or worktree creation during discovery. Obtain confirmation or exclusions before target preflight; confirmation selects targets, not delivery authority. List drift needs renewed confirmation.

## Preflight

Before target writes:

1. Confirm every selected source path at the named revision and freeze the exact manifest.
2. Fetch confirmed targets and verify their remote bases and evaluated revisions.
3. Compare the manifest with each target, including target-only files inside the selected source-skill folders. Missing files are expected; distinguish approved variants from unexplained drift. Never widen the manifest to make a target pass.
4. Inspect active-worktree state only to preserve it and select task-owned isolated worktrees.
5. State expected changes and trusted validation commands. Stop on unexplained drift, ambiguous authority, or unavailable required evidence.

## Delivery

- Use one isolated worktree/branch per target from its approved base. Never checkout, reset, clean, stash, or otherwise modify a target's active worktree.
- Copy only the approved manifest. Preserve approved metadata and runtime variants, comparing meaning rather than blindly forcing byte parity. Preserve existing or required runtime metadata, including approved Codex `agents/openai.yaml`; do not invent Claude metadata or silently add paths.
- Never sync `AGENTS.md`, `CLAUDE.md`, `.opencode/`, application files, or repository-specific guidance/skills unless those exact paths were approved.
- Delete target-only content only under explicit exact-source-parity authority naming the path and expected deletion. Verify resulting parity. Do not copy target-only content into the source or silently retain it as an undocumented variant.
- Validate `git diff --check`, trusted `quick_validate.py` for each copied skill, relative links, and exact manifest scope. Failed validation or unresolved required review blocks delivery; do not substitute an untrusted target validator.
- Use local Git for local inspection and authorized local work; stage explicit paths only. Use the repository's authorized GitHub route (GitHub MCP/app or authenticated `gh`). A workflow specifically requiring GitHub MCP permits no silent fallback. Neither route grants publication authority or expands a read-only reviewer's permissions.

For source changes, follow `phase-gated-implementation`: normally one plan-plus-implementation PR, with approved local phases/reviews/controller commits covered by the plan approval. A pure target copy skips source implementation, not target preflight. Explicitly authorized local-only preparation stops locally and reports unpublished state; do not invent a remote branch or PR.

## Approval

A scoped delivery request may group its named copy, validation, commit, push, and PR actions. Resolve source revision, configured destination, target, base, worktree/branch, exact payload, metadata, and actions before requesting approval. Owner-facing task/plan/PR names suffice; the controller resolves concrete revisions. No SHA-only approval or duplicate disclosure prompt for already-bound in-scope payloads. Local or source authority never implicitly authorizes target publication, readiness, or merge.

For an eligible draft without earlier readiness authority, one explicit request may cover marking that named PR ready and merging it. For an already-ready PR, request merge approval. Merge-only never implies readiness. Recheck scope, head/base, eligibility, and required checks/reviews before readiness and again before expected-head merge. Drift invalidates the affected approval; protections are never bypassed. Eligible exact plan-only or genuinely nonoperational documentation-only content may use the combined content-and-merge route in `phase-gated-implementation`; skills, profiles, policy, mixed, uncertain, or operational content do not qualify.

### Qualified mechanical shared-skill bundle

For a pure copy of shared skills from one already-merged source revision, freeze a stable bundle ID, exact portable manifest and content digests, every target/base and evaluated revision, task-owned worktree/branch, expected copy-only diff, applicability, trusted validation, exclusions, and permitted actions before any target write.

Local adaptation, runtime-variant edits, unexplained target-only drift, missing/extra files, untrusted validation, application/configuration/generated changes, or local guidance disqualify this shortcut. Preserve variants through the normal scoped route instead. Never add an unlisted repository or infer deletion authority.

One explicit bundle delivery approval may cover listed copying, validation, explicit staging, commits, pushes, draft-PR create/update, and administrative readiness. Readiness must be expressly included and still waits for validation/review eligibility. It grants no merge, retry, adaptation, force-push, settings/protection changes, auto-merge, bypass, release, or deployment authority.

Then request one final bundle merge approval listing the stable ID and every repository, PR, expected head, base plus evaluated revision, checks/reviews, and merge method. Merge sequentially, revalidating each binding immediately before expected-head merge. On drift, failure, eligibility regression, host rejection, or uncertain outcome, stop all remaining entries and report partial completion. Any remainder or retry needs fresh scoped authority; never silently narrow the bundle.

## Stop and report

Stop on ambiguous source/paths/targets/bases, unconfirmed lists, unexplained drift, out-of-manifest changes, failed validation/review, missing required MCP or authority, or risk to active worktrees/protected guidance. Keep useful authority and actual PR-state evidence, not a compulsory lifecycle ledger.

Report source revision, manifest, each target/base/branch/PR, validation, approved variants, skipped/blocked targets, and actual local/publication/readiness/merge state. Wait required jobs and commands to terminal and retrieve output before success; never duplicate live/unknown work. At owner stop or declared deadline, account for pending resources honestly. Return the source repository to its original branch when applicable; no worktree cleanup without explicit authority.
