---
name: skill-sync
description: Safely prepare, validate, and deliver an approved shared-skill sync across explicitly named repositories. Use when Claude must add, update, compare, or reconcile selected skills in multiple repositories while preserving repository-local guidance and active worktrees.
---

# Skill Sync

Sync shared skills across explicitly named repositories; protect repository-local guidance and all active worktrees. Claude applicability is approved-manifest based. The Codex coordination skill is `opencode_mcp`-source-local: never copy `.codex/skills/skill-sync/` to targets or include it in a distribution manifest. Target-only drift and deletion safeguards are the same across runtimes.

## Inputs

Before writing, establish the authoritative source repository and immutable revision, exact selected source manifest, targets and base branches, expected changed paths, validation and explicit authority for requested actions. Never assume the current repository is the source or reuse an earlier target list.

An owner discovery scope such as `../` permits direct-child read-only discovery only. Show each child as usable Git, non-Git or inaccessible/broken, with canonical path and available cached remote/base information. Do not recurse, fetch, inspect active worktrees or create worktrees during discovery. Confirm targets or exclusions before preflight; confirmation selects targets, not delivery authority. List drift requires renewed confirmation.

## Preflight

Before target writes:

1. Confirm every selected source path at the named revision and freeze the exact manifest.
2. Fetch confirmed targets; verify remote bases and evaluated revisions.
3. Compare each target with the manifest, including target-only files inside selected source-skill folders. Missing files are expected; distinguish approved variants from unexplained drift. Never widen the manifest to make a target pass.
4. Inspect active-worktree state only to preserve it and select task-owned isolated worktrees.
5. State expected changes and trusted validation commands. Stop on unexplained drift, ambiguous authority or unavailable required evidence.

## Delivery

- Use one isolated worktree/branch per target from its approved base; never checkout, reset, clean, stash or otherwise modify an active target worktree.
- Copy only the approved manifest; stop on out-of-manifest changes. Preserve approved metadata and runtime variants by comparing meaning, not blindly forcing byte parity. Preserve existing or required metadata, including approved Codex `agents/openai.yaml`; do not invent Claude metadata or silently add paths.
- Never sync `AGENTS.md`, `CLAUDE.md`, `.opencode/`, application files or repository-specific guidance/skills without approval of those exact paths.
- Delete target-only content only with explicit exact-source-parity authority naming path and expected deletion; verify resulting parity. Neither copy it into the source nor silently retain it as an undocumented variant.
- Validate `git diff --check`, trusted `quick_validate.py` for each copied skill, relative links and exact manifest scope. Failed validation or unresolved required review blocks delivery; never substitute an untrusted target validator.
- Use local Git for local inspection and authorized local work; stage explicit paths only. Use the repository's authorized GitHub route (GitHub MCP/app or authenticated `gh`). A workflow specifically requiring GitHub MCP permits no silent fallback. Neither route grants publication authority or expands a read-only reviewer's permissions.

For source changes, follow [phase-gated-implementation](../phase-gated-implementation/SKILL.md) for planning, local work and PR gates. Pure copies skip source implementation, not target preflight. Local-only preparation ends unpublished: no remote branch or PR.

## Approval

One delivery approval may group named copy, validation, commit, push and PR actions. Before asking, bind source revision, destination, target/base, worktree/branch, payload, metadata and actions. Names suffice for the owner; the controller resolves revisions. No duplicate disclosure prompt for an already-bound payload. Source or local approval never implies target publication, readiness or merge.

For ordinary PRs, apply the [owner boundaries](../phase-gated-implementation/SKILL.md#owner-boundaries) before readiness and expected-head merge. Ask once for both when needed; neither approval implies the other. Drift invalidates affected approval. Skills and policy are operational, not documentation-only exceptions.

### Qualified mechanical shared-skill bundle

For pure copying of shared skills from one already-merged source revision, freeze before any target write: stable bundle ID, exact portable manifest and content digests, every target/base and evaluated revision, task-owned worktree/branch, expected copy-only diff, applicability, trusted validation, exclusions and permitted actions.

Local adaptation, runtime-variant edits, unexplained target-only drift, missing/extra files, untrusted validation, application/configuration/generated changes or local guidance disqualify the shortcut. Preserve variants through the normal scoped route; never add an unlisted repository or infer deletion authority.

One explicit bundle delivery approval may cover listed copying, validation, explicit staging, commits, pushes, draft-PR create/update and administrative readiness. Readiness must be expressly included and still waits for validation/review eligibility. No merge, retry, adaptation, force-push, settings/protection changes, auto-merge, bypass, release or deployment is authorized.

Request one final bundle merge approval listing the stable ID and every repository, PR, expected head, base plus evaluated revision, checks/reviews and merge method. Merge sequentially, revalidating each binding immediately before expected-head merge. Drift, failure, eligibility regression, host rejection or uncertain outcome stops all remaining entries: report partial completion. Any remainder or retry needs fresh scoped authority; never silently narrow the bundle.

## Stop and report

Stop on ambiguous source/paths/targets/bases, unconfirmed lists, missing authority, or risk to active worktrees/protected guidance. Keep useful authority and actual PR-state evidence, not a compulsory lifecycle ledger.

Report source revision, manifest, each target/base/branch/PR, validation, approved variants, skipped/blocked targets and actual local/publication/readiness/merge state. Wait for required jobs and commands to reach terminal state and retrieve output before success; never duplicate live/unknown work. At owner stop or declared deadline, account honestly for pending resources. Return the source repository to its original branch when applicable; no worktree cleanup without explicit authority.
