---
name: skill-sync
description: Safely prepare, validate, and deliver an approved shared-skill sync across explicitly named repositories. Use when Claude must add, update, compare, or reconcile selected skills in multiple repositories while preserving repository-local guidance and active worktrees.
---

# Skill Sync

Synchronize only an approved shared-skill manifest. Treat repository-specific
guidance and every active worktree as protected.

## Applicability

Use this skill only for an approved shared-skill manifest across explicitly
named repositories. Claude runtime applicability intentionally remains named
approved-manifest based, while Codex is opencode_mcp-local and excludes
copying `.codex/skills/skill-sync/`. The target-only drift and deletion
safeguards below are otherwise equal across the runtime variants. Do not use it
to discover targets or to synchronize unapproved repository-local guidance.

## Required inputs

Before writing, require all of the following:

- authoritative source repository and source revision;
- exact source paths to synchronize;
- target repositories and each target's remote base branch, or an explicit
  owner candidate-discovery scope pending confirmation;
- whether the request is plan-only, source implementation, target delivery, or
  review/merge;
- explicit authority for the requested publication action.

The current repository is never an implicit source. Do not reuse a target list
from an earlier task.

## Candidate discovery

An explicit owner scope expression such as `../` authorizes **read-only
candidate discovery only**. Enumerate direct-child directories only; never
recurse. Classify each child as a usable Git repository, non-Git directory, or
inaccessible/broken Git entry without creating worktrees, fetching, inspecting
active-worktree state, or changing a repository. Show every candidate with its
canonical path, Git validity, cached discoverable remote/default base when
available, and blocker/exclusion reason.

Then ask the owner to name exclusions or confirm the exact displayed candidate
set. Confirmation freezes target selection only for the next read-only
preflight; it never authorizes source selection, delivery, PR actions, merge,
or reuse after the candidate list changes. An ambiguous scope or inaccessible
parent stops discovery.

## Context-bound publication approval

Normal source and target publication remains per-PR: delivery authority names
one repository/worktree, branch, exact paths or diff, and permitted actions;
the final request is `Approve merging <repository> PR #<N>?`. A plan never
authorizes delivery or merge. Readiness is administrative only after validation
and eligibility pass.

## PR-lifecycle ledger and combined classification

Maintain a non-authoritative lifecycle ledger for every workflow-touched source
or target PR, including created, adopted, updated, readied, reopened, closed,
and merged events. Each entry records host/repository/PR, source-or-target
role, branch, head, base, complete file set and file modes, artifact
classification, lifecycle state, owner disposition, authority evidence, and
last validation. It grants no delivery, implementation, readiness, close,
merge, exception, or sync authority.

Before source implementation, successful phase completion, downstream sync, or
success reporting, every entry must be merged, explicitly retained by
PR-specific boundary-scoped owner direction, or explicitly closed. Retained is
not an unmerged-plan exception. A terminal Stop report may list unresolved
entries but cannot claim success.

The only combined content-and-merge exceptions are an exact plan-only PR and a
verified non-operational documentation-only PR. Bind and revalidate
repository/PR/role/branch/head/base/eligibility, the complete file set, file
modes, and classification immediately before and after approval. Mixed,
renamed, symlinked, executable, generated, configuration-like, operational, or
uncertain documentation, plus profiles, skills, agents, local guidance, policy,
workflows, configuration, and code, uses ordinary per-PR merge control. A
plan-plus-implementation PR never qualifies. Drift requires fresh approval.

The sole cross-repository exception is a **qualified mechanical-sync bundle**.
Its frozen record must bind one already-merged source revision, exhaustive
portable manifest and content digests, every target/base and evaluated base
revision, task-owned isolated branch/worktree, expected copy-only diff,
trusted validation, and exclusions. It also confirms downstream applicability:
upstream approval alone never authorizes local guidance, permission, or policy
adaptation. Missing paths, target-only files, variants, semantic edits,
untrusted validation, code/configuration/generated/local-guidance changes, or
any unlisted action disqualify the bundle.

One explicit bundle-delivery authorization may name that unchanged record and
permit only isolated copy, trusted validation, explicit staging, commit, push,
draft-PR creation/update, and administrative readiness for all listed entries.
It expires on any record-field change and never permits force-push, settings or
protection changes, auto-merge, bypass, retry, adaptation, merge, release, or
an unlisted target. Source-side authority still never transfers implicitly.

One final bundle-merge request lists a stable bundle ID and every eligible
`repository / PR / expected head / base plus evaluated revision / checks and
reviews / merge method` snapshot. A direct `approve` is valid only for that
unchanged prompt. Revalidate every entry immediately before its sequential
expected-head merge. Cross-repository merges are non-atomic: on drift, failed
validation, eligibility regression, host failure, or uncertain outcome, stop
all remaining merges and report merged, unmerged, and uncertain entries.
Read-only reconciliation is allowed; retry, rollback, or a remaining subset
requires fresh exact bundle authority. Host protections remain mandatory.

Before source implementation, require either a merged plan PR or a complete
unmerged-plan exception recording its reason, implementation branch/PR when
available, and a concrete resolution event. Pause at that event until the
exception is resolved or explicitly renewed.

## Always-required read-only preflight

After target selection is explicit or confirmed, complete this read-only
preflight before any target write. Do not write until all five steps are
recorded:

1. Confirm every source path exists at the named revision and derive the
   manifest from those exact paths.
2. Fetch each named target and confirm its named remote base branch exists.
3. Compare the manifest against each target. Also enumerate target-only files
   inside every approved source-skill folder; do not treat repository-local
   skills outside those folders as manifest extras. Classify a difference as
   missing, approved drift, or unexpected drift; stop on unexpected drift.
4. Inspect target active-worktree state only to preserve it. Use isolated
   worktrees for all target changes, even when an active worktree is clean.
5. State the expected changed-file list and validation commands before edits.

## Protected guidance, worktrees, and drift

Never widen the manifest to make a target pass. Never synchronize `AGENTS.md`,
`CLAUDE.md`, `.opencode/`, application files, or repository-specific skills
unless the owner named those paths in the approved manifest.

A target-only file inside an approved source-skill folder is unexpected drift
by default. Do not copy it into the source or retain it as an undocumented
variant. Delete it only when an owner-approved source-parity intent names the
exact path and explicitly requires exact source parity. Before deletion, include
that exact path in the expected changed-file list; after copying, record
post-copy comparison evidence showing exact source parity.

Unexpected drift remains a stop condition; do not make the manifest broader to
absorb it.

Do not checkout, reset, clean, stash, or otherwise modify a target's active
worktree. Use an isolated worktree for target changes.

## Publication authority and required route

Stage explicit approved manifest paths only. Apply one exact pending delivery
action and only its separate explicit bounded delivery authority. Readiness is
mechanical after validation and review; it is not an owner action. Final merge
uses the bound per-PR prompt above unless the qualified mechanical-bundle record
and its fully enumerated final snapshot apply. A draft PR never authorizes
automatic merging of an implementation PR.

Use GitHub MCP for required PR creation, update, readiness, comments, and merge
actions. If it is unavailable, stop and report the missing capability; do not
silently use another publication route. A target validation or publication
failure blocks that target only unless it exposes a source-manifest defect.

## Source-side publication

Read [`references/source-workflow.md`](references/source-workflow.md)
immediately before source-side publication work.

## Target delivery

Read [`references/target-workflow.md`](references/target-workflow.md)
immediately before target worktree or PR work.

## Stop rules

Stop and request direction for:

- missing or ambiguous source authority, revision, paths, targets, or bases;
- ambiguous candidate-discovery scope, an inaccessible parent, or an unconfirmed
  candidate list;
- diff outside the approved manifest or unexplained target drift;
- validation failure, unresolved review blocker, or unavailable required MCP;
- a request to merge without explicit owner approval;
- any action that would alter an active target worktree or protected local
  guidance.

## Completion report

Report the source revision, approved manifest, each target/base/branch/PR,
validation evidence, approved variants, skipped or blocked targets, and merge
state. Return the source repository to its original branch when its work is
complete; do not delete temporary worktrees unless explicitly authorized.
