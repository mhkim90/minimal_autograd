---
name: skill-sync
description: Safely prepare, validate, and deliver an approved shared-skill sync across explicitly named repositories. Use when Codex must add, update, compare, or reconcile selected skills in multiple repositories while preserving repository-local guidance and active worktrees.
---

# Skill Sync

Synchronize only an approved shared-skill manifest. Treat repository-specific
guidance and every active worktree as protected.

## Required inputs

Before writing, require all of the following:

- authoritative source repository and source revision;
- exact source paths to synchronize;
- target repositories and each target's remote base branch;
- whether the request is plan-only, source implementation, target delivery, or
  review/merge;
- explicit authority for the requested publication action.

The current repository is never an implicit source. Do not discover targets by
searching parent directories or reuse a target list from an earlier task.

## Read-only preflight

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

Never widen the manifest to make a target pass. Never synchronize `AGENTS.md`,
`CLAUDE.md`, `.opencode/`, application files, or repository-specific skills
unless the owner named those paths in the approved manifest.

A target-only file inside an approved source-skill folder is unexpected drift
by default. Do not copy it into the source or retain it as an undocumented
variant. Delete it only when the owner-approved plan names the exact path and
explicitly requires exact source parity; record that removal in the expected
changed-file list and post-copy comparison.

## Source change workflow

For a new or material shared-skill change, follow `phase-gated-implementation`:

1. Create a plan-only draft PR in the authoritative source repository. Include
   source paths, target matrix, validation, non-goals, and stop rules.
2. Wait for owner approval tied to the plan commit SHA.
3. Implement only the approved source manifest. Initialize a new Codex skill
   with the skill-authoring tool and include `agents/openai.yaml`; create the
   corresponding Claude skill only when it is in scope.
4. Run `quick_validate.py` for every source skill folder and `git diff --check`.
   Confirm that the source diff contains only the approved plan and manifest.
5. Update the same PR. Mark it ready only after all source gates pass; merge
   only with explicit owner approval.

For an L1 mechanical fast-path source update, use the phase-gate qualification
instead: put the exact source/target manifest, validation, non-goals, and
publication policy in one normal source PR; skip the plan artifact and
plan-approval wait, but retain target preflight, validation, explicit staging,
and separate owner merge approval. For a pure target sync of an already merged
source revision, record that source revision and skip source implementation;
do not skip target preflight or gates.

For an approved initiative bundle, keep every declared source phase on the
same draft source PR. A green internal phase records its commit and evidence;
it does not create another source PR or require a new plan approval unless the
approved envelope changes or a manual boundary applies.

## Target delivery workflow

1. Create one isolated worktree per target from the named remote base branch.
   Do not checkout, reset, clean, stash, or otherwise modify the target's
   active worktree.
2. Copy only the approved manifest into that worktree. Keep documented
   runtime-specific variants separate; compare semantic parity rather than
   blindly forcing byte equality when such variants are approved.
3. Verify the exact changed-file list, run `git diff --check`, and run
   `quick_validate.py` for every copied skill folder.
4. Stage explicit manifest paths only. Commit and push one branch per target.
   Bundle every approved phase delivery for that target on the same branch; do
   not create a target PR per internal phase.
5. Create or update one PR per target per initiative through GitHub MCP. Use a
   **draft** PR when the source workflow is plan-first; use a normal PR only
   for a qualified L1 mechanical fast path. Include the source revision,
   manifest, validation results, and any approved variant.
6. Wait for separate owner review and merge authorization for each target.

Use GitHub MCP for required PR creation, update, readiness, comments, and merge
actions. If it is unavailable, stop and report the missing capability; do not
silently use another publication route. A target validation or publication
failure blocks that target only unless it exposes a source-manifest defect.

## Stop rules

Stop and request direction for:

- missing or ambiguous source authority, revision, paths, targets, or bases;
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
