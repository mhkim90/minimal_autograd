# Target Workflow

Read this reference immediately before target worktree or PR work. The core
skill's required inputs, preflight, protected worktrees, drift policy,
approval, staging, GitHub route, and stop rules remain authoritative.

When the owner supplied a candidate-discovery scope, first show the complete
direct-child read-only candidate table and obtain exclusions or confirmation.
Do not fetch, inspect active worktrees, create worktrees, or treat candidates
as targets before that confirmation.

1. Create one isolated worktree per target from the named remote base branch.
   Do not checkout, reset, clean, stash, or otherwise modify the target's
   active worktree.
2. Copy only the approved manifest into that worktree. Keep documented
   runtime-specific variants separate; compare semantic parity rather than
   blindly forcing byte equality when such variants are approved.
3. Verify the exact changed-file list, run `git diff --check`, and run
   `quick_validate.py` for every copied skill folder.
4. Use normal target-specific delivery and merge approval unless the complete
   qualified mechanical-sync bundle record is frozen before any target write.
   Its source revision/digests, target bases and evaluated revisions,
   task-owned branches/worktrees, expected copy-only diffs, applicability,
   trusted validation, exclusions, and permitted operations must all match.
5. Under one explicit bundle-delivery authorization, stage explicit manifest
   paths only and create/update one draft PR per listed target through GitHub
   MCP. It permits no force-push, local adaptation, settings/protection change,
   auto-merge, bypass, or unlisted action. A disqualified or failed target is
   quarantined; do not silently narrow or retry the bundle.
6. Mark eligible PRs ready administratively. For a qualified bundle, present
   one final stable-ID snapshot listing every `repository / PR / expected head
   / base plus evaluated revision / checks and reviews / merge method`; otherwise
   use the normal single-PR prompt. Revalidate every listed entry immediately
   before sequential expected-head merge. On drift, regression, host failure,
   or uncertain outcome, stop all remaining entries and report partial state;
   fresh exact bundle authority is required for any remainder.

Update the PR-lifecycle ledger for every target PR transition, including
created, adopted, updated, readied, reopened, closed, and merged events, and
for externally observed changes. Record repository/role/branch/head/base, the
complete changed-file set and file modes, classification, lifecycle, owner
disposition, authority evidence, and last validation. Do not complete sync or
claim success while any entry is unresolved; a Stop report may list it.
