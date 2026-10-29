---
name: opencode-delegate
description: Delegate approved work through OpenCode MCP. Use Terra by default, Luna only on explicit request, justified Sol/Astra implementers for harder work, and Astra expert for bounded read-only analysis.
---

# OpenCode Delegate

Use `mcp__opencode.opencode_run_async` for non-trivial work and blocking calls only for known-short work. Delegate execution, iteration, and long test runs, not ambiguous design decisions or single commands. Independent review uses the runtime reviewer skill, not an implementation agent.

## Boundary

OpenCode MCP runs in the caller's working directory and accepts no selected `cwd`; external paths, including arbitrary `/tmp` worktrees, are denied. Shell tools with explicit cwd do not widen this delegation boundary. Another repository needs an owner-authorized isolated checkout inside the caller's boundary. These constraints themselves authorize no checkout, clone, edit, or publication.

Explicitly requesting a configured delegate authorizes necessary in-scope source/diff disclosure to it; do not ask a duplicate disclosure question. Changed destinations, out-of-scope private material, unconfigured services, destructive actions, and publication retain their own authority requirements.

Before editing, confirm the expected checkout and relevant capabilities using tools actually available to the delegate. A contained worktree can still depend on inaccessible linked/common Git metadata. Do the checks needed for the task, not a universal marker/hash proof ritual; controller visibility does not prove delegate access.

Missing native `apply_patch` does not establish editing is impossible: an existing permitted shell editor may work. Discover rather than assume tool names, and verify the result of an authorized edit. Do not install tools, widen permissions, or invent extra probe writes.

Access denial stops affected work. Denied Git metadata means stop—not copy or redirect metadata, change git-dir/work-tree or symlinks, widen permissions, or relaunch to evade controls. A separately owner-approved independent clone is an option only after that failure: bind source/ref/revision, destination, task branch, scope, and actions; preserve original dirty work and plan/P0 provenance; then reverify capabilities there. Never create or substitute it automatically.

Implementers edit and test only: no planning, review, staging, commits, pushes, PR actions, further delegation, subagents, or publication. The explicitly approved controller-bound orchestrator is a planning/gate-recommendation exception, not an editor or publisher. Permissions are guardrails, not scope or a complete sandbox. Never access or expose credentials or secrets.

## Routes

- Standard and mechanical work: `agent="terra-implementer"`.
- Hard implementation, with a stated reason: `agent="sol-implementer"` or `agent="astra-implementer"`.
- Explicitly requested Luna implementation only: `agent="luna"`; never an implicit fallback.
- Hard question or blocker analysis: fresh read-only `agent="astra-expert"`. Return findings, proposed approach, acceptance gate, and stop/go. Inspect only evidence needed for the named decision and stop when evidence is missing or further inspection cannot make meaningful progress. Never edit, delegate, publish, or review its own advice.
- Explicitly approved whole phase: `agent="astra-orchestrator"` beneath an external Codex/Claude controller retaining final gates, independent review, and publication.
- Configured default only when the owner asks: omit `agent`, `model`, and `variant`.

Named agents bind exact model and high effort: omit model and variant on calls, continuations, and forks. Rejected selectors, conflicting overrides, reported role/model mismatches, fallback, or broken session lineage stop the phase; never substitute. Record the requested route, job/session lineage, and observed model when available. Missing model/usage telemetry alone after an honored selector and terminal requested-role response is an observability warning, not proof of mismatch.

Expert preflight is risk-driven, not universally compulsory. L3 independent finish review remains required through `claude-delegate` for a Codex controller or `codex-delegate` for a Claude controller; expert advice never replaces it.

## Prompt

Give a cold, self-contained task:

```text
Context: project, one bounded phase, relevant files and prior evidence
Task: approved change or run loop
Checkout: expected worktree, approved dirty state, governing plan
Scope: allowed paths/actions; no stage/commit/publish/delegation
Red and done: right-reason pre-change signal, focused checks and acceptance
Limits: task-specific limits/deadline and approved recovery, if any
Return: changed files, commands/results, blockers, job/session and observed route
```

Do not invent default allocations or extra approval gates. For Astra expert, send one focused question, relevant diff/test evidence, expected evidence, and meaningful progress/stop conditions. Read-only experts need authorized read capability, not an editor gate.

## Failures and bounded lifecycle

Fix routine in-scope mistakes—path typos, malformed patches, fixtures—without ceremony within unchanged scope, acceptance, and remaining allocation. An incidental typo in an intended approved path is not authoritative checkout drift; never search outside scope to repair a guess or call a broken fixture the intended red gate.

Repeated same-blocker failure returns evidence to the controller for a materially revised approach, useful expert help, or owner direction. No blind loop. Keep one bounded phase/subphase per same-role session and end at its acceptance milestone; preparation RED is not implementation GREEN.

Honor explicit task-specific limits, shared allocations, consumed history and deadlines across reloads, recovery and review fixes. No default numeric interactive candidate or consultation cap applies. Count work honestly where an explicit allocation requires it; a renamed stage/session/role never reopens an exhausted limit. A separately approved successor inherits only limits that actually bind it.

Missing essential capability suspends affected edits. Distinguish model-reported absence, observed invocation failure, denial, and unknown cause; unknown stays unknown. Wrapper exit, printed JSON, or prose is not proof of native tool execution. No installs, role swaps, denial workarounds, blind probes, or self-launched recovery. Only expressly approved controller recovery may proceed after prior terminal state, result retrieval, and scoped partial-diff inspection, retaining lineage, limits, provenance, and fresh capability verification.

## Waiting and reporting

Declare realistic task-specific checkpoints, maximum wait, and synthesis time within any overall deadline. Track required job/command IDs and statuses, keeping actual launch-to-terminal time distinct from controller/owner wait. Poll the same job until terminal and retrieve output before reporting a verdict or passing a gate. Advancing events can show progress without partial text; `running` alone is liveness, not failure or no-progress evidence.

Never duplicate live/unknown work or silently abandon it. Cancel only for a stated error, owner/controller stop, or declared deadline procedure. Cancellation requests are not terminal proof; confirm terminal state and retrieve output before any authorized replacement. Required live/unknown resources block success, commits, publication, and phase advancement. At owner stop or hard deadline, report each pending resource, retrieval state, and next decision honestly.

Return concise changes, commands/results, applicable allocation history, lineage, and blockers. Report authorized invocation/exit/readback evidence, not private provider logs or fabricated usage. Usage is observational, never a quality gate; do not revive `usage_mcp`. Cross-repository delivery follows `skill-sync`; the Codex skill is `opencode_mcp`-source-local and never distributed. GitHub actions retain the configured MCP/app route, never silent `gh` fallback.
