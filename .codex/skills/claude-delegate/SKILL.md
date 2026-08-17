---
name: claude-delegate
description: Delegate a triggered independent, read-only review to Claude through the Claude MCP. Use it to stress-test high-risk reasoning, plans, or assumptions; do not use it for edits or execution.
---

# Claude Delegate

Use Claude only for a bounded second opinion or independent gate. Claude may
inspect accessible files with `Read`, `Glob`, and `Grep`; it must not execute
commands, edit, publish, access credentials, or use network/filesystem writes.

## Trigger and boundary

Trigger review for architecture/API compatibility, security, memory or
concurrency, CUDA/numerical correctness, release behavior, weak tests,
reviewer disagreement, suspicious red gates, or explicit request. Record the
reviewer trigger reason and the active Claude model; do not infer identity from
the route label.

Use async for meaningful reviews; use blocking calls only for known-short
reviews. At launch, record the job ID, requested/bound model evidence,
phase-declared checkpoint interval, and maximum wait. Poll the same job at each
checkpoint; a `running` status with no activity telemetry is live and must be
reported as `review pending`, not unavailable. Missing partial text alone never
permits a duplicate, cancellation, or no-progress inference.

When a terminal `done` status arrives, fetch `claude_job_result` before
reporting the verdict. Report `review unavailable` only for a missing
route/tool, terminal error/cancellation, or failed terminal-result retrieval.
If a job remains live at its declared maximum wait, stop the affected phase as
`review pending at maximum wait` and request controller/owner direction; do
not cancel automatically. If tools are unavailable, report the missing route
rather than silently substituting it.

+## Usage correlation

When `usage_mcp` is configured, pass the controller's opaque `workflow_id` to
every Claude run. After its terminal result is fetched, query
`claude_usage_get`, bind the returned provider session to that workflow, and
retain partial or unavailable usage as a warning. Never use usage to infer a
budget, cancel the review, or weaken the phase gate.

## Prompt template

```text
Context: <project + phase and relevant files>
Question: <specific claim, failure mode, or design fork>
Current evidence: <compact red/green/diff evidence>
Reviewer trigger reason: <reason>
Checkpoint / maximum wait: <phase-declared values>
Constraints:
- DISCUSS/read-only; no shell, edits, publication, network, or credentials
- be concise; cite inspected paths when relevant
Return: findings, acceptance concern, and stop/go.
```

Report the job/thread identity, active/bound model, trigger reason, checkpoint
history, maximum-wait outcome, and one state: `review pending`, `go`,
`blocker`, or `unavailable`. Keep usage observational; do not add fixed token
or price semantics.
