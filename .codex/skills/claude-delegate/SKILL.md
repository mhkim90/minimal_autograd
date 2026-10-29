---
name: claude-delegate
description: Delegate a triggered independent, read-only review to Claude through the Claude MCP. Use it to stress-test high-risk reasoning, plans, or assumptions; do not use it for edits or execution.
---

# Claude Delegate

Claude gives a second opinion or the required L3 independent review. It can
read the caller's files with `Read`, `Glob`, and `Grep` and nothing else: no
commands, edits, publication, network, or credentials.

Use it for required L3 review and when architecture/API compatibility,
security, memory or concurrency, CUDA/numerical correctness, release
behavior, weak tests, or a disagreement make a second opinion worth its
cost. Report the model Claude actually ran on; do not infer it from the
route name.

## Prompt

```text
Context: <project + phase and relevant files>
Question: <specific claim, failure mode, or design fork>
Current evidence: <compact red/green/diff evidence>
Constraints: read-only; be concise; cite inspected paths
Return: findings, acceptance concern, and stop/go.
```

## Job

Use `claude_run_async` for anything meaningful and poll the same job. A
`running` job is pending, not unavailable; do not start a duplicate or
cancel it without a stated reason. Fetch `claude_job_result` before
reporting the verdict. Report one state: `review pending`, `go`, `blocker`,
or `unavailable` (missing tool, terminal error, or failed result
retrieval). If the required L3 review is unavailable, stop the phase.
