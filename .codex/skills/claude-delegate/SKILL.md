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

Use async for meaningful reviews, poll status, fetch the result, and cancel only
when irrelevant. Blocking calls are for known-short reviews. If tools are
unavailable, report the missing route rather than silently substituting it.

## Prompt template

```text
Context: <project + phase and relevant files>
Question: <specific claim, failure mode, or design fork>
Current evidence: <compact red/green/diff evidence>
Reviewer trigger reason: <reason>
Constraints:
- DISCUSS/read-only; no shell, edits, publication, network, or credentials
- be concise; cite inspected paths when relevant
Return: findings, acceptance concern, and stop/go.
```

Report the job/thread identity, active/bound model, trigger reason, and verdict.
Keep usage observational; do not add fixed token or price semantics.
