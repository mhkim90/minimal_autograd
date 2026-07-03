---
name: opencode-delegate
description: Delegate tedious, mechanical, or long-running work to OpenCode via the mcp__opencode tools. Use for applying precise edits, sweeps, run-inspect-tweak loops, test runs, and background mechanical work. Do not use for ambiguous design decisions, deep reasoning, or hard implementation without a clear plan.
---

# OpenCode Delegate

OpenCode is a tedious-work delegate. Use it when the work can be specified precisely and checked with clear success criteria.

## Tool Availability

If `mcp__opencode` tools are not already visible, use `tool_search` with a query such as `opencode session run delegate` to expose them. If the tools are unavailable, fall back to inline execution and tell the user.

Common tools:

- `mcp__opencode.opencode_run`: send a prompt, optionally with `session_id`, `model`, and `timeout`.
- `mcp__opencode.opencode_run_async`: start a background run and poll with job tools.
- `mcp__opencode.opencode_session_new`: create a fresh session.
- `mcp__opencode.opencode_session_list`: list recent sessions.
- `mcp__opencode.opencode_session_fork`: fork a session to reset or compact context.
- `mcp__opencode.opencode_job_result`: fetch a completed async job result.

## When to Delegate

Delegate when the task is:

- A precise mechanical edit with exact files and behavior.
- A long build, test loop, benchmark, or repeated experiment.
- A run -> inspect -> tweak loop with a known plan.
- A user-approved step-by-step plan where execution is mechanical.
- A parallel background workstream that does not block immediate local work.

Do not delegate to OpenCode:

- Ambiguous tasks or design decisions.
- Deep reasoning or hard implementation without a clear plan.
- Unicode-dense files where exact symbols matter.
- Single-step commands that can be run directly.

## Prompt Template

Give OpenCode enough context to work cold:

```text
Context: This repository is a minimal reverse-mode automatic differentiation library in C++17, built on Eigen3.

Task: <exact steps to perform>

Files to read first: <list key files if needed>

Success criteria:
- <what done looks like>
- <commands to run or outputs to check>

Constraints:
- Working dir: /home/mhkim90/workspace/minimal_autograd
- Never commit unless explicitly told
- Match existing code style
- Keep changes small and surgical
```

## Session Management

- New task: omit `session_id` on `opencode_run`, or call `opencode_session_new` first.
- Continue the same task: pass the previous `session_id`.
- Find a session: call `opencode_session_list`.
- Long context: fork with `opencode_session_fork` and continue on the returned session ID.

## Model selection

Pass `model="provider/model"` to `opencode_run`, `opencode_run_async`, or
`opencode_session_fork` to override OpenCode's default model for that call.
Omit `model` to use the configured default.

## Timeout

- Default to `timeout=3600` for long builds, test loops, sweeps, and multi-file edits.
- Use a lower timeout only for clearly quick tasks.
- Use `opencode_run_async` when a blocking tool call may exceed the MCP client or proxy limit.

## After Delegation

1. Report the session ID or job ID to the user.
2. Summarize what OpenCode was asked to do.
3. Relay results concisely when OpenCode returns.
4. Surface blockers or questions before continuing.
