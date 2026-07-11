---
name: opencode-delegate
description: Delegate tedious, mechanical, or long-running work to OpenCode via the mcp__opencode tools. Use for applying precise edits, sweeps, run-inspect-tweak loops, test runs, and background mechanical work. Do not use for ambiguous design decisions, deep reasoning, or hard implementation without a clear plan.
---

# OpenCode Delegate

Use `mcp__opencode.opencode_run_async` by default. Use blocking `mcp__opencode.opencode_run` only for trivial, known-short prompts where losing the result would be acceptable. Reviews, implementation, iteration, and test-running should use async even when they look quick.

OpenCode is a tedious-work delegate. Use it when the work can be specified precisely and checked with clear success criteria.

## Tool Availability

If `mcp__opencode` tools are not already visible, use `tool_search` with a query such as `opencode session run delegate` to expose them. If the tools are unavailable, fall back to inline execution and tell the user.

Common tools:

- `mcp__opencode.opencode_run_async` — default for delegated work; returns a job ID immediately.
- `mcp__opencode.opencode_session_fork_async` — reset long context in the background.
- `mcp__opencode.opencode_job_status` / `mcp__opencode.opencode_job_result` — poll and fetch results.
- `mcp__opencode.opencode_job_list` / `mcp__opencode.opencode_job_cancel` — discover or stop jobs.
- `mcp__opencode.opencode_run` / `mcp__opencode.opencode_session_fork` — blocking; only for trivial, known-short calls.
- `mcp__opencode.opencode_session_new` / `mcp__opencode.opencode_session_list` — create or find sessions.

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
- Working dir: <repo-root>
- Never commit unless explicitly told
- Match existing code style
- Keep changes small and surgical
```

## Session Management

- New task: omit `session_id` on `opencode_run_async`, or call `opencode_session_new` first.
- Continue the same task: pass the previous `session_id`.
- Find a session: call `opencode_session_list`.
- Long context: fork with `opencode_session_fork_async`, poll the job result, and continue on the new session ID.

## Model selection

Pass `model="provider/model"` to `opencode_run`, `opencode_run_async`, or
`opencode_session_fork_async` to override OpenCode's default model for that call.
Omit `model` to use the configured default.

## Async workflow

For delegated work:

1. Start with `mcp__opencode.opencode_run_async`, or `mcp__opencode.opencode_session_fork_async` when resetting context.
2. Poll with `mcp__opencode.opencode_job_status`.
3. Fetch the final response with `mcp__opencode.opencode_job_result`.
4. Use `mcp__opencode.opencode_job_list` to recover or discover jobs; `scope="all"` includes jobs recorded by other repository MCP instances.
5. Use `mcp__opencode.opencode_job_cancel` only when the job should stop.

Async job metadata is recorded under `/tmp/opencode_mcp/jobs`. Live response buffers remain local to the MCP process, but discovery and cancellation can work across repository MCP instances.

## Timeout

- Blocking `opencode_run` and `opencode_session_fork` retain their timeout arguments, but they are not the normal path.
- Use blocking tools only for trivial, known-short calls, with a small explicit timeout.
- For anything uncertain, start async first instead of retrying after a client timeout.
- A client may abort a blocking call before the server timeout fires, losing the result.

## After Delegation

1. Report the job ID immediately, and the session ID once the result contains it
2. Summarize what OpenCode was asked to do.
3. Relay results concisely when OpenCode returns.
4. Surface blockers or questions before continuing.
