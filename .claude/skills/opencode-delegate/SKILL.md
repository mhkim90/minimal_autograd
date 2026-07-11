---
name: opencode-delegate
description: Delegate TEDIOUS / mechanical / long-running work to OpenCode via async-first OpenCode MCP tools. Use for applying precise edits, sweeps, run→inspect→tweak loops, and test runs.
---

# OpenCode Delegate

Use `mcp__opencode__opencode_run_async` by default. Use blocking `mcp__opencode__opencode_run` only for trivial, known-short prompts where losing the result would be acceptable. Reviews, implementation, iteration, and test-running should use async even when they look quick.

OpenCode is the **tedious-work engine**: Claude manages/orchestrates, OpenCode executes
mechanical jobs. It runs a light LLM — excellent at executing precise instructions, weak
at judgment, so give it fully-specified, mechanical tasks.

OpenCode MCP tools:

- `mcp__opencode__opencode_run_async` — default for delegated work; returns a job ID immediately.
- `mcp__opencode__opencode_session_fork_async` — reset long context in the background.
- `mcp__opencode__opencode_job_status` / `mcp__opencode__opencode_job_result` — poll and fetch results.
- `mcp__opencode__opencode_job_list` / `mcp__opencode__opencode_job_cancel` — discover or stop jobs.
- `mcp__opencode__opencode_run` / `mcp__opencode__opencode_session_fork` — blocking; only for trivial, known-short calls.
- `mcp__opencode__opencode_session_new` / `mcp__opencode__opencode_session_list` — create or find sessions.

## When to delegate

Delegate when the task is any of:
- **Precise mechanical edit** — apply an exact, fully-specified change to files
- **Long CLI run** — build sweeps, multi-config test runs, benchmark loops
- **Tedious iteration** — run → inspect → tweak → repeat cycles with a known plan
- **Detailed agreed plan** — user approved a step-by-step plan; execution is mechanical
- **Parallel workstream** — background work while main conversation handles something else

Do NOT delegate to OpenCode:
- Hard reasoning, architecture/design decisions, or ambiguous tasks (resolve those in
  conversation first, or handle inline)
- Single-step commands (just run them directly)

## How to write the prompt

Include everything OpenCode needs to work cold:

```
Context: <1-2 sentences on what this project is and what we're doing>

Task: <exact steps to perform>

Files to read first: <list key files if needed>

Success criteria:
- <what done looks like>
- <any output to produce or check, e.g. which test binary must print ALL TESTS PASSED>

  Constraints:
    - Working dir: <repo-root>
- Never commit unless explicitly told
- Match existing code style
```

## Session management

- **New task** → omit `session_id` on `opencode_run_async` (starts fresh), or call
  `opencode_session_new` first to reserve an ID before sending work.
- **Continue same task** → pass the `session_id` returned from the previous call.
- **Find a session** → `opencode_session_list` returns recent session IDs + titles
  (use when you've lost the ID from an earlier turn).
- The response prefix `[session:ses_xxx]` carries the session ID — store it if you plan to follow up.

## Model selection

Pass `model="provider/model"` to `opencode_run`, `opencode_run_async`, or
`opencode_session_fork_async` to override OpenCode's default model for that call.
Omit `model` to use the configured default.

## Context reset (fork)

When a session grows long and context quality degrades, fork it:

```
mcp__opencode__opencode_session_fork_async(
    session_id="ses_xxx",
    message="Summarize what we've done so far and continue with <next task>"
)
```

- Returns a `[job:job_id]` immediately; poll the result for the new `[session:ses_yyy]` and use that session ID for subsequent async calls
- The fork snapshots the old session; the new session starts fresh from that point
- Original session is preserved and unaffected

## Async workflow

For delegated work:

1. Start with `mcp__opencode__opencode_run_async`, or `mcp__opencode__opencode_session_fork_async` when resetting context.
2. Poll with `mcp__opencode__opencode_job_status`.
3. Fetch the final response with `mcp__opencode__opencode_job_result`.
4. Use `mcp__opencode__opencode_job_list` to recover or discover jobs; `scope="all"` includes jobs recorded by other repository MCP instances.
5. Use `mcp__opencode__opencode_job_cancel` only when the job should stop.

Async job metadata is recorded under `/tmp/opencode_mcp/jobs`. Live response buffers remain local to the MCP process, but discovery and cancellation can work across repository MCP instances.

## Timeout

- Blocking `opencode_run` and `opencode_session_fork` retain their timeout arguments, but they are not the normal path.
- Use blocking tools only for trivial, known-short calls, with a small explicit timeout.
- For anything uncertain, start async first instead of retrying after a client timeout.
- A client may abort a blocking call before the server timeout fires, losing the result.

## After delegation

1. Report the job ID immediately, and the session ID once the result contains it
2. Summarize what OpenCode was asked to do
3. When OpenCode returns, relay its findings concisely — don't just dump the full response
4. If OpenCode hits a blocker or asks a question, surface it to the user before continuing
