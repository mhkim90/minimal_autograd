---
name: opencode-delegate
description: Delegate TEDIOUS / mechanical / long-running work to OpenCode via MCP (opencode_run + session tools). Use for applying precise edits, sweeps, run→inspect→tweak loops, and test runs.
---

# OpenCode Delegate

OpenCode is the **tedious-work engine**: Claude manages/orchestrates, OpenCode executes
mechanical jobs. It runs a light LLM — excellent at executing precise instructions, weak
at judgment, so give it fully-specified, mechanical tasks.

OpenCode MCP tools:
- `mcp__opencode__opencode_run` — send a prompt (optionally `session_id`, `model`, `timeout`).
- `mcp__opencode__opencode_session_new` — create a fresh session, returns its ID.
- `mcp__opencode__opencode_session_list` — list recent sessions (IDs + titles).
- `mcp__opencode__opencode_session_fork` — fork a session to reset/compact context.

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
    - Working dir: /home/mhkim90/workspace/minimal_autograd
- Never commit unless explicitly told
- Match existing code style
```

## Session management

- **New task** → omit `session_id` on `opencode_run` (starts fresh), or call
  `opencode_session_new` first to reserve an ID before sending work.
- **Continue same task** → pass the `session_id` returned from the previous call.
- **Find a session** → `opencode_session_list` returns recent session IDs + titles
  (use when you've lost the ID from an earlier turn).
- The response prefix `[session:ses_xxx]` carries the session ID — store it if you plan to follow up.

## Model selection

Pass `model="provider/model"` to `opencode_run`, `opencode_run_async`, or
`opencode_session_fork` to override OpenCode's default model for that call.
Omit `model` to use the configured default.

## Context reset (fork)

When a session grows long and context quality degrades, fork it:

```
mcp__opencode__opencode_session_fork(
    session_id="ses_xxx",
    message="Summarize what we've done so far and continue with <next task>"
)
```

- Returns a new `[session:ses_yyy]` — use this ID for all subsequent `opencode_run` calls
- The fork snapshots the old session; the new session starts fresh from that point
- Original session is preserved and unaffected

## Timeout

- **Default: pass `timeout=3600`** (1h) on every `opencode_run` call — build/test sweeps
  and multi-file edits routinely exceed the tool's built-in 300s.
- Short, clearly-quick tasks may use a lower value, but when in doubt use 3600.
- No timeout: pass `timeout=null`
- If you see `[OpenCode error: timed out]`, retry with a higher timeout

## After delegation

1. Report the session_id to the user so they can reference it
2. Summarize what OpenCode was asked to do
3. When OpenCode returns, relay its findings concisely — don't just dump the full response
4. If OpenCode hits a blocker or asks a question, surface it to the user before continuing
