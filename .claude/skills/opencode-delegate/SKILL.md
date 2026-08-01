---
name: opencode-delegate
description: Delegate precise, long-running, or plan-driven work through async-first OpenCode MCP tools. Use agent="luna" for normal role-bound implementation, the configured default for mechanical or economy work, and agent="terra" for focused read-only review.
---

# OpenCode Delegate

Use `mcp__opencode__opencode_run_async` by default. MCP clients may cap
blocking calls before OpenCode finishes, and delegated jobs can run for many
minutes. Use blocking `mcp__opencode__opencode_run` only for trivial,
known-short prompts. Use async for implementation, review, iteration, and test
runs.

Delegated runs can access only the caller's working directory. External paths,
including `/tmp`, are denied; `/tmp/opencode_mcp` is MCP-server-managed
registry storage.

## When to delegate

Delegate:

- **Long CLI run** — experiment scripts, training loops, or sweeps
- **Tedious iteration** — run → inspect → tweak → repeat work with a known plan
- **Detailed agreed plan** — execution of an approved step-by-step plan
- **Parallel workstream** — background work while the main conversation proceeds

Do not delegate design decisions, ambiguous tasks, work requiring unavailable
conversation history, or a single-step command that can be run directly.

## Agent routing

- Use `agent="luna"` as the normal role-bound implementation route. Omit
  `model` and `variant`; the named agent configuration is authoritative.
- Use the OpenCode configured default by omitting `agent`, `model`, and
  `variant` for mechanical, repetitive, economy, or quota-preserving work.
  Record this as the configured-default route; do not imply that it is Luna.
- Use `agent="terra"` for focused read-only review in a separate Terra session.
  Omit `model` and `variant`; never use Terra as an implementer.
- Use a raw `model="provider/model"` only for an explicit user override or an
  explicitly approved degraded fallback. In a fallback, restate the complete
  role constraints and report `routing mode: model fallback`.
- Use `agent="sol"` only for an explicit whole-phase triad handoff. Do not
  routinely nest OpenCode Sol beneath Codex/Sol. In a whole-phase handoff, do
  not also call Luna or Terra; verify the returned evidence once.

Fail closed when agent selection is missing from the MCP schema, unavailable,
or not proven by live evidence. Never silently substitute the configured
default or a raw model for a requested named agent. Stop and report the
missing or unproven route unless an approved fallback is selected explicitly.

## How to write the prompt

Include everything OpenCode needs to work cold:

```text
Context: <1-2 sentences on the project and task>

Task: <exact steps to perform>

Files to read first: <key files if needed>

Success criteria:
- <what done looks like>
- <outputs or checks to run>

Routing:
- requested agent: <luna | terra | sol | none>
- routing mode: <role-bound | mechanical/economy | review | whole-phase | user override | model fallback>
- model/variant: <omitted for named agents; otherwise explicit override>

Constraints:
- Working dir: <absolute caller working directory>
- Never commit unless explicitly told
- Match existing code style
- Do not edit outside the stated scope

Final response: changed files, decisions, commands, metrics, blockers,
requested agent, session ID, bound/reported model, and routing mode
```

## Session management

- Start a new task without `session_id`.
- Keep one session lineage per role. Never reuse or convert a Luna session for
  Terra review, or a Terra session for implementation.
- Continue the same role with its returned `session_id` and repeat the same
  named `agent` on every run or fork:

```text
mcp__opencode__opencode_run_async(
    session_id="ses_xxx",
    agent="luna",
    message="Continue the Luna implementation with <next task>"
)
```

- Fork a role lineage with the same agent:

```text
mcp__opencode__opencode_session_fork_async(
    session_id="ses_xxx",
    agent="luna",
    message="Summarize the work and continue the Luna implementation"
)
```

Do not pass `model` or `variant` on named-agent calls. Capture the returned
session ID and the bound/reported model; requested agent text alone is not
identity evidence.

## Async workflow

1. Start with `mcp__opencode__opencode_run_async`, or fork with
   `mcp__opencode__opencode_session_fork_async` when resetting context.
2. Poll with `mcp__opencode__opencode_job_status`.
3. Fetch the final response with `mcp__opencode__opencode_job_result`.
4. Use `mcp__opencode__opencode_job_list` to recover jobs, including
   `scope="all"` for jobs recorded by other repository MCP instances.
5. Use `mcp__opencode__opencode_job_cancel` only when the job should stop.

## Focused review prompt

Start Terra review in a fresh session and use:

```text
Read-only focused review. Do not edit.
Requested agent: terra
Context: <phase and changed files>
Validation: <red gate, passing checks, and metrics>
Review scope: <diff summary and key hunks>
Question: any blocking scope, test-validity, acceptance, or bug issue?
Return findings first or "no blocker".
Report session ID and bound/reported model.
```

## Timeout and reporting

- Keep known-short blocking calls small and explicit.
- Use async first for uncertain work instead of retrying after a client timeout.
- Report the job ID immediately and the session ID once returned.
- Summarize the request and relay findings concisely.
- Surface blockers or questions before continuing.
