---
name: claude-delegate
description: Delegate bounded second opinions, adversarial review, planning critique, and concise reasoning checks to Claude through the mcp__claude tools. Use when Codex should ask Claude for an independent perspective, review a plan, stress-test an assumption, or run a small read-only reasoning pass. Do not use for mechanical edits, live shell work, or tasks that require Claude filesystem access.
---

# Claude Delegate

Use Claude as a bounded reasoning delegate through this repository's MCP server. The wrapper runs Claude Code print mode with read-only Claude tools (`Read,Glob,Grep`) and `dontAsk` permissions, so treat Claude as a reviewer that may inspect files but must not execute commands or edit anything.

## Tool Availability

If `mcp__claude` tools are not already visible, use `tool_search` with a query such as `claude mcp claude_run delegate` to expose them. If the tools are unavailable, fall back to inline reasoning and tell the user.

Common tools:

- `mcp__claude.claude_run`: send a prompt and wait for the completed response.
- `mcp__claude.claude_run_async`: start a background Claude run and poll with job tools.
- `mcp__claude.claude_job_status`: check a background job, optionally including the response.
- `mcp__claude.claude_job_result`: fetch a completed background result.
- `mcp__claude.claude_job_cancel`: cancel a running job.

## When to Delegate

Delegate to Claude when the task benefits from a second model's perspective:

- Review a plan, root-cause analysis, or design choice before implementation.
- Stress-test assumptions, risks, edge cases, or failure modes.
- Ask for a concise alternate approach to a difficult reasoning problem.
- Get a read-only critique of MCP tool schemas, prompt contracts, or user-facing behavior.
- Run an independent review after Codex has drafted a solution.

Do not delegate to Claude:

- Mechanical edits, sweeps, build loops, or long test runs; use `$opencode-delegate` when appropriate.
- Tasks where Claude needs shell, network, credentials, editing, or non-read-only project-specific tool access.
- Single-step commands that Codex can run directly.
- Ambiguous decisions that should be resolved with the user first.

## Prompt Template

Give Claude cold-context instructions and ask for a verdict, not a broad summary:

```text
Context: This repository builds a Claude MCP server for Codex. The relevant files are <files or snippets>.
Question: <specific claim, plan, failure mode, or design fork to evaluate>.
I currently lean <option>. Argue the strongest case against it, then state which option you would pick and why.
Constraints:
- Be concise.
- Do not assume filesystem or shell access.
- Cite provided file paths or snippets when relevant.
```

For most calls, prefer:

- `model: "sonnet"`
- `effort: "low"` or `"medium"` for quick reviews; `"high"` only for hard reasoning.
- omit `max_budget_usd` for normal second opinions and repo reviews; the server default is `0.50`.
- use `max_budget_usd: 0.05` to `0.10` only for very small, bounded questions.
- use an explicit higher cap up to the server hard cap when the user asks for deeper work.
- `mode: "safe"` unless there is a clear reason to use `"bare"`.

## Async Runs

Use `claude_run_async` only when the review may exceed the MCP client or proxy blocking limit. Poll with `claude_job_status` or `claude_job_result`, then cancel if the task becomes irrelevant.

## After Delegation

1. State that Claude was consulted through MCP.
2. Summarize the question asked and Claude's verdict.
3. Highlight any disagreement with Codex's current plan.
4. Continue with Codex-owned decisions and verification; do not treat Claude's response as authoritative without checking it.
