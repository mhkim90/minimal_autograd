---
name: claude-delegate
description: Delegate bounded second opinions, adversarial review, planning critique, and concise reasoning checks to Claude through the mcp__claude tools. Use when Codex should ask Claude for an independent perspective, review a plan, stress-test an assumption, or run a small read-only reasoning pass. Do not use for mechanical edits or tasks requiring shell, network, credentials, filesystem writes, or tools beyond read-only file inspection.
---

# Claude Delegate

Use Claude as a bounded reasoning delegate through this repository's MCP server. The wrapper runs Claude Code print mode with read-only Claude tools (`Read,Glob,Grep`) and `dontAsk` permissions, so treat Claude as a reviewer that may inspect files but must not execute commands or edit anything.

## Tool Availability

If `mcp__claude` tools are not already visible, use `tool_search` with a query such as `claude mcp claude_run delegate` to expose them. If the tools are unavailable, fall back to inline reasoning and tell the user.

Common tools:

- `mcp__claude.claude_run_async`: default for meaningful Claude delegation; start a background run and poll with job tools.
- `mcp__claude.claude_run`: blocking; only for trivial, known-short prompts where losing the result is acceptable.
- `mcp__claude.claude_job_status`: check a background job, optionally including the response.
- `mcp__claude.claude_job_result`: fetch a completed background result.
- `mcp__claude.claude_job_list`: discover recorded jobs, including jobs started by another repo-local MCP server instance.
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
Context: This repository is <project and task context>. The relevant files are <files or snippets>.
Question: <specific claim, plan, failure mode, or design fork to evaluate>.
I currently lean <option>. Argue the strongest case against it, then state which option you would pick and why.
Constraints:
- Be concise.
- Use `Read`, `Glob`, and `Grep` for relevant accessible project files when useful.
- Do not assume shell, editing, network, credential, or filesystem-write access.
- Cite inspected file paths or provided snippets when relevant.
```

For most calls, prefer:

- `model: "sonnet"`
- `effort: "low"` or `"medium"` for quick reviews; `"high"` only for hard reasoning.
- omit `max_budget_usd` for normal second opinions and repo reviews; the server default is `2.00`.
- use `max_budget_usd: 0.05` to `0.10` only for very small, bounded questions.
- use an explicit higher cap up to the server hard cap (`5.00`) when the user asks for deeper work.
- `mode: "safe"` unless there is a clear reason to use `"bare"`.

## Async Runs

Use `claude_run_async` as the default for meaningful reviews, adversarial checks, planning critique, and non-trivial reasoning. Poll with `claude_job_status` or `claude_job_result`, then cancel if the task becomes irrelevant.

Use `claude_job_list` to discover recorded jobs after losing a job ID or crossing MCP server instances. Final result retrieval remains most reliable from the instance that started the job.

Use blocking `claude_run` only for trivial, known-short prompts where losing the result is acceptable if the client aborts.

## After Delegation

1. State that Claude was consulted through MCP.
2. Summarize the question asked and Claude's verdict.
3. Highlight any disagreement with Codex's current plan.
4. Continue with Codex-owned decisions and verification; do not treat Claude's response as authoritative without checking it.
