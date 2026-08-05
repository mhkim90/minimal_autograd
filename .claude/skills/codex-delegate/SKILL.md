---
name: codex-delegate
description: Delegate independent high-risk reasoning, adversarial review, and blocker gates from Claude Code to Codex Terra. Use DISCUSS/read-only by default. Use EXECUTE only when the user explicitly requests Codex implementation or approves it as a fallback.
---

# Codex Delegate

Use `mcp__codex__codex` to start a session and `mcp__codex__codex-reply` to
continue it. Always set `cwd` to the caller's absolute working directory.

Terra is the intended independent Codex review model, not an assumption about
the configured default. For a phase-gated Terra review, select the supported
Terra model explicitly until runtime metadata proves the configured default is
Terra. Record the requested model, thread ID, and active model. Stop on an
explicit mismatch; do not trust a generic self-label over invocation/runtime
metadata.

## Modes

| Mode | `sandbox` | `approval-policy` | Use |
|---|---|---|---|
| **Discuss** | `read-only` | `never` | Default independent blocker gate, adversarial review, root-cause reasoning, or design fork |
| **Execute** | `workspace-write` | `never` | Explicit user request or user-approved fallback only |

Claude Code orchestrates and controls publication. OpenCode's configured
default handles mechanical work, Luna implements standard/difficult work, and
Sol-expert supplies bounded difficult preflight or breakthrough reasoning.
Codex Terra supplies one triggered independent review; do not duplicate it
with routine OpenCode Terra review.

## Discuss prompt

```text
Read-only independent review. Do not edit.
Context: <project + phase>
Read first: <smallest relevant files and diff>
Validation: <red gate and passing checks>
Trigger reason: <architecture/security/CUDA/etc.>
Question: <specific correctness or blocker question>
Return findings first or "no blocker", then stop/go.
```

## Execute prompt

```text
Context: <project and approved fallback>
Task: <exact bounded implementation>
Files/scope: <approved paths>
Red gate and success criteria: <checks>
Constraints:
- working directory: <absolute path>
- do not commit or edit outside scope
```

## Thread, timing, and evidence

- Start a fresh thread for a new review. Continue only the same role/task with
  `codex-reply`; start fresh when context becomes stale.
- Declare an elapsed-time checkpoint and maximum wait. At each checkpoint,
  recover status before any retry. Never duplicate a slow running review.
- After two no-progress checkpoints, record a wait-policy review; do not stop
  or cancel solely for that condition. Stop only when the declared maximum
  wait is reached or the controller makes an explicit stop decision.
- Report mode, requested model, active model, thread ID, elapsed time, findings,
  and any evidence warning. Missing usage is observational; an explicit model
  mismatch is blocking.
