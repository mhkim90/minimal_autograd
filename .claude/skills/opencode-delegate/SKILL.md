---
name: opencode-delegate
description: Delegate approved, tedious, or long-running work through async-first OpenCode MCP. Route mechanical work to the configured default, standard work to Luna, and justified difficult preflight to Sol-expert.
---

# OpenCode Delegate

Use `mcp__opencode__opencode_run_async` by default; use blocking calls only for
known-short work. Delegate execution, iteration, review, and long test runs,
not ambiguous design decisions or single-step commands.

## Routes

- **Mechanical/economy**: omit `agent`, `model`, and `variant`; use the
  configured default and report that route explicitly.
- **Standard**: use `agent="luna"`; omit `model` and `variant`.
- **Difficult**: use `agent="sol-expert"` only for a bounded preflight or
  breakthrough when warranted, then use `agent="luna"` for implementation.
- Use `agent="terra"` only for a separate, fresh-context, read-only review
  with a recorded reason; never use Terra as implementer.
- Use `agent="sol"` only for explicit whole-phase triad compatibility. Do not
  call Luna or Terra separately in that route.
- Use a raw model only for an explicit user override or approved fallback.

## Route evidence

- Record requested agent, job ID, session ID, and bound/reported model.
- Accept an honored named selector plus a terminal role response as minimum
  route evidence. Query normalized usage after terminal state when available.
- Report unresolved normalized model identity or partial accounting as an
  observability warning when no evidence contradicts the selected role.
- Stop on selector rejection, mismatched continuations, explicit model
  contradiction, or silent fallback. Never substitute silently.

## Luna and Sol-expert lifecycle

- Keep one bounded phase or subphase per Luna session and end it after green.
- Start or fork a new same-role session after a material scope, red-gate,
  strategy, or blocker change. Repeat the same named agent on continuations.
- Allow at most three implementation/fix attempts. After two failures on the
  same blocker, request one bounded Sol-expert consultation. Permit a third
  attempt only after a materially revised approach is agreed.
- Give Sol-expert approved scope, one focused question or blocker, and compact
  diff/test evidence. Allow one consultation and at most one follow-up.
  Require: findings, proposed approach, acceptance gate, stop/go. Sol-expert
  never implements, edits, publishes, or delegates.

## Prompt template

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Routing: <configured default | luna | sol-expert | terra | sol>
Elapsed-time checkpoint / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; no commit or edits outside scope
- stop after two same-blocker failures; revise materially before a third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```

## Async workflow and elapsed-time policy

1. Keep one session lineage per role. Repeat the same named agent on every
   continuation or fork, and pass a stable workflow ID.
2. Poll job status and fetch the terminal result. Query normalized usage after
   terminal state.
3. At each phase-defined elapsed-time checkpoint, recover status and inspect
   material progress. Never duplicate a running job because it is slow.
4. After two no-progress checkpoints or the maximum wait, stop and report.
   Do not silently retry, extend, or cancel; cancellation requires an explicit
   stop decision.
5. Use job list to recover recorded jobs. Record session count, retries,
   elapsed time, checkpoints, route evidence, and reviewer trigger reason.

Usage is observational evidence only. Do not add fixed token, price, or
provider-private-path limits.
