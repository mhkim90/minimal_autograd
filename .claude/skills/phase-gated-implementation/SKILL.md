---
name: phase-gated-implementation
description: Use for phased work from a plan, issue, design, or approved checklist. Keep Claude Code as read-only orchestrator and publication controller, route implementation by difficulty, and use Codex Terra for triggered independent review.
---

# Phase-Gated Implementation

Use this skill for work with phases, acceptance gates, review reports, or
explicit approval points. Claude Code controls scope, gates, publication, and
the final stop/go decision; do not implement by default.

This role does not change the `claude-delegate` contract: Claude invoked
through Claude MCP is read-only. Interactive Claude Code may stage, commit,
push, and update a PR only as publication duties.

## Controller and routes

- Claude Code is the controller and publication gate. Record the active Claude
  controller model from runtime evidence rather than assuming it.
- Keep L1-L4 as safety and approval risk. Select implementation difficulty
  independently:
  - **Mechanical/economy**: repetitive, known edits or routine commands ->
    omit `agent`, `model`, and `variant` and use the configured OpenCode
    default.
  - **Standard**: locally understood non-trivial work -> `agent="luna"`.
  - **Difficult**: architecture-sensitive, unclear, numerical/CUDA, or
    repeated-blocker work -> use a bounded `agent="sol-expert"` preflight or
    breakthrough only when warranted, then `agent="luna"` implements.
- Risk selects review and approval. Difficulty selects implementer reasoning.
- Preserve `agent="sol"` as an explicit whole-phase triad compatibility route.
  In that route, do not call Luna or Terra separately.
- Keep user-selected models as explicit overrides. Never silently fall back to
  a configured default or raw model.

## Route evidence

- Record requested agent, job ID, session ID, and bound/reported model.
- Treat an accepted named-agent selector plus a terminal role response as the
  minimum route evidence. Query normalized provider usage when available.
- Treat missing normalized model resolution or partial usage as an
  observability warning, not a correctness blocker, when no evidence
  contradicts the returned role/model.
- Stop on a rejected selector, unavailable named agent, mismatched continuation,
  explicit model contradiction, or silent fallback.

## Review triggers

- Do not automatically launch OpenCode Terra for Claude-controlled L2 or L3
  work. Use it only as a fresh-context, read-only review for a recorded concern.
- Trigger one independent Codex Terra DISCUSS/read-only review for
  architecture/API compatibility, security, memory/concurrency, CUDA or
  numerical correctness, release behavior, weak tests, disagreement,
  suspicious red gates, or an explicit request.
- Use `codex-delegate`; require runtime evidence of Terra. Do not assume the
  current Codex default is Terra. One triggered review replaces a review stack.
- Claude Code remains the gate after review; review evidence is not publication
  authority.

## Session, retry, and elapsed-time controls

- Use one bounded phase or subphase per Luna session and end it after green.
  Start or fork a new session when scope, the red gate, strategy, or blocker
  changes materially.
- Allow at most three implementation/fix attempts. After two failures on the
  same blocker, stop blind retries and request one bounded `sol-expert`
  consultation. Permit a third attempt only after a materially revised
  approach is agreed.
- Give Luna a compact capsule: task, approved scope, relevant files, red gate,
  success criteria, commands, and stop rules.
- Give Sol-expert only that capsule, prior diff/test evidence, and one focused
  question. Allow one consultation and at most one follow-up. Require:
  findings, proposed approach, acceptance gate, stop/go. Sol-expert never
  implements, edits, publishes, or delegates.
- Declare an expected elapsed-time checkpoint and maximum wait policy for each
  async delegate/reviewer. At each checkpoint, recover status and inspect
  material progress; never launch a duplicate because a job is slow.
- After two consecutive checkpoints with no material progress, or at the
  phase-defined maximum wait, stop and report. Do not silently extend, retry,
  or cancel; cancellation requires an explicit stop decision.

## Plan preflight and standard loop

1. Read the plan and confirm its current audit or `grilled-me` preflight. Record
   dependencies, consumers, manual gates, and stop rules.
2. Confirm branch and dirty state. At phase start, check both
   `.claude/state/PAUSE` and `.codex/state/PAUSE`; never remove either
   automatically. Identify approved scope globs, acceptance gates, risk,
   implementation difficulty, attempt cap, and wait policy.
3. Run the smallest structural, lint, artifact, behavioral, or manual red
   gate. Use no more than three red-gate attempts unless approved otherwise.
4. Delegate implementation using the difficulty route above. Repeat the same
   named agent on continuations and forks; omit model and variant for named
   agents. Require changed files, decisions, commands, blockers, session
   identity, and bound/reported model.
5. Inspect the diff, red/green evidence, tests, formatting, lint, and scope.
   Run only the triggered independent review and record its reason. Do not turn
   escalation into direct implementation by Claude.
6. Evaluate the gate before any commit. Use `Phase-gate: auto (L1)` or
   `Phase-gate: manual` as the commit trailer. Stage only explicit intended
   paths; never use broad staging. Publish only as repository policy permits.
   Keep publication and continuation separate; if commit, push, or PR is
   prohibited, stop before it and report local state.
7. Inspect the contiguous commit suffix and its trailers. After two consecutive
   `Phase-gate: auto` trailers, force the current phase manual. Treat an
   unexpected interleaved commit as scope drift; use no counter state file.
8. If approved per-phase scope globs are missing, disable auto-continuation.
   Otherwise continue automatically only for final L1, in-scope changes, the
   right red failure followed by a pass within the cap, no plan deviation, and
   no review blocker. Apply a manual gate otherwise.

## Stop rules

Stop and report on stale, missing, or blocked preflight; an invalid or
wrong-reason gate; failed acceptance; plan-intent change; unverifiable
correctness; unavailable required tool, named agent, or reviewer evidence;
explicit route contradiction; two no-progress elapsed-time checkpoints; scope
expansion; blocking unrelated changes; or attempt-cap exhaustion.

## Delegation prompts

Implementation prompt:

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <configured default | agent="luna" | sol-expert then Luna>
Elapsed-time checkpoint / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; do not commit or edit outside scope
- max attempts: <approved cap>; stop after two same-blocker failures
- use a materially revised approach before any third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```

Sol-expert capsule:

```text
Bounded read-only consultation; do not implement, edit, publish, or delegate.
Scope: <approved scope>
Question: <difficult preflight question or repeated blocker>
Evidence: <compact diff/test evidence>
Return: findings; proposed approach; acceptance gate; stop/go.
```

Whole-phase compatibility exception:

```text
Use agent="sol" for this whole phase only because <explicit approval>.
The outer Claude Code controller retains final stop/go and publication
authority; Sol returns its diff and evidence without committing, pushing, or
creating a PR. Do not call Luna or Terra separately. Only if Sol returns a
repeated blocker may the outer controller invoke one bounded sol-expert
consultation. Report the active controller model, requested agent, job/session
identity, bound/reported model, and routing mode.
```

## Phase report

```text
Phase <N> complete: <commit or uncommitted state>
Publication: <published or stopped before prohibited action>
Plan preflight: <artifact/revision, audit, freshness, manual gates>
Scope: <approved globs>; changed files: <list>
Controller: Claude Code; active model: <runtime evidence>
Safety risk level: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <configured default | agent="luna" | sol-expert then Luna | agent="sol">
Implementation sessions / retries: <count> / <count>
Route evidence: <requested agent, job/session IDs, bound/reported model, warnings>
Elapsed time: <per role>; checkpoints: <count>; wait deviations: <none or list>
Reviewer trigger reason: <reason or none>
Independent reviewer: <Codex Terra | optional OpenCode Terra | none>
Independent verdict: <no blocker | blocker>
Red gate: <right failure then pass>; attempts: <used>/<cap>
Validation: <commands/results>; metrics: <values>
Usage: <workflow/model accounting and completeness warnings, if available>
Plan deviations: <none or rationale>
Gate: <auto-proceeding | waiting-for-approval>
```

Usage is observational evidence only. Do not introduce fixed token, price, or
provider-private-path budgets or semantics.
