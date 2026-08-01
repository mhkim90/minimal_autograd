---
name: phase-gated-implementation
description: Use for multi-phase work from a plan, PR description, issue, design doc, or approved checklist. Keep Claude Code as read-only orchestrator and publication controller, use OpenCode agent="luna" for normal role-bound implementation, the configured default for mechanical or economy work, and agent="terra" for focused read-only review.
---

# Phase-Gated Implementation

Use this skill for work with phases, acceptance gates, review reports, or
explicit approval points. Apply it to code, docs, experiments, migrations,
data pipelines, build systems, and UI work.

## Operating Principle

Claude Code orchestrates the phase. Read the plan, define and verify the red
gate, inspect scope and validation evidence, control publication, report, and
decide continue/stop. Do not implement by default.

This role does not change the `claude-delegate` contract: Claude invoked through
the Claude MCP is read-only. Interactive Claude Code may stage, commit, push,
and update a PR only as publication duties.

Use OpenCode as the implementation engine:

- Use `agent="luna"` for normal role-bound implementation. Omit `model` and
  `variant` so the named agent configuration remains authoritative.
- Omit `agent`, `model`, and `variant` for mechanical, repetitive, economy, or
  quota-preserving work; record the configured-default route explicitly.
- Use `agent="terra"` for focused read-only review in a separate session.
- Use a raw `model="provider/model"` only for an explicit user override or an
  explicitly approved degraded fallback. Restate role constraints and report
  `routing mode: model fallback` for that fallback.

Use Codex/Sol through `codex-delegate` in DISCUSS/read-only mode for important
L2 risk and every L3 phase. Do not use Codex EXECUTE as the normal implementer.
Do not routinely nest OpenCode Sol beneath Codex/Sol. Use `agent="sol"` only
for an explicit whole-phase triad handoff; do not also call Luna or Terra in
that mode.

Fail closed when the MCP schema lacks `agent`, a requested agent is unavailable,
or live evidence cannot prove the selected role. Never silently substitute the
configured default or a raw model for a requested named agent. Stop and report
unless an approved fallback is selected explicitly.

Keep user-selected implementation or review models as explicit overrides. Do
not put fixed prices, token shares, or output caps in this skill.

## Role Routing

Classify the work before selecting a route:

- **Level 1, low risk/default**: mechanical edits, docs/config, narrow fixes,
  routine tests, or repetitive refactors. Use the configured default and have
  Claude perform the local gate.
- **Level 2, medium risk**: important logic, non-trivial tests, broader scope,
  or uncertainty. Use `agent="luna"` for implementation and `agent="terra"`
  for an important focused review; use Codex/Sol when its independent gate is
  triggered.
- **Level 3, high risk**: architecture/API, memory, concurrency, build,
  release, CUDA, numerical correctness, weak tests, suspicious red gates, or
  unexpected diff expansion. Use `agent="luna"` when role-bound implementation
  is needed, `agent="terra"` for focused review, and Codex/Sol for the required
  independent read-only gate.
- **Level 4, stop**: invalid gates, plan-intent changes, unavailable tooling,
  blocking unrelated changes, or unverifiable correctness.

Escalate implementation to `agent="luna"` when the work is non-trivial or when
the implementer reports uncertainty, repeats a focused failure, broadens scope,
or produces a weak diff. Do not switch to a raw Luna model. Use the configured
default directly for the explicit Level 1 mechanical/economy route.

Use Codex/Sol independently when final level is 3, architecture/API,
security, memory, concurrency, CUDA, numerical, release, weak-test,
disagreement, uncertainty, or explicit-request conditions apply. Avoid
duplicate review: Terra checks the focused diff, while Codex/Sol supplies the
independent gate.

If OpenCode tools are unavailable, stop before implementation unless the user
explicitly authorizes another engine. If Codex tools are unavailable, continue
only for L1/L2 phases that do not require the independent gate; stop Level 4 on
phases that do.

## Session isolation

Keep one session lineage per role. Start Terra in a fresh session; never reuse
or convert a Luna session for review. Continue or fork a role only with the
same agent argument on every call:

```text
mcp__opencode__opencode_run_async(
    session_id="ses_xxx",
    agent="luna",
    message="Continue the Luna implementation with <next task>"
)

mcp__opencode__opencode_session_fork_async(
    session_id="ses_xxx",
    agent="terra",
    message="Continue the Terra focused review"
)
```

Record requested agent, session ID, and bound/reported model. Do not treat the
requested agent string alone as proof of role identity.

## Plan Audit Preflight

Identify the plan artifact, authoring source, and revision before implementation.

- For an externally authored plan, run `plan-audit` once when no audit exists
  for the current material revision.
- For a plan drafted by the active agent, use `grilled-me` instead.
- Do not repeat an audit unless scope, contracts, sequencing, acceptance,
  risk, rollback, or implementation strategy materially changes.
- Stop before the affected phase on a blocked or stale audit, and preserve all
  manual stops from a `ready-with-manual-gates` result.
- Import audit dependencies, gates, risks, stop rules, and consumers into the
  phase report. Candidate scope globs and recommendations remain unapproved
  until the owner approves them.

## Risk Levels

Use the same four levels and routing above. Treat implementation complexity,
test strength, scope, and engine evidence as risk signals; do not use fixed
token shares or prices.

## Standard Loop

1. **Orient Lightly**
   - Read the plan, issue, PR, or design document.
   - Confirm the plan-audit or grilled-me preflight and revision.
   - Confirm branch and dirty state.
   - Stop at phase start if `.claude/state/PAUSE` or `.codex/state/PAUSE` exists;
     never remove either file automatically.
   - Identify phase boundary, approved scope globs, acceptance gates, and stop
     rules; classify risk and select the route.

2. **Red Gate**
   - Run or add the smallest failing check proving the phase is not done.
   - For behavior changes, show the right failure before implementation and
     success afterward.
   - For docs/config/mechanical work, define a structural, lint, artifact,
     benchmark, or manual acceptance gate before editing.
   - Use one cap: at most 3 red-gate attempts unless the approved phase raises it.

3. **Delegate Implementation**
   - Use `mcp__opencode__opencode_run_async` with `agent="luna"` for normal
     role-bound implementation and omit `model` and `variant`.
   - Omit all three selectors for the explicit configured-default
     mechanical/economy route.
   - Repeat the agent on continuations and forks, and keep each role in its own
     session lineage.
   - Use `agent="sol"` only for the explicit whole-phase handoff; do not make
     separate Luna or Terra calls in that mode.
   - Provide cold context: exact task, scope, red gate, success criteria,
     constraints, stop rules, and no commit.
   - Require changed files, decisions, commands, metrics, blockers, requested
     agent, session ID, and bound/reported model.

4. **Claude Local Gate**
   - Inspect diff stat and key hunks.
   - Verify red/green evidence, test validity, command output, formatting, lint,
     and diff checks.
   - Confirm every changed file is inside the approved phase scope.
   - Do not turn risk escalation into direct implementation by Claude.

5. **Targeted Reviews**
   - Start a fresh `mcp__opencode__opencode_run_async` session with
     `agent="terra"` for important L2/L3 focused read-only review; omit
     `model` and `variant`.
   - Continue or fork that Terra lineage with `agent="terra"` repeated.
   - Use `codex-delegate` in DISCUSS/read-only mode for triggered L2 and every
     L3 phase; preserve Codex/Sol as the independent gate.
   - Ask for blockers, plan mismatches, test validity, scope drift, and stop/go.
   - Treat review evidence as input; Claude retains publication control.

6. **Commit + Report**
   - Evaluate step 7 immediately before committing so the trailer is known.
   - Stage only intended files; never use `git add -A` around unrelated work.
   - Commit one phase at a time with `Phase-gate: auto (L1)` or
     `Phase-gate: manual`.
   - Treat commit, push, and PR publication as repository-owner policy. The
     step-7 gate controls continuation, not retroactive publication approval.
   - Push the branch and open or update the PR by default.
   - If the user prohibits commit, push, or PR actions, stop before that action
     and report local state.
   - Report commit, scope, validation, deviations, route, requested agent,
     session ID, bound/reported model, reviewers, verdicts, and next-phase state.

7. **Gate**
   - Publication and continuation are separate; this gate decides whether the
     next phase starts.
   - If approved per-phase scope globs are missing, disable auto-continuation
     and wait for approval.
   - Auto-continuation requires final Level 1, in-scope files, the correct red
     failure then pass within the cap, no plan deviations, and no required
     review blocker.
   - Otherwise apply a manual gate and wait for explicit approval.
   - Inspect the contiguous commit suffix before `HEAD`; after two consecutive
     `Phase-gate: auto` commits, classify the current phase manual. Treat an
     unexpected interleaved commit as scope drift; do not create a counter.

## Stop Rules

Stop and report instead of weakening gates when the preflight is missing, stale,
or blocked; an acceptance metric fails; implementation changes plan intent;
tests are indefensible; a required tool, delegate, or named-agent proof is
unavailable; scope expands; unrelated work blocks safety; the red-gate cap is
reached; or a required Terra or Codex/Sol gate reports a blocker.

## Delegation Prompts

Implementation prompt:

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <tests/metrics>
Requested agent: luna, unless this is explicitly configured-default mechanical/economy work
Routing mode: <role-bound | mechanical/economy | approved fallback>
Constraints:
- Working dir: <path>
- Do not commit
- Do not edit outside scope
- Max red-gate attempts: <approved cap, default 3>
- Stop and report on <stop rules>
Final response: changed files, decisions, commands, metrics, blockers,
requested agent, session ID, bound/reported model, routing mode
```

Focused Terra prompt:

```text
Read-only focused review. Do not edit.
Requested agent: terra; use a fresh Terra session and omit model and variant.
Context: <phase + changed files>
Validation: <red gate + passing checks>
Review scope: <diff summary + key hunks>
Question: any blocking scope, test-validity, acceptance, or bug issue?
Return findings first or "no blocker".
Report session ID and bound/reported model.
```

Whole-phase exception:

```text
Use agent="sol" for this whole phase only because <explicit whole-phase handoff approval>.
Do not call Luna or Terra separately. Report requested agent, session ID,
bound/reported model, evidence, and routing mode.
```

## Phase Report Template

```text
Phase <N> complete: <commit or uncommitted state>

Publication:
- <commit/push/PR URL or stopped before prohibited action>

Plan preflight:
- artifact and audited revision: <identity + revision>
- review: <plan-audit / grilled-me>
- outcome: <ready / ready-with-manual-gates / blocked / findings resolved>
- freshness: <current / stale>
- unresolved manual gates: <none or list>

Imported audit handoff:
- dependencies and preconditions: <none or list>
- downstream consumers: <none or list>
- applicable stop rules: <list>

Scope:
- approved scope globs: <list or missing - manual gate>
- changed files: <list>

Implementation:
- final level: <1 / 2 / 3>
- route: <agent="luna" | configured default | agent="sol" | user override | model fallback>
- requested agent: <value>
- session ID: <value>
- bound/reported model: <value>
- routing reason: <default or evidence>

Routine review:
- requested agent: <agent="terra" | none>
- session ID: <value>
- bound/reported model: <value>
- verdict: <no blocker | blocker>

Independent gate:
- reviewer: <Codex/Sol | user-selected | unavailable | none>
- verdict: <no blocker | blocker>

Claude local review:
- <diff stat, key hunks, and validation evidence inspected>

Red gate:
- <failed for the right reason, then passed>
- attempts used / cap: <used> / <cap>

Validation:
- <commands and results>
- key metrics: <values>

Plan deviations:
- <none or rationale>

Gate: <auto - proceeding / waiting-for-approval>
```
