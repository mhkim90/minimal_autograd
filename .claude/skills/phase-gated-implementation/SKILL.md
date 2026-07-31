---
name: phase-gated-implementation
description: Use when Claude Code implements multi-phase work from a plan, PR description, issue, design doc, or approved checklist. Keeps Claude Code as the lightweight orchestrator and publication controller, routes bulk implementation to OpenCode's configured default, escalates implementation to Luna and focused review to Terra only with evidence, and uses Codex/Sol for independent high-risk read-only gating.
---

# Phase-Gated Implementation

Use this skill for work with phases, acceptance gates, review reports, or
explicit approval points. It applies to code, docs, experiments, migrations,
data pipelines, build systems, and UI work.

## Operating Principle

Claude Code orchestrates the phase. It reads the plan, defines and verifies the
red gate, checks scope and validation evidence, controls publication, reports,
and decides continue/stop. Claude Code does not implement by default.

This role does not change the `claude-delegate` contract: Claude invoked through
the Claude MCP is read-only. Interactive Claude Code may stage, commit, push,
and update a PR because those are publication duties, not implementation.

OpenCode is the implementation engine:

- OpenCode configured default: bulk L1 and routine L2 implementation.
- `openai/gpt-5.6-luna`: non-trivial implementation escalation, only with
  evidence.
- `openai/gpt-5.6-terra`: focused read-only review of important L2 and L3
  diffs.

Codex uses its current default Sol-class model as the strongest independent
read-only gate. Call `codex-delegate` in DISCUSS/read-only mode for important
L2 risk and every L3 phase. Do not use Codex EXECUTE as the normal implementer;
Luna is the implementation escalation path.

Native OpenCode `sol`/`terra`/`luna` triad agents are reserved for standalone
OpenCode operation, unavailable Codex tooling, quota pressure,
breakthrough/replan, or explicit user request. Do not add routine native
OpenCode Sol review on top of Codex/Sol.

An explicitly user-selected implementation or review model overrides automatic
routing for that role. Do not use fixed token percentages or output caps;
route from risk and evidence.

## Role Routing

Use OpenCode's configured default by omitting the `model` override. Escalate
implementation to `openai/gpt-5.6-luna` only when one or more hold:

- repeated focused failure on the same red gate
- the implementer reports uncertainty, broadens scope, or asks for guidance
- the diff is clearly weak: missing files, broken tests, or obvious regression
- the task is known non-trivial generation: protocol contracts, complex
  refactors, concurrency, numerical kernels, build/CUDA
- explicit user or approved-plan request

Use `openai/gpt-5.6-terra` for focused read-only review of important L2 and L3
diffs. Skip delegated review for trivial L1 changes.

Use Codex/Sol through `codex-delegate` in DISCUSS/read-only mode when one or
more hold:

- final level is 3
- architecture or API contracts change
- security, memory, concurrency, CUDA, or numerical correctness is material
- release or packaging behavior changes
- tests are weak, missing, flaky, or suspicious
- implementer and Terra disagree, or uncertainty repeats
- explicit user request

Avoid duplicate review. Terra checks the focused diff; Codex/Sol supplies the
independent high-risk gate. Claude verifies their evidence and does not repeat
the same deep review.

If OpenCode tools are unavailable, stop before implementation unless the user
explicitly authorizes a different engine. If Codex tools are unavailable, L1
or L2 phases that do not require the independent Sol gate may continue. Any
phase requiring that gate is Level 4 and stops before implementation or
publication; report the missing gate instead of silently replacing it with
Claude.

## Plan Audit Preflight

Before implementation, identify the plan artifact, its authoring source, and
the revision being executed.

- For an externally authored plan, design document, issue, PR description,
  decision record, migration outline, or checklist, run `plan-audit` once if
  there is no audit for the current material revision.
- For a plan drafted by the active agent, use `grilled-me` instead.
- Do not repeat `plan-audit` at every phase. Re-run it only when a material
  revision changes scope, contracts, sequencing, acceptance criteria, risk,
  rollback, or implementation strategy.
- An audit verdict of `blocked` is Level 4. Stop before implementation or
  before the affected phase.
- Preserve every manual stop named by `ready-with-manual-gates`.
- Import the audit's phase dependencies, gates, risks, stop rules, and
  downstream consumers into orientation and reporting.

Audit recommendations are not approvals. Candidate scope globs, gates,
sequencing changes, and risk levels remain drafts until the user or repository
owner approves them. A `ready` verdict may allow manual implementation to
start, but it cannot enable auto continuation unless the plan itself contains
approved per-phase scope globs and gates.

## Risk Levels

- **Level 1, low risk/default**: mechanical edits, docs/config, narrow bug
  fixes, routine test fixes, repetitive refactors. OpenCode default implements;
  Claude performs the local gate.
- **Level 2, medium risk**: important logic, non-trivial tests, broader scope,
  or implementer uncertainty. OpenCode default normally implements; Terra
  reviews important diffs; Codex/Sol gates when a trigger above applies.
- **Level 3, high risk**: architecture/API, memory, concurrency, build, CUDA,
  release, subtle numerical correctness, weak tests, suspicious red gates, or
  unexpected diff expansion. Luna implements only when engine evidence
  warrants it; Terra reviews; Codex/Sol read-only gate is required.
- **Level 4, stop**: invalid gate, plan-intent change, unavailable tooling,
  blocking unrelated changes, or unverifiable correctness.

## Standard Loop

1. **Orient Lightly**
   - Read the plan, PR, issue, or design doc.
   - Confirm the `plan-audit` or `grilled-me` preflight for the artifact and
     revision. Stop on a blocking or stale audit before the affected phase.
   - Confirm branch and dirty state.
   - At phase start, stop if `.claude/state/PAUSE` or `.codex/state/PAUSE`
     exists. Never remove either file automatically.
   - Identify the phase boundary, approved scope globs, acceptance gates, and
     stop rules.
   - Classify risk and select the implementation and review engines.

2. **Red Gate**
   - Run or add the smallest failing check that proves the phase is not done.
   - For behavior changes, show failure for the right reason, then success.
   - For docs/config/mechanical work, define an observable structural, lint,
     artifact, benchmark, or manual acceptance gate before editing.
   - Use one attempt cap: max 3 red-gate attempts unless the approved phase
     explicitly raises it.

3. **Delegate Implementation**
   - Use OpenCode configured default for bulk work.
   - Escalate to Luna only from the evidence rules above.
   - Provide cold context: exact task, approved scope, red gate, success
     criteria, constraints, stop rules, and no commit.
   - Require a concise report: changed files, decisions, commands, metrics,
     blockers, and attempts used.

4. **Claude Local Gate**
   - Review diff stat and key hunks.
   - Verify red/green evidence, test validity, and command output.
   - Run appropriate formatting, lint, and diff checks.
   - Confirm every changed file is inside the approved phase scope.
   - Do not turn risk escalation into direct implementation by Claude.

5. **Targeted Reviews**
   - Use Terra for focused read-only review of important L2 and L3 diffs.
   - Use Codex/Sol DISCUSS/read-only for triggered L2 and every L3 phase.
   - If the user names another review model, use that model for the named role.
   - Ask for blockers, spec mismatches, test validity, scope drift, and stop/go.
   - Treat reviews as evidence; Claude retains publication control.

6. **Commit + Report**
   - Evaluate step 7 immediately before committing so the trailer is known.
   - Stage only intended files; never use `git add -A` around unrelated work.
   - Commit one phase at a time with `Phase-gate: auto (L1)` or
     `Phase-gate: manual`.
   - Treat commit, push, and PR publication by default as an explicit
     repository-owner policy. The step-7 manual gate controls whether the next
     phase starts; it does not retroactively require approval to publish the
     completed phase.
   - Push the current branch by default.
   - Open or update the PR by default.
   - If the user explicitly prohibits commit, push, or PR actions, stop before
     the first prohibited action, report local state, and wait.
   - Report commit, scope, validation, deviations, engines/reviewers, verdicts,
     and next-phase state.

7. **Gate**

   Publication and continuation are separate. This gate decides whether the
   next phase starts.

   Degrade rule, evaluated first: if the approved plan lacks per-phase scope
   globs, automatic continuation is disabled. Drafted globs do not count until
   the owner/user approves them.

   Auto-continuation requires all of:

   - final level is 1
   - every changed file is inside the approved scope globs
   - the red gate failed for the right reason, then passed within the cap
   - plan deviations: none
   - every required review: no blocker

   Any violation forces a manual gate and explicit approval.

   Consecutive-auto limit: inspect the contiguous commit suffix before `HEAD`.
   Count commits whose `Phase-gate` trailer value begins with `auto`; the
   canonical value is `auto (L1)`. Stop at the first commit without that
   prefix. If the count is already 2, classify the current phase manual. An
   unexpected interleaved commit is scope drift and also forces a manual gate.
   Do not create a state counter.

## Stop Rules

Stop and report instead of weakening gates when:

- the required `plan-audit` or `grilled-me` preflight is missing, stale, or
  blocked
- an acceptance metric fails
- implementation must change plan intent
- a test expectation appears wrong without strong supporting evidence
- a required dependency, delegate, or validation tool is unavailable
- diff scope expands beyond the approved boundary
- an implementer exceeds scope
- unrelated user changes block safe work
- the red-gate attempt cap is reached
- a required Terra or Codex/Sol gate reports a blocker

## Delegation Prompts

Implementation prompt:

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <tests/metrics>
Constraints:
- Working dir: <path>
- Do not commit
- Do not edit outside scope
- Max red-gate attempts: <approved phase cap, default 3>
- Stop and report on <stop rules>
Final response: changed files, decisions, commands, metrics, blockers,
red-gate attempts used
```

Terra review prompt:

```text
Read-only focused review. Do not edit.
Context: <phase + changed files>
Validation: <red gate + passing checks>
Review scope: <diff summary + key hunks>
Question: any blocking scope, test-validity, acceptance, or bug issue?
Return findings first or "no blocker".
```

Codex/Sol gate prompt:

```text
DISCUSS mode. Read-only; do not edit.
Context: <phase + risk>
Read first: <key files>
Evidence: <diff stat + key hunks + validation>
Question: is there a blocking correctness, scope, or test-validity issue?
Return findings first or "no blocker"; cite file:line when useful.
```

## Phase Report Template

```text
Phase <N> complete: <commit or uncommitted state>

Publication:
- <commit/push/PR URL or stopped before prohibited action>

Plan preflight:
- artifact and audited revision: <identity + revision>
- review: <plan-audit / grilled-me>
- outcome: <ready / ready-with-manual-gates / blocked / grilled-me findings resolved>
- freshness: <current / stale>
- unresolved manual gates: <none or list>

Imported audit handoff:
- phase dependencies and preconditions: <none or list>
- downstream consumers: <none or list>
- applicable stop rules: <list>

Scope:
- approved scope globs: <list or missing - manual gate>
- changed files: <list>

Implementation:
- final level: <1 / 2 / 3>
- model: <OpenCode default / Luna / other>
- routing reason: <default or evidence>

Routine review:
- reviewer: <Terra / user-selected / none>
- verdict: <no blocker / blocker>

Independent gate:
- reviewer: <Codex/Sol / user-selected / unavailable / none>
- verdict: <no blocker / blocker>

Claude local review:
- <diff/test/evidence inspected>

Red gate:
- <failed for the right reason, then passed>
- attempts used / cap: <used> / <cap>

Validation:
- <commands and results>

Plan deviations:
- <none or rationale>

Gate: <auto - proceeding / waiting-for-approval>
```
