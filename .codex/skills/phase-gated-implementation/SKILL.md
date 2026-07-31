---
name: phase-gated-implementation
description: Use when implementing multi-phase work from a plan, PR description, issue, design doc, or approved checklist. Keeps Codex/Sol as final gatekeeper, routes bulk implementation to OpenCode's configured default, escalates implementation to Luna and focused review to Terra only with evidence, and reserves Claude/Sonnet for scarce independent read-only review.
---

# Phase-Gated Implementation

Use this skill for implementation work that has phases, acceptance gates, review
reports, or explicit user approval points. Keep the process general: works for
code, docs, experiments, migrations, data pipelines, build systems, and UI work.

## Operating Principle

Codex/Sol owns the gate. Codex reads the plan, classifies risk, runs the red
gate, reviews the final diff and validation output, commits, pushes, opens or
updates the PR/issue, and decides continue/stop. Codex does not implement by
default; it verifies and decides.

OpenCode is the bulk implementer. The OpenCode configured default model handles
source exploration, mechanical edits, run-inspect-tweak loops, and routine L1
and L2 implementation. Codex defines and verifies the red gate. Delegates
produce patches or reviews; Codex verifies and decides.

Claude/Sonnet is scarce, independent, and read-only. Claude does not implement.
Use Claude only for architecture/API, security, memory/concurrency/CUDA,
numerical, release, suspicious tests, uncertainty/disagreement, or explicit
request. Respect user overrides: if the user says Claude quota is low/exhausted,
or asks to avoid Claude, skip Claude and keep the normal non-Claude route. If
the user names an OpenCode model for second opinion, use that exact model
instead. Example: `opencode-go/glm-5.2`.

Native OpenCode `sol`/`terra`/`luna` triad ids are reserved for standalone
OpenCode operation, Codex quota pressure, breakthrough/replan, or explicit
user request. Never nest routine OpenCode Sol under Codex/Sol.

An explicit user-selected implementation or review model overrides automatic
routing for that role.

Use delegation to reduce duplicate context loading, not to reduce verification
quality. Do not outsource final accountability.

## Role Routing

Default implementation engine: OpenCode configured default model (omit the
`model` override). Treat this as the bulk implementer for L1 and routine L2.

Escalate implementation to `openai/gpt-5.6-luna` only with evidence:

- repeated focused failure on the same red gate
- implementer reports uncertainty, broadens scope, or asks for guidance
- diff is clearly weak (missing files, broken tests, obvious regressions)
- task is known non-trivial code generation (protocol contracts, complex
  refactors, concurrency, numerical kernels, build/CUDA)
- explicit user or approved-plan request

Use `openai/gpt-5.6-terra` for focused read-only review of important L2 and
L3 diffs. It does not replace Claude and is not a routine substitute for Codex
judgment. Skip routine review on trivial L1 patches to avoid duplicate work.

Use Claude/Sonnet for scarce, independent, read-only escalation when one or
more apply:

- architecture or API contract risk
- security, memory, concurrency, CUDA, or numerical correctness risk
- release or packaging behavior risk
- weak, missing, flaky, or suspicious tests
- implementer and reviewer disagree, or repeated uncertainty
- explicit user request

Avoid duplicate delegated review. After `terra` reviews a diff, do not ask a
second routine OpenCode reviewer to repeat it. Codex's final gate review remains
mandatory. Claude is an independent escalation, not a routine duplicate.

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

Levels describe scope and Codex effort, not fixed token shares.

- **Level 1, low risk / default**: mechanical edits, docs/config, narrow bug
  fixes, routine test fixes, repetitive refactors. OpenCode default implements.
  Codex reviews diff/gate only.
- **Level 2, medium risk / Codex-expanded**: touched logic matters, tests are
  nontrivial, scope is broader than expected, or implementer reports
  uncertainty. OpenCode default implements; `openai/gpt-5.6-terra` reviews
  important L2 diffs.
- **Level 3, high risk / Codex-reviewed gate**: architecture/API changes,
  memory, concurrency, build, CUDA, release behavior, subtle numerical
  correctness, weak tests, suspicious red gates, or unexpected diff expansion.
  Escalate implementation to `openai/gpt-5.6-luna` only when evidence warrants
  it; otherwise OpenCode default implements and Codex reads source/tests
  directly to verify.
- **Level 4, stop**: stop and report when the gate is invalid, implementation
  must change plan intent, required tooling is unavailable, unrelated user
  changes block safe work, or correctness cannot be verified.

## Standard Loop

1. **Orient Lightly**
   - Read plan/PR/issue/design doc.
   - Confirm the `plan-audit` or `grilled-me` preflight for the artifact and
     revision. Stop on a blocking or stale audit before the affected phase.
   - Confirm branch and dirty state.
   - At phase start, stop if `.codex/state/PAUSE` or `.claude/state/PAUSE`
     exists. Never remove either file automatically.
   - Identify phase boundary, scope globs, acceptance gates, and stop rules.
   - Classify the risk level and pick the implementation engine.
   - Avoid broad source loading until a risk signal justifies it.
   - Tell user a concise proceed plan if asked or if next step is risky.

2. **Red Gate**
   - Run or add the smallest failing check that proves the phase is not done.
   - For code behavior changes, show the check failing for the right reason
     before implementation, then passing after implementation.
   - For docs/config/mechanical phases, define an alternate observable gate
     before editing: structural check, link check, lint, expected artifact
     check, benchmark threshold, or manual acceptance evidence.
   - Do not skip the red gate unless no meaningful gate is practical; document
     the alternate evidence in the phase report.
   - Apply exactly one attempt cap: max 3 red-gate attempts. Do not introduce a
     second loop cap.

3. **Delegate Implementation**
   - Use OpenCode configured default model as the bulk implementer for L1 and
     routine L2.
   - Escalate to `openai/gpt-5.6-luna` only with the evidence above.
   - Keep prompts narrow: phase context, files/scope, red gate, success
     criteria, constraints, stop rules, and no commit.
   - Let the implementer read source, edit, run tests, and iterate within scope.
   - Require a compact final report: changed files, key decisions, commands
     run, metrics, and blockers.

4. **Codex Gate Review**
   - Review diff stat and key hunks before trusting delegate output.
   - Verify red/green evidence, test validity, and command output.
   - Run formatting/lint/diff checks appropriate to the repository.
   - Confirm diff scope matches the approved phase.
   - Read deeper source or drive targeted fixes directly when escalation rules
     apply.

5. **Targeted Reviews**
   - Use `openai/gpt-5.6-terra` for focused read-only review of important L2
     and L3 diffs.
   - Skip routine review on trivial L1 patches to avoid duplicate work.
   - Ask Claude/Sonnet only for scarce, independent, read-only escalation when
     the routing policy above lists it.
   - If the user signals Claude quota pressure or asks to avoid Claude, skip
     Claude and keep the normal non-Claude route.
   - If the user names an OpenCode model for second opinion, use that exact
     model instead.
   - Ask focused questions: blockers, spec mismatches, test validity, scope,
     and stop/go.
   - Treat advice as input, not authority.

6. **Commit + Report**
   - After steps 2-5 have produced all gate evidence, evaluate the step-7
     criteria immediately before committing so the correct trailer is known.
   - Stage only intended files. Never `git add -A` when unrelated files exist.
   - Commit one phase at a time with the trailer `Phase-gate: auto (L1)` or `Phase-gate: manual`.
   - Treat commit, push, and PR publication by default as an explicit
     repository-owner policy. The step-7 manual gate controls whether the next
     phase starts; it does not retroactively require approval to publish the
     completed phase.
   - Push current branch by default.
   - Open or update the PR by default.
   - If the user explicitly prohibits commit, push, or open/update PR actions,
     stop before the first prohibited action, report local state, and wait.
   - Leave PR/issue comment with:
     - commit hash
     - scope
     - validation commands and key metrics
     - deviations from plan and rationale
     - level used, implementation model, routine reviewer, independent
       reviewer, and routing/escalation reason
     - explicit next phase / waiting-for-approval state

7. **Gate**

   Publication and continuation are distinct: a successful phase may already be
   published by step 6. This gate decides whether the next phase starts.

   Degrade rule (evaluated first): if the approved plan lacks per-phase scope
   globs, automatic continuation is disabled and the loop waits for explicit
   user approval. An agent may draft scope globs, but they do not count as
   approved until the owner/user approves them.

   Auto continuation — proceed to the next phase without waiting only when all hold:
   - final level == 1 (any escalation raises the level, so an escalated phase
     cannot auto-continue)
   - every changed file falls inside the phase's approved scope globs
   - the red gate failed for the right reason, then passed, within the red-gate
     attempt cap (max 3 by default; an approved phase override may raise it)
   - plan deviations: none
   - every required routine or independent review: no blocker

   Any violation: manual gate — stop and wait for explicit approval.

   Consecutive auto-approve rule: before classifying the current phase auto,
   inspect the contiguous commit suffix immediately preceding `HEAD`. Count
   commits whose `Phase-gate` trailer value begins with `auto`; the canonical
   value is `auto (L1)`. Stop at the first commit without that prefix. If the
   count reaches 2, classify the current phase manual and wait. An unexpected
   interleaved commit is scope drift and independently forces a manual gate. Do
   not create a state counter for this.

   Kill switch: if `.codex/state/PAUSE` or `.claude/state/PAUSE` exists at
   phase start, stop. These files are never removed automatically.

## Risk Escalation

Escalate one or more levels when:

- architecture or API contracts change
- security, data loss, concurrency, memory, build, packaging, or release
  behavior is involved
- tests are weak, missing, flaky, or suspicious
- numerical correctness risk appears
- the implementer reports uncertainty, blockers, or broadens scope
- unexpected files change
- the red gate is missing or fails for the wrong reason
- routine or independent review finds a blocker
- implementer and reviewer disagree
- diff scope expands beyond phase boundary
- Codex cannot verify quality from diff/test evidence without reading more
  context

Escalation means spending more Codex tokens on direct source/test inspection
and/or switching the implementation engine to `openai/gpt-5.6-luna` only when
the engine-evidence rules above apply. For high risk, Codex may drive
targeted fixes directly. For blocking risk, stop instead of weakening the
gate.

## Stop Rules

Stop and report instead of weakening gates when:

- the required `plan-audit` or `grilled-me` preflight is missing, stale, or
  blocked
- core acceptance metric fails
- implementation must change plan intent
- test expectation appears wrong but fix is not defensible from plan facts
- required dependency/tool is unavailable
- diff scope expands beyond phase boundary
- implementer exceeds requested scope
- existing unrelated user changes block safe work
- red-gate attempt cap reached; report red-gate attempt history instead of
  weakening the gate. Codex decides whether to raise the level or stop; the
  gate is never weakened.

If a test gate is wrong, fix only when evidence is strong. Document the
correction in the plan/report so future readers do not revert it.

## Delegation Prompts

Mechanical edit prompt (OpenCode default or `openai/gpt-5.6-luna`) should
include:

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <paths or boundaries>
Red gate: <command/check + expected failure>
Success criteria: <tests/metrics>
Constraints:
- Working dir: <path>
- Do not commit
- Do not edit outside <paths/scope>
- Max red-gate attempts: <approved phase cap, default 3>
- Stop and report if <stop rules>
Final response (concise): changed files, key decisions, commands run, metrics,
blockers, red-gate attempts used
```

Focused review prompt (`openai/gpt-5.6-terra` or user-specified OpenCode
model):

```text
Read-only review. Do not edit.
Context: <phase + changed files>
Validation: <red gate + passing commands + metrics>
Review scope: <diff summary + key hunks or files>
Question: any blocking issue before commit? Focus on scope drift, test validity,
missed acceptance criteria, and obvious bug risk.
Return findings first, concise.
```

Targeted Claude/Sonnet or user-specified OpenCode escalation prompt should be
smaller:

```text
Read-only blocker check. Do not edit.
Context: <phase + risk>
Evidence: <diff stat + key hunk summary + validation result>
Question: is there a blocking issue? Return only findings or "no blocker".
```

## Phase Report Template

```text
Phase <N> complete: <commit or uncommitted local state>

Publication:
- <committed/pushed/PR URL or stopped before an explicitly prohibited action>

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
- approved scope globs: <list or "no per-phase scope declared - manual gate">
- changed files: <list>

Implementation:
- final level: <1 low / 2 medium / 3 high>
- model: <OpenCode configured default | openai/gpt-5.6-luna | other>
- routing/escalation reason: <why this engine; "default" if none>

Routine review:
- reviewer: <openai/gpt-5.6-terra | OpenCode default | none>
- verdict: <no blocker | blocker: ...>

Independent review:
- reviewer: <Claude/Sonnet | user-specified OpenCode model | none>
- verdict: <no blocker | blocker: ...>

Codex review scope:
- <diff stat/key hunks/source files inspected>

Red gate:
- `<command>`: <failing result before implementation, then passing result after>
- evidence: failed for the right reason, then passed within the red-gate attempt cap
- attempts used / cap: <used> / <cap, default 3>

Validation:
- `<command>`: <result>
- key metrics: ...

Plan deviations:
- <none or rationale>

Gate: <auto - proceeding to Phase N+1 / waiting-for-approval>
```
