---
name: phase-gated-implementation
description: Use when implementing multi-phase work from a plan, PR description, issue, design doc, or user-approved checklist. Guides Codex to preserve TDD phase gates with adaptive risk levels: OpenCode handles most source reading, implementation, test iteration, and routine review; Codex remains the final scope/test/diff/commit gatekeeper; Claude/Sonnet or a user-specified OpenCode model provides targeted read-only second opinion when useful.
---

# Phase-Gated Implementation

Use this skill for implementation work that has phases, acceptance gates, review
reports, or explicit user approval points. Keep the process general: works for
code, docs, experiments, migrations, data pipelines, build systems, and UI work.

## Operating Principle

Default to OpenCode-heavy implementation. OpenCode spends tokens on source
exploration, red gates, mechanical edits, run-inspect-tweak loops, and routine
read-only review. Codex spends judgment on phase boundaries, TDD evidence,
final diff review, commits, pushes, PR/issue comments, and stop/go decisions.
Claude/Sonnet is scarce and reserved for targeted high-value blocker checks;
use a user-specified OpenCode model instead when requested. Second opinion is
read-only and advisory.

Use delegation to reduce duplicate context loading, not to reduce verification
quality. Do not outsource final accountability. Delegates produce patches or
reviews; Codex verifies and decides.

## Role Budgeting

- **Level 1, low risk / default**: OpenCode 80-90%, Codex 5-15%,
  Claude/Sonnet 0-5%. Use for mechanical edits, docs/config changes, narrow bug
  fixes, routine test fixes, and repetitive refactors.
- **Level 2, medium risk / Codex-expanded**: OpenCode 65-80%, Codex 15-25%,
  Claude/Sonnet 0-5%. Use when touched logic is important, tests are nontrivial,
  scope is broader than expected, or OpenCode reports uncertainty.
- **Level 3, high risk / Codex-reviewed gate**: OpenCode 45-65%, Codex 30-45%,
  Claude/Sonnet 0-10%. Use for architecture/API changes, memory, concurrency,
  build, CUDA, release behavior, subtle numerical correctness, weak tests,
  suspicious red gates, or unexpected diff expansion. OpenCode remains the
  plurality of token share even here; "Codex-reviewed" reflects Codex's
  judgment/gatekeeper role, not a token-share majority.
- **Level 4, stop**: stop and report when the gate is invalid, implementation
  must change plan intent, required tooling is unavailable, unrelated user
  changes block safe work, or correctness cannot be verified.

Within each level, OpenCode is the default implementer and must not commit.
Codex reads the plan, phase boundary, git status, diff stat, key hunks,
validation output, and risky files only as needed. If the user says Claude
quota is low/exhausted, asks to avoid Claude, or names an OpenCode model for
second opinion, use that exact model instead. Example: `opencode-go/glm-5.2`.

## Standard Loop

1. **Orient Lightly**
   - Read plan/PR/issue/design doc.
   - Confirm branch and dirty state.
   - Check `.codex/state/PAUSE` at phase start; if present, stop.
   - Identify phase boundary, scope globs, acceptance gates, and stop rules.
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

3. **Delegate Implementation**
   - Use OpenCode default model as the primary implementer.
   - Keep prompts narrow: phase context, files/scope, red gate, success
     criteria, constraints, stop rules, and no commit.
   - Let OpenCode read source, edit, run tests, and iterate within scope.
   - Require a compact final report: changed files, key decisions, commands run,
     metrics, and blockers.

4. **Codex Gate Review**
   - Review diff stat and key hunks before trusting delegate output.
   - Verify red/green evidence, test validity, and command output.
   - Run formatting/lint/diff checks appropriate to repo.
   - Confirm diff scope matches phase.
   - Read deeper source, raise the level, or drive directly when risk escalation
     rules apply.

5. **Second Opinion**
   - Ask OpenCode by default for routine read-only blocker review.
   - Use Claude/Sonnet only for architecture/API risk, weak or suspicious tests,
     subtle correctness risk, delegate uncertainty, delegate/reviewer
     disagreement, or explicit user request.
   - If the user notified quota limits, asked to avoid Claude, or specified an
     OpenCode model for second opinion, ask OpenCode with that exact model.
   - Ask focused questions: blockers, spec mismatches, test validity, scope,
     and stop/go.
   - Treat advice as input, not authority.

6. **Commit + Report**
   - After steps 2-5 have produced all gate evidence, evaluate the step-7
     criteria immediately before committing so the correct trailer is known.
   - Stage only intended files. Never `git add -A` when unrelated files exist.
   - Commit one phase at a time with the trailer `Phase-gate: auto (L1)` or `Phase-gate: manual`.
   - Push current branch by default.
   - Open or update the PR by default.
   - If the user explicitly prohibits commit, push, or open/update PR actions, stop before the first prohibited action, report local state, and wait.
   - Leave PR/issue comment with:
     - commit hash
     - scope
     - validation commands and key metrics
     - deviations from plan and rationale
     - level used and Codex review scope
     - second-opinion result
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
   - second opinion: no blocker

   Any violation: manual gate — stop and wait for explicit approval.

   Consecutive auto-approve rule: before classifying the current phase auto,
   inspect the contiguous commit suffix immediately preceding `HEAD`. Count
   commits carrying `Phase-gate: auto` trailers until the first commit without
   one. If the count reaches 2, classify the current phase manual and wait. An
   unexpected interleaved commit is scope drift and independently forces a
   manual gate. Do not create a state counter for this.

   Kill switch: if `.codex/state/PAUSE` exists at phase start, stop. This file
   is never removed automatically.

## Risk Escalation

Escalate one or more levels when:

- architecture or API contracts change
- security, data loss, concurrency, memory, build, packaging, or release
  behavior is involved
- tests are weak, missing, flaky, or suspicious
- numerical correctness risk appears
- OpenCode reports uncertainty, blockers, or broadens scope
- unexpected files change
- the red gate is missing or fails for the wrong reason
- second opinion finds a blocker
- delegate and reviewer disagree
- diff scope expands beyond phase boundary
- Codex cannot verify quality from diff/test evidence without reading more
  context

Escalation means spending more Codex tokens on direct source/test inspection.
For high risk, Codex may drive targeted fixes directly. For blocking risk,
stop instead of weakening the gate.

## Stop Rules

Stop and report instead of weakening gates when:

- core acceptance metric fails
- implementation must change plan intent
- test expectation appears wrong but fix is not defensible from plan facts
- required dependency/tool is unavailable
- diff scope expands beyond phase boundary
- OpenCode implementation exceeds requested scope
- existing unrelated user changes block safe work
- red-gate attempt cap reached; report red-gate attempt history instead of
  weakening the gate. Codex decides whether to raise the level or stop; the
  gate is never weakened.

If a test gate is wrong, fix only when evidence is strong. Document the
correction in the plan/report so future readers do not revert it.

## Delegation Prompts

Mechanical edit prompt should include:

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
- Max red-gate attempts: 3
- Stop and report if <stop rules>
Final response (concise): changed files, key decisions, commands run, metrics,
blockers, red-gate attempts used
```

Second-opinion prompt should include:

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

Scope:
- approved scope globs: <list or "no per-phase scope declared - manual gate">
- changed files: <list>

Implementation:
- final level: <1 low / 2 medium / 3 high>
- model: <OpenCode default or other>
- Codex review scope: <diff stat/key hunks/source files inspected>
- red-gate attempts used / cap: <used> / <cap, default 3>

Red gate:
- `<command>`: <failing result before implementation, then passing result after>
- evidence: failed for the right reason, then passed within the red-gate attempt cap

Validation:
- `<command>`: <result>
- key metrics: ...

Plan deviations:
- <none or rationale>

Second opinion:
- <OpenCode routine review and optional Claude/Sonnet or named OpenCode model escalation>: <verdict; no blocker?>

Gate: <auto - proceeding to Phase N+1 / waiting-for-approval>
```
