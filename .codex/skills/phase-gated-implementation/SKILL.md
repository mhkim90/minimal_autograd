---
name: phase-gated-implementation
description: Use when implementing multi-phase work from a plan, PR description, issue, design doc, or user-approved checklist. Guides Codex to preserve TDD phase gates with adaptive risk levels: OpenCode handles most source reading, implementation, test iteration, and routine review; Codex remains the final scope/test/diff/commit gatekeeper; Claude/Sonnet is reserved for targeted high-value blocker checks.
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
Claude/Sonnet is scarce and reserved for targeted high-value blocker checks.

Use delegation to reduce duplicate context loading, not to reduce verification
quality. Do not outsource final accountability. Delegates produce patches or
reviews; Codex verifies and decides.

## Role Budgeting

- **Level 1, low risk / default**: OpenCode 75-90%, Codex 10-20%,
  Claude/Sonnet 0-5%. Use for mechanical edits, docs/config changes, narrow bug
  fixes, routine test fixes, and repetitive refactors.
- **Level 2, medium risk / Codex-expanded**: OpenCode 60-75%, Codex 25-35%,
  Claude/Sonnet 0-5%. Use when touched logic is important, tests are nontrivial,
  scope is broader than expected, or OpenCode reports uncertainty.
- **Level 3, high risk / Codex-heavy gate**: OpenCode 40-60%, Codex 35-55%,
  Claude/Sonnet 0-10%. Use for architecture/API changes, memory, concurrency,
  build, CUDA, release behavior, subtle numerical correctness, weak tests,
  suspicious red gates, or unexpected diff expansion.
- **Level 4, stop**: stop and report when the gate is invalid, implementation
  must change plan intent, required tooling is unavailable, unrelated user
  changes block safe work, or correctness cannot be verified.

## Standard Loop

1. **Orient Lightly**
   - Read plan/PR/issue/design doc.
   - Confirm branch and dirty state.
   - Identify phase boundary, expected files, acceptance gates, and stop rules.
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

4. **Routine Second Opinion**
   - Ask OpenCode by default for read-only blocker review.
   - Prefer a separate OpenCode pass or model when available.
   - Focus on scope drift, invalid tests, missed acceptance criteria, and
     obvious bug risk.
   - Treat advice as input, not authority.

5. **Codex Gate Review**
   - Review diff stat and key hunks before trusting delegate output.
   - Verify red/green evidence, test validity, and command output.
   - Run formatting/lint/diff checks appropriate to repo.
   - Confirm diff scope matches phase.
   - Read deeper source, raise the level, or drive directly when risk escalation
     rules apply.

6. **Targeted Claude/Sonnet Escalation**
   - Do not use Claude/Sonnet by default.
   - Use it only for architecture/API risk, weak or suspicious tests, subtle
     correctness risk, delegate uncertainty, delegate/reviewer disagreement, or
     explicit user request.
   - Keep the prompt compact and ask for blocking issues only.

7. **Commit + Report**
   - Stage only intended files. Never `git add -A` when unrelated files exist.
   - Commit one phase at a time.
   - Push current branch.
   - Leave PR/issue comment with:
     - commit hash
     - scope
     - validation commands and key metrics
     - deviations from plan and rationale
     - level used and Codex review scope
     - second-opinion result
     - explicit next phase / waiting-for-approval state

8. **Wait**
   - Stop after report if user requested per-phase approval.
   - Do not start next phase until approval is given.

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
- Stop and report if <stop rules>
Final response: changed files, key decisions, commands run, metrics, blockers
```

OpenCode routine review prompt should include:

```text
Read-only review. Do not edit.
Context: <phase + changed files>
Validation: <red gate + passing commands + metrics>
Review scope: <diff summary + key hunks or files>
Question: any blocking issue before commit? Focus on scope drift, test validity,
missed acceptance criteria, and obvious bug risk.
Return findings first, concise.
```

Claude/Sonnet escalation prompt should be smaller:

```text
Read-only blocker check. Do not edit.
Context: <phase + risk>
Evidence: <diff stat + key hunk summary + validation result>
Question: is there a blocking issue? Return only findings or "no blocker".
```

## Phase Report Template

```text
Phase <N> complete and pushed: <commit>

Scope:
- ...

Implementation:
- level: <1 low / 2 medium / 3 high>
- model: <OpenCode default or other>
- Codex review scope: <diff stat/key hunks/source files inspected>

Red gate:
- `<command>`: <failing result before implementation>

Validation:
- `<command>`: <result>
- key metrics: ...

Plan deviations:
- <none or rationale>

Second opinion:
- <OpenCode routine review and optional Claude/Sonnet escalation>: <verdict>

Waiting for approval before Phase <N+1>.
```
