---
name: phase-gated-implementation
description: Use when implementing multi-phase work from a plan, PR description, issue, design doc, or user-approved checklist. Guides Codex to preserve TDD phase gates while using OpenCode as the primary implementer, Codex as final gatekeeper for scope/test/diff/commit decisions, and Claude or a user-specified OpenCode model for read-only second opinion before each phase report.
---

# Phase-Gated Implementation

Use this skill for implementation work that has phases, acceptance gates, review
reports, or explicit user approval points. Keep the process general: works for
code, docs, experiments, migrations, data pipelines, build systems, and UI work.

## Operating Principle

Default to OpenCode-led implementation. Codex owns judgment, phase boundaries,
TDD evidence, final diff review, commits, pushes, PR/issue comments, and
stop/go decisions. OpenCode owns source exploration, mechanical edits, and
run-inspect-tweak loops. Second opinion is read-only and advisory.

Use delegation to reduce duplicate context loading, not to reduce verification
quality. Do not outsource final accountability. Delegates produce patches or
reviews; Codex verifies and decides.

## Role Budgeting

- **Codex/GPT-5.5 gatekeeper**: read the plan, phase boundary, git status,
  diff stat, key hunks, validation output, and risky files only when needed.
  Expand context or drive directly when risk or ambiguity requires it.
- **OpenCode default implementer**: read source, add or run the red gate,
  perform edits, run tests, iterate, and return a compact report. Do not commit.
- **Second opinion reviewer**: use Claude MCP by default. If the user says
  Claude quota is low/exhausted, asks to avoid Claude, or names an OpenCode
  model, use OpenCode with that exact model instead. Example:
  `opencode-go/glm-5.2`.

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

4. **Codex Gate Review**
   - Review diff stat and key hunks before trusting delegate output.
   - Verify red/green evidence, test validity, and command output.
   - Run formatting/lint/diff checks appropriate to repo.
   - Confirm diff scope matches phase.
   - Read deeper source or drive directly when risk escalation rules apply.

5. **Second Opinion**
   - Ask Claude MCP by default for read-only review/advice.
   - If user notified quota limits, asked to avoid Claude, or specified an
     OpenCode model, ask OpenCode with that exact model.
   - Ask focused questions: blockers, spec mismatches, test validity, scope,
     and stop/go.
   - Treat advice as input, not authority.

6. **Commit + Report**
   - Stage only intended files. Never `git add -A` when unrelated files exist.
   - Commit one phase at a time.
   - Push current branch.
   - Leave PR/issue comment with:
     - commit hash
     - scope
     - validation commands and key metrics
     - deviations from plan and rationale
     - second-opinion result
     - explicit next phase / waiting-for-approval state

7. **Wait**
   - Stop after report if user requested per-phase approval.
   - Do not start next phase until approval is given.

## Risk Escalation

Codex must expand context, inspect source directly, or drive implementation
directly when:

- architecture or API contracts change
- security, data loss, concurrency, memory, build, packaging, or release
  behavior is involved
- tests are weak, missing, flaky, or suspicious
- OpenCode reports uncertainty, blockers, or broadens scope
- second opinion finds a blocker
- delegate and reviewer disagree
- diff scope expands beyond phase boundary
- Codex cannot verify quality from diff/test evidence without reading more
  context

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

Second-opinion prompt should include:

```text
Read-only review. Do not edit.
Context: <phase + changed files>
Validation: <red gate + passing commands + metrics>
Review scope: <diff summary + key hunks or files>
Question: any blocking issue before commit? Focus on <risks>.
Return findings first, concise.
```

## Phase Report Template

```text
Phase <N> complete and pushed: <commit>

Scope:
- ...

Implementation:
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
- <Claude MCP or OpenCode model>: <verdict>

Waiting for approval before Phase <N+1>.
```
