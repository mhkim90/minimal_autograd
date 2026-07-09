---
name: phase-gated-implementation
description: Use when implementing multi-phase work from a plan, PR description, issue, design doc, or user-approved checklist. Guides Claude to preserve TDD phase gates with adaptive risk levels: OpenCode handles most source reading, implementation, test iteration, and routine review; Codex (via codex-delegate) absorbs escalation and hard reasoning as risk rises; Claude stays minimal-footprint — orchestration, verification, and the final scope/test/diff/commit decision only, since Claude's own token budget is the scarcest of the three engines.
---

# Phase-Gated Implementation

Use this skill for implementation work that has phases, acceptance gates, review
reports, or explicit user approval points. Keep the process general: works for
code, docs, experiments, migrations, data pipelines, build systems, and UI work.

## Operating Principle

Default to OpenCode-heavy implementation. OpenCode spends tokens on source
exploration, red gates, mechanical edits, run-inspect-tweak loops, and routine
read-only review. Claude spends judgment on phase boundaries, TDD evidence,
final diff review, commits, pushes, PR/issue comments, and stop/go decisions —
and nothing more, by default. Codex (via `codex-delegate`, DISCUSS mode) is
the engine that absorbs escalation as risk rises: hard reasoning, adversarial
review, numerical/architecture correctness. As risk increases, send *more
work to Codex*, not more direct inspection to Claude — see Token Budget below
for why.

Use delegation to reduce duplicate context loading, not to reduce verification
quality. Do not outsource final accountability. Delegates produce patches or
reviews; Claude verifies and decides.

## Token Budget

Relative available quota, approximate: **Claude 1x, Codex 5-8x, OpenCode
10-16x**. This is a hard ordering, not a preference — Claude's own context is
the scarcest resource in this loop, roughly 5-8x smaller than Codex and
10-16x smaller than OpenCode.

- **Claude ≤ Codex ≤ OpenCode whenever Codex is actively escalated (Level
  2+).** At Level 1, Codex is normally idle (~0%) while Claude retains an
  irreducible floor of work — diff review, scope confirmation, commit — so
  Claude can exceed Codex's near-zero share there. From Level 2 on, no
  escalation should ever assign Claude a larger share than Codex. The
  original (Codex-authored) version of this skill let its own self-role grow
  to 35-55% at high risk while capping the secondary reviewer at 0-10% — that
  pattern does not transfer here, because Claude is not the largest-budget
  engine the way Codex was in that version.
- **Claude depletes first even at a small per-phase share.** Because Claude's
  absolute quota is smallest, a flat 10-20% share compounds fast across a
  multi-phase run and can exhaust Claude's budget while Codex and OpenCode
  still have headroom left. Treat Claude budget as a running total across the
  *whole* multi-phase job, not a fresh per-phase allowance.
- **Degradation rule**: if Claude's own cumulative usage is trending high
  partway through a multi-phase run, shift further phases' gate review and
  second-opinion routing to Codex sooner — don't wait for a level escalation
  to trigger it structurally.
- **Minimum-footprint default**: even at Level 1, Claude reads diff stats,
  summaries, and OpenCode's compact report — not full source — reserving
  direct inspection for cases that actually need it.

## Role Budgeting

- **Level 1, low risk / default**: OpenCode 80-90%, Codex 0-5%,
  Claude 8-15%. Use for mechanical edits, docs/config changes, narrow bug
  fixes, routine test fixes, and repetitive refactors. Matches OpenCode's
  10-16x headroom — default to it; Codex and Claude both stay minimal.
- **Level 2, medium risk / Codex-expanded**: OpenCode 65-80%, Codex 15-25%,
  Claude 10-15%. Use when touched logic is important, tests are nontrivial,
  scope is broader than expected, or OpenCode reports uncertainty. The
  escalation goes to Codex's 5-8x headroom; Claude's share barely moves.
- **Level 3, high risk / Codex-reviewed gate**: OpenCode 45-65%, Codex 30-45%,
  Claude 10-15%. Use for architecture/API changes, memory, concurrency,
  build, CUDA, release behavior, subtle numerical correctness, weak tests,
  suspicious red gates, or unexpected diff expansion. OpenCode remains the
  plurality of token share even here; "Codex-reviewed" reflects Codex's
  judgment/gatekeeper role, not a token-share majority. Codex absorbs the deep
  reasoning work — including the headroom Claude gives up at this level;
  Claude still only verifies and decides.
- **Level 4, stop**: stop and report when the gate is invalid, implementation
  must change plan intent, required tooling is unavailable, unrelated user
  changes block safe work, or correctness cannot be verified.

Within each level, OpenCode is the default implementer and must not commit.
Claude reads the plan, phase boundary, git status, diff stat, key hunks,
validation output, and risky files only as needed, preferring summaries over
full source per the minimum-footprint default above. Escalate to Codex (via
`codex-delegate`, DISCUSS mode) for hard reasoning or adversarial review;
check the MCP-availability restriction in `AGENTS.md` first — `mcp__codex__*`
tools are only present when running as a remote-controlled agent (CCR). If
absent, fall back to Claude's own direct inspection instead of escalating,
and note that this session cannot follow the normal budget ordering.

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
   - Use OpenCode default model as the primary implementer (`opencode-delegate`).
   - Keep prompts narrow: phase context, files/scope, red gate, success
     criteria, constraints, stop rules, and no commit.
   - Let OpenCode read source, edit, run tests, and iterate within scope.
   - Require a compact final report: changed files, key decisions, commands run,
     metrics, and blockers.

4. **Claude Gate Review**
   - Review diff stat and key hunks before trusting delegate output.
   - Verify red/green evidence, test validity, and command output.
   - Run formatting/lint/diff checks appropriate to repo.
   - Confirm diff scope matches phase.
   - When risk escalation rules apply, raise the level and escalate to Codex
     (via `codex-delegate`) for deeper source/test inspection and hard
     reasoning. Do not respond to escalation by having Claude read deeper
     source directly — that spends the scarcest budget first (see Token
     Budget). Claude's own direct role stays the same size; only Codex's
     share grows.

5. **Second Opinion**
   - Ask OpenCode by default for routine read-only blocker review.
   - Use Codex (`codex-delegate`, DISCUSS mode) for architecture/API risk,
     weak or suspicious tests, subtle correctness risk, delegate uncertainty,
     delegate/reviewer disagreement, or explicit user request — this is the
     normal escalation path, not an exception.
   - If `mcp__codex__*` tools are unavailable in this session, note that and
     fall back to Claude's own direct inspection instead of escalating.
   - Ask focused questions: blockers, spec mismatches, test validity, scope,
     and stop/go.
   - Treat advice as input, not authority.

6. **Commit + Report**
   - Stage only intended files. Never `git add -A` when unrelated files exist.
   - Commit one phase at a time.
   - Push current branch only if the user has authorized it.
   - Leave PR/issue comment with:
     - commit hash
     - scope
     - validation commands and key metrics
     - deviations from plan and rationale
     - level used and Claude review scope
     - second-opinion result
     - explicit next phase / waiting-for-approval state

7. **Wait**
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
- Claude cannot verify quality from diff/test evidence without reading more
  context

Escalation means calling Codex for deeper source/test inspection and hard
reasoning — not spending more Claude tokens on direct inspection. Claude's
budget is the smallest of the three engines (see Token Budget) and depletes
first even at a small per-phase share, so Claude's own direct involvement
stays the same size across levels; only Codex's share grows. Claude's role at
every level, including high risk, remains the final scope/diff/commit
decision. For blocking risk, stop instead of weakening the gate.

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

Mechanical edit prompt (OpenCode) should include:

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

Second-opinion prompt (OpenCode, routine) should include:

```text
Read-only review. Do not edit.
Context: <phase + changed files>
Validation: <red gate + passing commands + metrics>
Review scope: <diff summary + key hunks or files>
Question: any blocking issue before commit? Focus on scope drift, test validity,
missed acceptance criteria, and obvious bug risk.
Return findings first, concise.
```

Codex escalation prompt (`codex-delegate`, DISCUSS mode, `sandbox: read-only`)
is the default escalation path at Level 2+, not a last resort:

```text
Context: <phase + risk>.
Read first: <key files>.
Evidence: <diff stat + key hunk summary + validation result>.
Question: is there a blocking issue? Return only findings or "no blocker".
Be concrete; cite file:line where relevant.
```

## Phase Report Template

```text
Phase <N> complete and pushed: <commit>

Scope:
- ...

Implementation:
- level: <1 low / 2 medium / 3 high>
- model: OpenCode default (or noted override)
- Claude review scope: <diff stat/key hunks/source files inspected>
- cumulative Claude budget: <on track / trending high -> escalate remaining
  phases to Codex sooner>

Red gate:
- `<command>`: <failing result before implementation>

Validation:
- `<command>`: <result>
- key metrics: ...

Plan deviations:
- <none or rationale>

Second opinion:
- <OpenCode routine review and Codex escalation, if used>: <verdict>

Waiting for approval before Phase <N+1>.
```
