---
name: opencode-delegate
description: Delegate approved work through OpenCode MCP. Use Terra by default, Luna only on explicit request, justified Sol/Astra implementers for harder work, and Astra expert for bounded read-only analysis.
---

# OpenCode Delegate

Use `mcp__opencode.opencode_run_async` for substantive execution, iteration and long tests; block only for known-short work. Don't delegate single commands or ambiguous design decisions.

## Boundary

No selectable `cwd`. Even explicit-cwd tools cannot escape the caller directory (including to `/tmp`). Approved contained checkouts, including other repos, need no repeat approval; containment authorizes no checkout, clone, edit or publication. A requested configured delegate may receive necessary in-scope source/diffs. Changed destinations, unconfigured services, out-of-scope private material, destructive acts and publication need separate authority.

Before editing, verify checkout and needed capabilities **inside the delegate**. Containment doesn't guarantee linked/common Git metadata access. No universal marker/hash test. Missing native `apply_patch`? Discover an existing permitted shell editor and verify the authorized edit; no installs, probe writes or wider permissions.

- **Unavailable capability:** suspend affected edits. A reported absence isn't observed failure or denial; wrapper/prose/JSON isn't native execution evidence. After terminal output and scoped state/diff checks, only the controller may start a fresh same-role session under unchanged authority. Preserve pending effects, provenance, history and limits; reverify capabilities. No repeat approval, blind replay or scope/model switch. Repeated unavailability stops for diagnosis.
- **Denied access:** stop affected work; never relaunch to evade denial or copy/redirect Git metadata via git-dir, work-tree or symlinks. Only after metadata denial may the owner separately authorize an independent clone. Bind its source/ref/revision, destination, task branch and allowed scope/actions. Preserve dirty originals and plan/P0 provenance; reverify capabilities. Never substitute automatically.

Implementers only edit/test: no planning, review, staging, commit, publication, PRs or subdelegation. Permissions aren't authorization or a complete sandbox. Never access/expose secrets or credentials.

## Routes

- Standard and mechanical work: `agent="terra-implementer"`.
- Hard implementation, with justification: `agent="sol-implementer"` or `agent="astra-implementer"`.
- Explicitly requested Luna only: `agent="luna"`; never an implicit fallback.
- Hard question/blocker: fresh read-only `agent="astra-expert"`. Inspect only decision-relevant evidence; return findings, approach, acceptance gate and stop/go. Stop on missing evidence or no meaningful progress. No edits, publication, delegation, allocation resets or self-review.
- Explicitly approved whole phase: `agent="astra-orchestrator"` advises on plans/gates only. The external controller retains gates, independent review and publication; this is not caller authentication.
- Owner-requested configured default: omit `agent`, `model`, `variant`.

Named profiles pin model and high effort: omit model/variant on launches, continuations and forks. On selector rejection, conflicting overrides, reported role/model mismatch, fallback or broken lineage, stop the phase without substitution. Record requested route, job/session and observed model if available. Missing telemetry alone after a terminal response honoring the selector is not a mismatch.

Use expert preflight by risk. For phase approvals and finish review read [phase-gated-implementation](../phase-gated-implementation/SKILL.md); L3 requires the independent runtime reviewer (`claude-delegate` for Codex, `codex-delegate` for Claude), never an implementer or expert.

## Prompt

Give one self-contained phase/subphase per same-role session: outcome, plan, checkout/dirty state, allowed paths/actions, relevant evidence, meaningful pre-change signal, checks, acceptance/stop criteria, limits and authorized recovery. For experts, give one question, diff/test evidence and evidence needed to decide; verify read access, not editing.

## Failures and bounded lifecycle

Fix ordinary typos, patches and fixtures within scope, acceptance and budget. A path typo isn't checkout drift; don't search outside scope. A broken fixture isn't valid RED; preparation RED isn't implementation GREEN. End at acceptance. Return recurring blockers with evidence for a changed approach, expert help or owner direction.

Carry explicit shared limits, consumption and deadlines across sessions, roles, reloads, recovery and review fixes. Stop at exhaustion; renamed stages don't reset budgets. Successors inherit only binding limits. Invent no caps or approval gates.

For approved Nsight profiling, the controller captures **and converts on the host**; the confined delegate analyzes verified report/SQLite outputs. Follow local profiling notes if available. Host conversion of a sandbox capture is not equivalent; never disable the sandbox or widen permissions.

## Waiting and reporting

Set checkpoints, maximum wait and synthesis time within any overall deadline. Track required job/command IDs and status; separate runtime from controller/owner wait. Poll the same job to terminal and retrieve output before a verdict, commit, publication or phase advancement. Events without text may show progress; `running` proves only liveness.

Never duplicate or silently abandon live/unknown work. Cancel only for a stated error, owner/controller stop or declared deadline procedure. Confirm terminal state and retrieve output before authorized replacement. At owner stop/hard deadline report each pending resource, retrieval state and next decision.

Return changed files, commands/results, allocation history, lineage and blockers. Use authorized invocation/exit/readback evidence, never private logs or fabricated usage. Usage is observational; no quality gate or `usage_mcp`. Cross-repo delivery follows `skill-sync`; never distribute the Codex source-local skill. Use the configured GitHub MCP/app; stop if unavailable, never silently use `gh`.
