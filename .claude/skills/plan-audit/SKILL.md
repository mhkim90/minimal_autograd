---
name: plan-audit
description: Audit an externally authored implementation plan before execution. Use when a user supplies a plan, design document, issue, PR description, decision record, migration outline, or checklist and Codex or Claude must determine whether it is safe and actionable enough for phase-gated implementation. Do not use for a plan drafted by the active agent; use grilled-me for that.
---

# Plan Audit

Audit an external plan once per material revision; do not loop with `grilled-me`. This is a read-only preflight, not approval, implementation or permission to edit. Report consequential gaps rather than inventing scope or gates. Proposed globs, risk levels and sequencing remain drafts until approved; audit output alone never advances a phase.

## Establish the audited revision

Identify the artifact, source, revision (when available) and authoritative constraints. Without a stable revision, bind findings to the visible conversation content. Re-audit material changes to scope, contracts, sequencing, acceptance, risk, rollback or strategy, not editorial changes.

## Audit procedure

Before evaluating, read the [audit procedure](references/audit-procedure.md). Apply relevant checks and mark others not applicable. Stop and clarify ambiguous artifact identity, conflicting authority, unclear outcomes/contracts, destructive or irreversible steps without approved rollback, unbounded scope, wrong-reason gates or missing evidence, credentials, tools or human authority.

## Severity and disposition

Order findings by severity: **critical** (unsafe, destructive or wrong outcome), **major** (phase cannot run or be verified), **moderate** (requires a manual gate, narrower scope or clarified contract), **minor** (deferrable precision or maintainability). Give the smallest adequate repair; distinguish blockers before a named phase, manual gates and nonblocking follow-up.

## Verdict

Choose one:

- `ready`: no blockers; newly proposed boundaries and gates still need approval.
- `ready-with-manual-gates`: execution may begin within existing authority, stopping at named human decisions or predicates that cannot be checked mechanically.
- `blocked`: do not begin the affected phase until its blockers are resolved.

Before writing findings and the phase-gate handoff, read the [output format](references/output-format.md) for the required evidence, fields and owner decisions. Resolve blocking findings and obtain required owner approvals before implementation.
