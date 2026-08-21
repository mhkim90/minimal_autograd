---
name: plan-audit
description: Audit an externally authored implementation plan before execution. Use when a user supplies a plan, design document, issue, PR description, decision record, migration outline, or checklist and Codex or Claude must determine whether it is safe and actionable enough for phase-gated implementation. Do not use for a plan drafted by the active agent; use grilled-me for that.
---

# Plan Audit

Audit an externally authored plan before implementation. Find the smallest
repairs needed to make the plan executable, and produce a structured handoff
for phase-gated implementation.

This is a read-only review skill. Do not edit the plan, approve proposed
boundaries, begin implementation, or silently fill consequential gaps.

## Relationship to Other Skills

- Use `plan-audit` for a plan or execution artifact authored outside the active
  agent.
- Use `grilled-me` to stress-test a plan drafted by the active agent before it
  is presented.
- Use `phase-gated-implementation` only after blocking audit findings are
  resolved and the user has approved any required phase boundaries and gates.

Do not combine these skills into one loop. A plan audit is normally a one-time
preflight for a specific artifact revision, not a review repeated at every
implementation phase.

## Authority Boundary

The audit reports evidence and proposes repairs. It does not authorize them.

- Candidate scope globs are drafts until explicitly approved by the user or
  repository owner.
- Candidate gates, sequencing changes, and risk levels are also drafts.
- Unapproved audit output must not enable automatic phase continuation.
- If the plan lacks enough information to define a narrow boundary or
  falsifiable gate, report the gap instead of inventing a broad substitute.
- Do not modify the audited artifact unless the user separately requests an
  edit.

## Establish the Audited Revision

Before reviewing, identify:

- artifact title and path, URL, issue, PR, or conversation reference;
- revision, commit, blob, version, or other stable identifier when available;
- how the content was obtained;
- related artifacts that are authoritative for scope or constraints.

If no stable revision exists, state that the audit applies only to the content
visible in the current conversation.

Treat the audit as stale when a later revision materially changes scope,
contracts, sequencing, acceptance criteria, risk, rollback, or implementation
strategy. Editorial changes alone do not require a new audit.

## Audit Procedure

Read [`references/audit-procedure.md`](references/audit-procedure.md)
immediately before evaluating the plan. Apply all applicable checks and mark
irrelevant checks as not applicable.

## Severity and Disposition

Order findings by severity:

- **Critical**: implementation would be unsafe, destructive, or directed at the
  wrong outcome.
- **Major**: a phase cannot be executed or verified as written.
- **Moderate**: the plan is executable only with a manual gate, narrower scope,
  or clarified contract.
- **Minor**: useful precision or maintainability improvement that can be
  handled later.

For each finding include:

- evidence or exact plan location;
- what breaks and when;
- affected phase;
- smallest adequate repair;
- disposition: blocking before a named phase, manual-gate required, or
  non-blocking follow-up.

Praise only load-bearing strengths: constraints or decisions that materially
reduce risk.

## Verdict

Use exactly one:

- `ready`: no blocking findings; proposed boundaries and gates still require
  approval if the plan did not already contain approved ones.
- `ready-with-manual-gates`: execution may begin, but named phases must stop for
  a human decision or because a predicate is not mechanically decidable.
- `blocked`: implementation must not begin, or must stop before the named
  phase, until blocking findings are resolved.

## Output Format

Read [`references/output-format.md`](references/output-format.md) immediately
before writing findings or the phase-gate handoff.

## Stop Rules

Stop and request clarification rather than guessing when:

- the artifact or authoritative revision cannot be identified;
- two sources of truth conflict;
- the intended outcome or externally visible contract is ambiguous;
- a destructive or irreversible step lacks an approved rollback;
- scope cannot be bounded narrowly enough to detect drift;
- a phase gate cannot distinguish the intended change from a wrong-reason
  result;
- the plan depends on unavailable evidence, credentials, tooling, or human
  authority.
