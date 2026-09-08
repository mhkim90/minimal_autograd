---
name: isolated-review
description: Fresh-context, same-model review of a diff before it's published. Use in a single-model environment where a genuinely different reviewer model isn't available — the reviewer sub-agent sees only the diff and gate evidence, never the implementer's reasoning, chat history, or rationale.
---

# Isolated Review

A single-model environment cannot get genuine model diversity between
implementer and reviewer — same weights, same blind spots. This skill
substitutes information isolation for model diversity: the reviewer
sub-agent gets a clean context and only the artifacts an outside reviewer
would have, never the implementer's own justification for the choices it
made.

## When to use

- Before publishing (commit/push/delivery draft) any Standard- or
  Difficult-tier phase
  under `phase-gated-implementation`.
- Optional for Mechanical/economy changes when automated gates (lint, type
  check, tests) are already green and sufficient on their own.
- Not a substitute for `airgap-export` escalation. Use this for routine
  per-phase review; escalate to `airgap-export` on a repeated blocker or an
  explicit high-risk trigger.

## Isolation rules

1. Start a genuinely new sub-agent session/context. Do not fork or continue
   the implementer's session — no shared history, no shared scratch notes.
2. Give the reviewer only: the diff (or changed-files list + patch), the
   red/green gate commands and their output, and the stated success
   criteria. Never give it the implementer's rationale, chat transcript, or
   "why I did it this way" explanation.
3. Reviewer gets read-only tool access. It must not edit, commit, or run
   anything beyond the stated validation commands.
4. Reviewer works from the structured checklist below, not an open-ended
   "does this look ok?" prompt — structure is what substitutes for the
   missing model diversity.

## Reviewer checklist

- Does the diff match the stated success criteria, or does it solve an
  adjacent/easier problem?
- Any change outside the approved scope/files?
- Any claim in the phase report ("tests pass", "works") not backed by an
  actual command + output shown in the evidence?
- Edge cases the gate doesn't exercise: empty/null input, concurrency,
  error/partial-failure paths, boundary values.
- Anything that looks copied from elsewhere in the repo but subtly
  diverges (off-by-one, wrong variable, stale comment)?
- Would this diff surprise someone who only reads the code, not the
  conversation that produced it?

## Output

```text
[Isolated Review]
Scope match: <yes | no — what's out of scope>
Evidence check: <each claim -> backed by shown command output, or unverified>
Findings: <list, or "none">
Verdict: <no blocker | blocker: reason>
```

A `blocker` verdict stops publication; return to implementation with the
specific finding, not a re-run of the same prompt.

## Limits

This is the same model reviewing itself under different framing. It catches
inconsistency, unmet criteria, and structural gaps well. It will not catch a
mistake the model would make identically as implementer and reviewer — a
shared blind spot. Where that risk matters (architecture-sensitive,
security, numerical/concurrency correctness, or a verdict you don't trust),
escalate via `airgap-export` instead of accepting this review as sufficient.
