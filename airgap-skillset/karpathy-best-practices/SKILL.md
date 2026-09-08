---
name: karpathy
description: Apply simple, surgical, goal-driven coding practice to code changes. Use when implementing, fixing, refactoring, or reviewing code.
---

# Karpathy Coding Practice

Use these rules for code changes. They favor clarity and verified outcomes over
speed; apply judgment for a genuinely trivial edit.

## Before editing

State material assumptions and tradeoffs. Do not silently choose between
reasonable interpretations, libraries, or designs. If a missing decision would
change scope or behavior, stop and ask; otherwise record the narrow assumption
and validate it from source.

## Implement

Solve the stated problem with the smallest correct change. Do not add speculative
features, single-use abstractions, unrequested flexibility, or handling for
impossible cases. Prefer code a new maintainer can understand quickly.

Touch only lines justified by the request. Preserve local style; do not refactor,
reformat, or improve adjacent code. Remove only imports, variables, or helpers
made unused by this change. Report pre-existing dead code rather than deleting
it. Every changed line must trace to the goal.

## Verify

Turn the request into observable acceptance criteria before coding. Use the
smallest relevant test, lint, build, or reproduction; make a right-reason
failure pass when feasible. For multi-step work, state each step with its
verification. Inspect the final diff for scope, simplicity, unintended API or
behavior changes, and stale artifacts. Do not claim completion until the
relevant evidence is green or an explicit limitation is reported.
