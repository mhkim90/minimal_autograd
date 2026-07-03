---
name: karpathy
description: "Use when Codex is building, modifying, reviewing, or debugging code and should follow Karpathy-style LLM coding best practices: think before coding, keep changes simple, avoid speculative abstractions, make surgical edits, and verify against explicit success criteria."
---

# Karpathy Coding Practices

Use these guidelines to reduce common LLM coding mistakes during code work.

## Think Before Coding

- State assumptions explicitly before implementing.
- Surface ambiguity instead of silently choosing a risky interpretation.
- If multiple reasonable approaches exist, describe the tradeoff briefly.
- Ask only when local context cannot resolve an important ambiguity.

## Simplicity First

- Write the minimum code that solves the requested problem.
- Avoid features, abstractions, configurability, and error handling that were not requested or are not needed.
- Prefer code a maintainer can understand quickly.
- If a solution grows larger than necessary, simplify before finishing.

## Surgical Changes

- Touch only files and lines that trace directly to the request.
- Do not refactor, reformat, or clean up adjacent code unless required by the task.
- Match the repository's existing style even when a different style would also work.
- Remove only unused imports, variables, or helpers introduced by the current change.
- Mention unrelated dead code or cleanup opportunities instead of changing them.

## Goal-Driven Execution

- Define success criteria before editing.
- For bugs, prefer a reproducing test or the smallest command that proves the failure.
- After editing, run the smallest useful validation for the touched area.
- Broaden validation only when the change affects shared behavior or cross-module contracts.

## Quick Checklist

- Does every changed line serve the immediate goal?
- Is any abstraction premature?
- Did the change preserve local conventions?
- What is the most likely failure mode?
- What check proves this is done?
