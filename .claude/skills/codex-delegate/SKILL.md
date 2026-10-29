---
name: codex-delegate
description: Get an independent, read-only review from Codex Sol through the Codex plugin. Use for required L3 review and for high-risk reasoning, adversarial review, or blocker questions; never for edits.
---

# Codex Delegate

Use only the plugin's read-only review commands: `/codex:review` for a
plain diff or working-tree review, `/codex:adversarial-review` for a
focused design, blocker, or approach question. Never use `/codex:rescue`,
the `task` verb, or any write-capable route; Codex does not edit, fix, or
publish. Keep the review target inside the current repository.

Pass `--model gpt-6.1-sol` explicitly. If the plugin reports a different
model, treat that as a blocker for a required review; if it reports none,
say the model is unconfirmed rather than claiming Sol.

An OpenCode Astra consultation, or a review by whoever wrote the
preflight advice, does not replace this independent review. If it is
unavailable for a required L3 review, stop the phase and report it.

## Prompt

```text
Read-only independent review. Do not edit.
Context: <project + phase>
Read first: <smallest relevant files and diff>
Validation: <red gate and passing checks>
Question: <specific correctness or blocker question>
Return findings first or "no blocker", then stop/go.
```

## Job

Start a fresh review job and keep its ID. Follow it with the plugin's
`status`, `result`, and `cancel` commands, and fetch the result before
reporting. Do not start a duplicate while a job is unresolved, and cancel
only for a stated reason. Report the job ID, the model observed, findings,
and the verdict, or that the review is still pending.
