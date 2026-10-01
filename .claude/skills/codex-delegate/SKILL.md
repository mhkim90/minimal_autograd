---
name: codex-delegate
description: Delegate independent high-risk reasoning, adversarial review, and blocker gates from Claude Code to Codex Sol. Use it only for DISCUSS/read-only review.
---

# Codex Delegate

Use the installed Codex plugin's native review commands, not task or write
routes. For an unfocused native git-diff/working-tree review, invoke
`/codex:review`. For a focused review of a design, blocker, or specific
approach, invoke `/codex:adversarial-review`. Keep the review target and
context inside the caller's current repository.

Every mandatory Codex review must pass `--model gpt-6.1-sol` explicitly to
the selected review command. Never silently use the configured default or
claim the intended model when the requested model is not observed.

Sol is the intended independent Codex review model, not an assumption about
the configured default. Record the requested model, plugin review/job
identity, and active model when the plugin exposes it. Stop on an explicit
mismatch; do not trust a generic self-label over invocation or runtime
metadata.

## Mode

The two review commands are the only allowed Codex execution routes here. They
must remain read-only and reviewer-only: do not invoke `/codex:rescue`, the
plugin's `task` verb, or any write-capable/task route. Codex must not edit,
implement, fix, or publish; implementation remains with the existing OpenCode
routes.

Claude Code orchestrates and controls publication. OpenCode Luna implements
standard work, justified Sol/Astra implementers handle harder bounded work,
and Astra expert supplies fresh read-only preflight or blocker reasoning.
Codex Sol supplies one triggered independent review. Do not substitute an
OpenCode Astra review for that external independent reviewer.

## Discuss prompt

```text
Read-only independent review. Do not edit.
Context: <project + phase>
Read first: <smallest relevant files and diff>
Validation: <red gate and passing checks>
Trigger reason: <architecture/security/CUDA/etc.>
Question: <specific correctness or blocker question>
Return findings first or "no blocker", then stop/go.
```

## Job, timing, and evidence

- Start a fresh plugin review job for a new review and capture its job ID.
  Recover the job only through the plugin's `status`, `result`, and `cancel`
  commands. Do not invent thread continuity or use a thread/reply mechanism.
- Declare an elapsed-time checkpoint and maximum wait. At each checkpoint,
  query `status` before any further action; use `result` when completion is
  reported. Never duplicate a slow, unresolved job.
- If continuity for a needed follow-up is not proven, wait for the original
  job to become terminal or explicitly cancel it, then start at most one fresh
  contextualized review job with the relevant target, findings, and question
  repeated. A fresh follow-up is not permission to retry an unresolved job.
- After two no-progress checkpoints, record a wait-policy review; do not stop
  or cancel solely for that condition. Stop only when the declared maximum
  wait is reached or the controller makes an explicit stop decision.
- On a failed, cancelled, or unknown job, report that state and preserve its
  job ID for evidence. While a job remains unresolved, do not launch a
  duplicate. At the declared maximum wait or an explicit controller stop, use
  `cancel` as appropriate and report the cancellation and elapsed time.
- Report mode, requested model, observed active model (if available), plugin
  review/job ID, elapsed time, findings, and warnings. Distinguish observed
  model evidence from unavailable model evidence; missing metadata is
  observational and must not be filled in or treated as proof of Sol. An
  explicit model mismatch is blocking.

## Temporary review-capacity restriction

If this skill's independent Codex reviewer is unavailable for an L3 phase,
stop the phase and report the missing independent review. Neither a second
OpenCode Astra consultation nor a review by the preflight author satisfies
the independent-review gate. Do not silently substitute a reviewer or change
publication authority.
