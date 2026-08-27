# Review and Wait Policy

Read this immediately before starting a triggered review, polling an async
delegate/reviewer, or passing a gate with local commands. Return evidence to
the core; this reference does not decide publication or continuation.

## Triggered independent review

Do not automatically launch OpenCode Terra for Codex-controlled L2/L3 work.
Use `agent="terra"` only for a recorded fresh-context, read-only concern.
Trigger one independent Claude read-only review for architecture/API
compatibility, security, memory/concurrency, CUDA or numerical correctness,
release behavior, weak tests, disagreement, suspicious red gates, or an
explicit request. One triggered review replaces a review stack. Codex remains
the gate; review evidence is not publication authority.

## Async checkpoints

Declare expected elapsed-time checkpoints and a maximum wait for every async
delegate/reviewer. At each checkpoint recover status and classify it as
measurable activity or liveness only. For OpenCode, advancing completed-step
or event count is material progress even without partial text; never duplicate
a slow job. A Claude reviewer exposing only `running` is live but not measurable:
missing partial text does not increment no-progress. Keep its job through the
declared maximum wait and report `review pending`.

While a required async job is live, keep the controller turn active. A
commentary checkpoint reports state and the next poll only; it is never a final
response, implicit owner handoff, phase stop, gate, or publication action.
Finalize only after the terminal result has been retrieved, the owner explicitly
interrupts/stops the work, or the declared maximum wait has been reached.

## Active OpenCode implementation jobs

Operational polling is defined in
[`opencode-delegate`'s wait policy](../../opencode-delegate/references/wait-policy.md).
Before every phase gate and final response, the implementation-job ledger must
be empty; an unrecoverable job state blocks the gate.

After two unchanged measurable checkpoints, record a wait-policy review; do
not stop, abandon, or cancel solely for that state. On terminal `done`, fetch
the terminal result before a verdict. At maximum wait, stop the affected phase
as `review pending at maximum wait` and request controller/owner direction;
never automatically cancel or treat a live reviewer as unavailable. Evidence is
unavailable only for missing route/tool, terminal error/cancellation, or failed
terminal-result retrieval. For OpenCode no-progress, use the
`opencode-delegate` recovery, including exactly one terminal-result query
before the controller stops. For a full Sol-expert consultation that reaches
`step_start` after inspection, retain the same job for a final-synthesis grace
of at least 10 minutes; the phase maximum wait cannot be shorter.

## Local commands

Treat every `exec_command` `session_id` as an active phase resource. Record it,
poll it at least every 60 seconds until exit, and never duplicate a slow local
command. Do not answer finally, evaluate/pass a gate, commit, publish, change
phase, or claim validation while any active session exists. Before every gate
and final response verify the list is empty; unrecoverable state is unknown and
blocks the gate. On user interruption, report each session and resume polling
unless explicitly stopped.
