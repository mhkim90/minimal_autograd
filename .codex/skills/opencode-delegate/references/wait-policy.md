# Wait Policy

## Async workflow and elapsed-time policy

1. Keep one session lineage per role. Repeat the same named agent on every
   continuation or fork, and pass a stable workflow ID.
2. Poll job status and fetch the terminal result. Query normalized usage after
   terminal state.
3. While a required job is live, keep the controller turn active. A commentary
   checkpoint may report status and its next poll, but it never becomes a final
   response, implicit owner handoff, phase stop, gate, or publication action.
   Finalize only after terminal-result retrieval, explicit interruption/stop,
   or the declared maximum wait.
4. At each phase-defined elapsed-time checkpoint, recover status and inspect
   material progress. For OpenCode jobs, an advancing completed-step or event
   count is material progress even when no partial text exists. Never duplicate
   a running job because it is slow.
5. After two no-progress checkpoints, record a wait-policy review and run the
   no-progress recovery below. Do not stop, abandon, or cancel solely for that
   condition. Cancel only on terminal error, the phase-defined maximum wait,
   or an explicit controller/owner stop decision.
6. For a full `sol-expert` consultation that has completed inspection work and
   then reports `step_start`, treat the state as final synthesis pending. Keep
   the same job through a declared final-synthesis grace of at least 10 minutes;
   its phase-defined maximum wait must be no shorter than that grace.
7. Use job list to recover recorded jobs. Record session count, retries,
   elapsed time, checkpoints, route evidence, and reviewer trigger reason.

## No-progress recovery

1. At delegation start, record job status and completed-step/event counts. For
   a job expected to edit, also record its scoped worktree state.
2. After the two-checkpoint trigger, fetch the terminal result exactly once and
   record the result query. For an editing job, record whether its scoped
   worktree changed; absence of a change is supporting evidence only.
3. Retain the same job when events or live process output advance, when full
   `sol-expert` final synthesis remains within its declared grace, or when a
   known-long task reports live output. Do not use elapsed time alone to call
   these jobs stalled.
4. If the result remains running and no retention condition applies, cancellation
   requires an explicit controller stop decision. Record the reason and any
   approved replacement route before cancelling.
5. Continue locally or start a fresh narrower delegate only when the approved
   phase scope and route permit it; otherwise wait for owner direction. Do not
   use a stalled job as an unreported reason to pause an authorized phase.

This recovery rule never treats two unchanged polls or a missing scoped diff as
enough evidence to cancel a read-only review or Sol preflight.
