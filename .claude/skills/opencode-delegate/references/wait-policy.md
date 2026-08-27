# Wait Policy

## Async workflow and elapsed-time policy

1. Keep one session lineage per role. Repeat the same named agent on every
   continuation or fork, and pass a stable workflow ID.
2. At launch, record an active-job ledger: job ID, session ID, role, phase,
   start time, most recent completed-step or event count, and next polling
   deadline. Poll every active implementation job at least every 60 seconds;
   update its progress and deadline. Retrieve terminal output and normalized
   usage before removing a terminal entry.
3. While the ledger has a running implementation job, keep the controller turn
   active. A user-requested status response is a non-final checkpoint: report
   the ledger snapshot and resume polling next. Do not finalize, pass a gate,
   commit, publish, change phase, or claim completion until the ledger is empty.
4. At each phase-defined elapsed-time checkpoint, recover status and inspect
   material progress. For OpenCode jobs, an advancing completed-step or event
   count is material progress even when no partial text exists. Never duplicate
   a running job because it is slow.
5. After two no-progress checkpoints, record a wait-policy review and run the
   no-progress recovery below. Do not stop, abandon, or cancel solely for that
   condition. Cancel only on terminal error, the phase-defined maximum wait,
   or an explicit controller/owner stop decision.
6. On a tool-step-cap exit, fetch the terminal result, inspect the scoped
   worktree independently, and start the planned same-session continuation when
   work remains. A cap exit alone is not successful implementation evidence.
7. For a full `sol-expert` consultation that has completed inspection work and
   then reports `step_start`, treat the state as final synthesis pending. Keep
   the same job through a declared final-synthesis grace of at least 10 minutes;
   its phase-defined maximum wait must be no shorter than that grace.
8. Use job list to recover recorded jobs. Record session count, retries,
   elapsed time, checkpoints, route evidence, and reviewer trigger reason.

## No-progress recovery

1. At delegation start, record the ledger's job status and completed-step/event
   counts. For a job expected to edit, also record its scoped worktree state.
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
