# Wait Policy

## Async workflow and elapsed-time policy

1. Keep one session lineage per role; repeat the same named agent on every
   continuation or fork, omitting model/variant. No timeout implies new authority.
2. At actual launch, record job/session IDs, role, phase, launch time,
   completed-step/event count, next poll/checkpoint, declared maximum wait and
   task-specific final-synthesis allowance. Keep one active-resource ledger;
   retrieve terminal output before marking an entry resolved. Poll active
   implementation jobs at least every 60 seconds using the exposed tools.
3. Account **each job's actual launch-to-confirmed-terminal runtime separately**
   from controller work and owner-wait time. Concurrent role runtimes are not
   summed wall-clock time. If terminal time is unknown, report runtime-to-date
   and unknown terminal state. Keep any explicit overall initiative wall-clock
   deadline; setup/owner time does not debit role runtime but does not reset or
   extend that deadline. Retain prior runtime/budgets across recovery.
4. Keep the controller turn active while required jobs are live: commentary or
   user-requested status is a non-final checkpoint, then resume polling. Do not
   pass a gate, commit, publish, change phase, or claim completion while required
   resources are live/unknown. Before a successful gate/final report, verify no
   unresolved active resources remain. At owner stop or hard wait expiry, a
   blocked/pending final report instead lists every live/unknown job and command;
   it must not assert an empty ledger or successful completion.
5. At declared elapsed checkpoints, recover status and classify measurable
   activity versus liveness only. Advancing completed steps/events are progress
   without partial text. Silence or a read-only reviewer's `running` alone is
   not failure or no-progress evidence. Never duplicate a slow/live job.
6. After two unchanged measurable checkpoints, record a wait-policy review and
   use the procedure below, not automatic cancellation or abandonment. Retrieve
   terminal results before any verdict; a tool-step-cap exit alone is not green.
   After terminal-result retrieval and scoped diff review, an authorized planned
   same-session continuation may proceed within unchanged scope/route/caps.
7. For synthesis-pending expert work (including post-inspection `step_start`),
   retain the same job for its **declared task-specific synthesis allowance**
   within maximum wait and any overall deadline. Choose realistic values at
   launch, not a universal ten-minute minimum or unexplained micro-budgets.
   Progress never extends an approved hard limit.
8. At hard expiry, block the affected gate as pending/blocked and report job IDs,
   known statuses, retrieved/unavailable results, scoped diff, elapsed evidence,
   and next required controller/owner decision. Do not infer cancellation,
   unavailability, completion, or replacement permission from timeout. Record
   session counts, candidate attempts, wait reviews, and route/reviewer evidence.

## No-progress recovery

1. At the two-checkpoint trigger, query the terminal result exactly once for
   that review and record the query. If still running, retain the entry and
   continue ordinary polling/terminal retrieval; this is not a one-query lifetime
   limit. Inspect only the approved scoped worktree for an editing job; an
   unchanged diff is supporting evidence, not proof of a stall.
2. Retain the same job while events/output advance, known-long work is live, or
   synthesis is pending within its declared allowance. Missing partial text or
   scoped diff never alone justifies cancelling a read-only reviewer/preflight.
3. Cancellation requires an explicit controller/owner stop decision or declared
   maximum-wait procedure, with recorded reason and any approved replacement
   route. A request to cancel is **not** confirmation of terminal cancellation.
   Recover status, retrieve the terminal result, and account for the resource.
   If status/result cannot be recovered, mark unknown and block the gate.
4. Never duplicate a live/unknown job or launch replacement on a cancellation
   request alone. Only after confirmed terminal state, terminal-result retrieval,
   and scoped diff review may an already-authorized continuation proceed.

## Essential-capability recovery

Use the [failure taxonomy](../SKILL.md#failure-classification-and-recovery), not
no-progress as a permission bypass. Essential absence suspends affected edits;
known host/access denial is a hard stop, never fresh-session recovery.

If the initial envelope expressly approves it, **the controller only** may
launch one fresh session with the same named role per **phase**, after the prior
job is confirmed terminal, its result retrieved, and partial scoped diff reviewed.
Record old/new job and session lineage, original plan/P0/checkout identity,
partial edits, attempt counts, and retained runtime limits. No recovery per
failure/session, role/model/destination change, cap reset, new clone, denial
evasion, or self-launched replacement. Prove the full actual-delegate capability
and identity gate in the new session before further edits; existing diff is
preserved as reviewed state, not silently discarded. Failed or unavailable
recovery returns narrow evidence to the controller and ends affected execution.

Usage is observational evidence only; do not introduce fixed token, price, or
provider-private-path limits or use `usage_mcp`.
