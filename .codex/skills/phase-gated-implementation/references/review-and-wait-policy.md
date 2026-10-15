# Review and Wait Policy

Read this immediately before starting a triggered review, polling an async
delegate/reviewer, or passing a gate with local commands. Return evidence to
the core; this reference does not decide publication or continuation.

## Triggered independent review

Follow the core's [review router](../SKILL.md#router-and-final-gate) and
[Astra policy](../SKILL.md#routes-and-selector-stops). Do not substitute an
OpenCode expert for the external independent reviewer. Before any Astra
delegation, read
[delegation.md](delegation.md) immediately.
Trigger one independent Claude read-only review for L3 post-implementation
review and for architecture/API, security, concurrency, numerical correctness,
release behavior, weak tests, disagreement, suspicious red gates, or an
explicit request. If unavailable at L3, stop. Codex remains the final gate;
review evidence is not publication authority.

L3 requires fresh expert preflight and a distinct post-implementation
independent review through the external controller. Do not add a review stack
or let the preflight author review its own recommendation.

## Async checkpoints

Declare task-specific checkpoints, maximum wait, and a final-synthesis
allowance before launching each delegate/reviewer, bounded by any explicit
overall wall-clock deadline. Use the canonical
[async ledger, polling, accounting, and recovery procedure](../../opencode-delegate/references/wait-policy.md).
It separates actual job launch-to-terminal runtime from controller and owner
wait, retains job lineage and caps, and never converts timeout into authority.

For OpenCode, advancing completed steps/events are progress without partial
text. A reviewer exposing only `running` is live but not measurable; missing
partial text does not increment no-progress. Keep the same job, poll and retrieve
terminal results. Expert synthesis uses the declared task-specific allowance,
not a mandatory ten-minute grace. Two unchanged measurable checkpoints trigger
a wait-policy review, not abandonment, duplication, or automatic cancellation.

## Active OpenCode implementation jobs

The linked wait policy owns active-resource accounting. Before a successful
gate or final success response, all required jobs must be terminal with results
retrieved and no unresolved active resource. Keep the controller turn active
while work is live; commentary checkpoints are not owner handoffs or gates.
At explicit owner stop or hard expiry, report the phase **blocked/pending** and
every live/unknown job, command, result-retrieval state, and next required decision.
That exceptional stop report does not assert an empty ledger or completion.
Cancellation requests alone never enable replacement; unconfirmed resource
state blocks the gate. Evidence unavailable from missing route/tool, terminal
error/cancellation, or failed retrieval is a narrow blocker, not proof that a
running reviewer failed. No automatic reviewer substitution or live duplication.

## Local commands

Treat every `exec_command` `session_id` as an active phase resource. Record it,
poll it at least every 60 seconds until exit, and never duplicate a slow local
command. Do not evaluate/pass a gate, commit, publish, change phase, or claim validation
while an active/unknown session exists. Before success, verify the list is
empty. An owner stop or hard expiry permits only a blocked/pending final report
accounting for each live/unknown session, not an empty-list or completion claim.
On user interruption, report each session and resume polling unless explicitly
stopped. A cancellation request is not terminal exit evidence.
