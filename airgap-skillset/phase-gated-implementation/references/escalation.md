# Escalation

Read this immediately before triggering `isolated-review` or considering
an `airgap-export` round trip. Return the verdict/outcome to the core;
this reference does not decide publication or continuation.

## Session and retry controls

One bounded phase or subphase per implementation session; end it after
green. Start a fresh session when scope, the red gate, strategy, or
blocker changes materially. Allow at most three implementation/fix
attempts. After two failures on the same blocker, stop blind retries:
either write a materially revised approach before a third attempt, or
escalate via `airgap-export`.

## Isolated review

Run before publish for Standard/Difficult ceremony tiers (see core). Same
model, fresh context, diff + gate evidence only — see the `isolated-review`
skill for the full isolation rules, checklist, and output format. It will
not catch a blind spot the model shares between its implementer and
reviewer roles; that gap is what `airgap-export` is for.

## airgap-export escalation

Escalate to `airgap-export` (the manual, human-carried, de-identified
round trip to an outside model) only when:

- two implementation/fix attempts on the same blocker have failed, or
- a Difficult-tier phase needs a second opinion on the approach itself
  before an irreversible or high-risk step — export as `CLASS:
  design-review`, or
- `isolated-review` returns a blocker the controller cannot resolve with
  available evidence.

An escalation trigger firing does not itself authorize anything to leave
the closed environment — `airgap-export`'s own authorization,
de-identification, and human-review rules still apply in full.

Round trips are manual and expensive — batch the question, follow
`airgap-export`'s own rules (try locally first, minimal reproducible
payload, one precise ask). Permit one initial consultation and at most
one follow-up per unresolved question; a follow-up may only clarify the
same answer or address a named remaining uncertainty, never open a new
design loop. Opening a new case id for the same unresolved question does
not restart this allowance. If still blocked after the follow-up, stop
the affected phase and return control to the owner rather than exporting
again.

The returned advice is advisory only: it cannot authorize implementation,
publication, scope expansion, a gate waiver, or an additional retry, and
it does not reset the implementation/fix attempt cap. It does not satisfy
a required `isolated-review` or any other required review — a
design-review reply is an opinion the controller validates against its
own gates, not a review of the resulting implementation. One escalation
does not upgrade later phases to export-by-default; each phase re-earns
its own trigger.

Do not use it for routine per-phase review; that is what `isolated-review`
is for. Once `airgap-triage`'s answer is applied back, record the case
id, consultation/follow-up count, and outcome in the phase report.
