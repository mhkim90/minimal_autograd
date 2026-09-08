# Output Format

## 1. Artifact

- identity:
- audited revision:
- content source:
- related authoritative artifacts:
- audit freshness caveat:

## 2. What the Plan Does

Concise summary of outcome, phases, success condition, and non-goals.

## 3. Load-Bearing Strengths

Only strengths that materially improve executability or safety.

## 4. Findings

List findings in severity order. For each:

- severity:
- evidence:
- affected phase:
- failure mode:
- smallest repair:
- disposition:

## 5. Sequencing and Contract Repairs

Show the minimal corrected ordering and unresolved contract decisions. Do not
rewrite the whole plan.

## 6. Phase-Gate Handoff

### Audit verdict

`ready`, `ready-with-manual-gates`, or `blocked`

### Global controls

- approved non-goals already present in the plan:
- unresolved human decisions:
- characterized-but-not-guaranteed behavior:
- enforcement gaps:
- stale-audit triggers specific to this plan:

### Per-phase handoff

Repeat for each phase:

- phase identifier and title:
- dependencies and preconditions:
- candidate scope globs: `<draft / already approved in plan / unavailable>`
- red gate and expected right-reason failure:
- green validation and exit criteria:
- candidate initial risk level and evidence:
- stop rules and attempt bound:
- downstream consumers:
- blocking findings before this phase:
- required manual decisions:

## 7. Owner Decisions Required

List only unresolved decisions that materially affect scope, contracts,
sequencing, gates, or safety.

End by stating that candidate boundaries and audit recommendations remain
unapproved unless the owner explicitly approves them.
