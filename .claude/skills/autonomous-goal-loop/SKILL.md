---
name: autonomous-goal-loop
description: Run approved work unattended for hours or days toward an objectively measurable goal, using a machine-checkable metric instead of a per-phase owner approval. Use when the owner authorizes one envelope up front and expects no further intervention until the goal, a budget, or a stop condition is reached.
---

# Autonomous Goal Loop

The owner approves one envelope, once. After that the loop iterates without
intervention until the goal is met, a budget is exhausted, or a stop condition
trips. It never publishes.

This is not `phase-gated-implementation` with approvals removed, and it grants
no exemption from it. That skill's gate is an owner decision; this one's is a
runnable measurement. A completed run yields an unreviewed branch that then
enters a normal phase under that skill's own gates. A loop iteration is not a
phase and never satisfies one.

## Threat model and what this does not give you

The actors are treated as **fallible, not adversarial**. The loop is *meant* to
defend against crashes, interrupted writes, lost or double-counted
observations, runaway budgets, two actors racing, and silent stopping.

Whether it does is a claim about a specific implementation, not about this
document. Narrowing the threat model licenses dropping audit-grade properties;
it never licenses a defect in the properties that were kept. A run may only
claim the defences its components actually test — the difference between an
accepted risk and an unfixed bug is that the first one is written down here on
purpose.

It does not defend against a deliberately dishonest actor or a privileged actor
escaping its unit cgroup. Runtime source is digest-bound to an approval manifest
and commit, but this is not a general adversary-proof attestation system.

**Therefore: output is operational evidence, not audit-grade evidence.** A
number this loop produces may guide work. It may not be cited as an
independently collected, preregistered measurement. Say so when reporting it.

## Authority boundary

Hold three planes separately.

**Subject** — repository content under study. Editable only inside the run's own
worktree and `scope.allow`, committable only to its own branch. Never push,
open or update a PR, mark ready, merge, tag, delete a branch, touch a remote,
or modify the owner's working tree.

**Control plane** — the run's ledger, results, lease, sentinel, and unit files.
These live outside the worktree so that restoring the worktree cannot destroy
the record of what the run did. They are not an exception to the boundary:
containment grants exactly these paths and denies the rest.

**Host** — everything else. The charter declares every permitted external effect
(network, GPU, container or daemon, shared caches) and containment denies the
rest. An undeclared effect is a stop condition. Declared is not the same as
reversible: an allowed network or daemon effect leaves the local-and-reversible
guarantee behind, so declare as little as the run truly needs.

A permission allowlist is the inner layer, never the boundary. The repository
guide is explicit that permissions are guardrails rather than a complete trust
boundary, and an allow-listed test runs real code with the owner's privileges.
Containment is the boundary.

## Loop classes

**Evidence** — the subject is frozen and the run only accumulates measurements:
repeated trials, flaky-test hunting, counterexample search, cross-configuration
validation. The P4 implementation enforces the unattended entry contract,
resource correspondence, absolute trial bounds, and approval-bound runtime
source. The approved-unit binding is evidenced by bundle3 Runs A/B/C. An
evidence iteration is a collector plus an evaluator and needs no model at all,
which removes the question of whether a model reports honestly rather than
trying to answer it.

**Ratchet** — the run edits the subject and measures it; coverage and
performance goals live here. **Specified below, not built.** Entry conditions
block it until its components exist.

**Search** — the run explores a space where progress is discovered rather than
monotone, so the ratchet binds to the best recorded result instead of HEAD,
allowing temporary retreat while the branch still accumulates only improvements.
**Specified, not built.**

Never widen a class to make a run possible. An evidence run that wants to edit
its subject has become a ratchet run and needs its own charter.

## Components

Every safety property must be enforced by a program, not by adherence to this
document. A property no component checks is not a property of the run.

The working implementation in `experiments/goal-loop-p4/loop/` is the normative
artifact for the evidence class; read it rather than reimplementing from prose.

| Component | Sole authority over |
| --- | --- |
| Containment | host effects, writable paths, resource ceilings |
| Lease | which process may act on this run |
| Collector | producing a measurement or trial result |
| Attempt id | which attempt an artifact is evidence about |
| Evaluator | improvement, stopping, and discard decisions |
| Invariant checker | passing or failing integrity checks (ratchet/search) |
| Ledger writer | appending records |

The model orchestrates, implements subject changes in a ratchet or search run,
and reports. It may not author a measurement, decide a stopping rule is met,
discard a trial, or write a ledger record.

## Entry conditions

Refuse unattended execution and report unless all hold:

- The goal is a command emitting a value plus a threshold, read by the collector
  from a machine-readable artifact under a declared parse contract — never
  scraped from output the subject can format. Model judgment never qualifies.
- A ratio goal declares both the count that must rise and a floor on the ratio;
  guarding one alone lets the other be gamed.
- A performance goal names a correctness oracle with a tolerance, a repeat
  count, a noise band, and a held-out set. Making code faster by making it wrong
  is the default failure of an unsupervised optimizer.
- Safety risk is L1–L2. Security, API/compatibility, migration, release,
  dependency, and credential-adjacent work require owner gates.
- For unattended mode, risk is exactly L1 or L2; scope is present; the subject is
  explicitly frozen and outside all writable paths; the baseline is measured;
  evidence mode explicitly declares `model: null` and the closed headless route;
  external effects are declared; and the driver deadline exceeds the collector
  outer deadline by an explicit termination reserve.
- Unattended approval requires `containment.required` to be exactly `true` and
  statically validates the unit digest, canonical `negative_test.py` source
  path and digests, and a charter-named receipt. Approval reads the host-
  produced receipt's exact bytes, parses the canonical four outcomes, targets,
  and contained execution provenance, and records its path, digest, parsed
  outcomes, and provenance in the external anchor. This is operator-supplied
  evidence: approval parses and binds the recorded provenance but does not
  independently execute or attest it. Inline outcomes, if retained, must match;
  they cannot mint evidence. These checks precede baseline, component, and
  source-manifest work; approval does not run probes or inspect runtime binding.
  Every entry re-reads and re-hashes the bound receipt before any probe, trial,
  or other effect. The live containment and runtime-binding gates remain entry
  checks.
- Trial and termination durations are finite positive values, bounded and
  charged inside `TimeoutStartSec`; `termination_duration_s` equals
  `TimeoutStopSec` for intentional owner-stop paths, and the unit uses
  `TimeoutStartFailureMode=kill` so unattended start expiry has no stop grace.
  `MemoryMax`, `CPUQuota`, and `TasksMax` have finite positive semantic
  correspondence with the unit.
- The charter is complete, its baseline measured rather than assumed, and the
  goal not already met at baseline.
- Every selected-class component has a canonical source path and digest and
  anchor-held local execution evidence. An unattended run is not
  where you discover the invariant checker was never written.
- Containment is verified by negative test — an undeclared write, an undeclared
  network call, and a privilege escalation must each fail. A unit file that
  reads correctly is not evidence.
- The worktree, branch, and lease are ready, and the owner's tree is clean of
  the run's concerns.
- Evidence mode has no model and uses only the closed headless route; modelful
  classes pin an allowed model and never inherit a configured default.
- The route is available or the charter's degrade rule says what to do. An
  unstated degrade rule means stop; never silently substitute inline work.
- Every selected-class component has a canonical source path and digest plus
  anchor-held local execution evidence with a result and test ID. Exact residual IDs are structured in
  `unenforced_risks` and documented under the implementation README's
  `Not defended` heading.
- The approved charter declares an approval-bound declared command-source set.
  Commands
  use structured argv with explicit source paths and dependencies. Python
  scripts, `python -m` modules/packages, shell scripts, and direct executables
  are mechanically resolved to exact canonical targets; opaque, dynamic, or
  unresolvable forms are rejected. Approval writes the canonical manifest and
  commit beside the anchor, and every entry verifies both before mkdir, STOP
  recovery, notification, probes, reconciliation, or trials. Unattended approval
  executes the declared baseline instruction and every fixed component exercise
  with real subprocesses before its first durable write. The external anchor
  owns authoritative execution records; legacy receipt files are not evidence.
  Reapproval verifies those records and does not rerun them. Approval detects
  drift in the tracked control-plane tree, declared source/component inputs, and
  subject (tracked-tree/source drift), but does not isolate or roll them back;
  only approval-owned durable state is absent after refusal.

## Charter and ledger

The charter is the approved artifact; the ledger is the run's state. Owner
approval binds to the exact charter, and the loop may never amend its own.
Recording results is not an amendment. Read
[`references/charter-and-ledger.md`](references/charter-and-ledger.md) before
writing or validating a charter, or before any ledger read or write.

## Iteration

Each iteration runs under the lease and is atomic: one commit or no trace, never
uncommitted work at a boundary.

1. Take the lease. Immediately after locked-FD identity/anchor validation,
   verify runtime identity, then the exact source manifest and all digests.
   Only then inspect the sentinel, create results, recover, notify, probe
   containment, or run a trial. A mismatch is a stop condition, not a repair
    task. A valid terminal STOP proceeds to reconciliation and notification
    without live containment. Source or containment-receipt binding refusal is
    intentionally unnotified: the charter notifier is governed by the failed
    source contract; iteration refusal writes a locked durable stop plus fsynced
    `STOP`, while `stop.sh` refusal writes fsynced `STOP` only. Neither path
    recovers, reconciles, probes, or notifies.
2. Measure through the collector. If the evaluator says the goal is met, stop.
3. Choose the smallest in-scope change plausibly advancing the metric.
4. Implement within the attempt cap, routed per the charter.
5. Re-measure, run the regression check and guard metrics, run the invariant
   checker.
6. Re-check the sentinel, then commit only if the evaluator clears
   `min_improvement`, the diff stayed in scope and under its size cap, nothing
   regressed, and every invariant passed. Otherwise restore and record the
   failure. Append the terminal record either way, then release the lease.

An evidence run replaces steps 3–6 with one collector trial the evaluator
accepts or discards by predicate, stopping when the registered rule holds. It
never edits or commits to the subject.

Recompute stopping from the ledger every iteration rather than trusting a stored
flag, so a crash between the deciding trial and the stop record cannot buy an
observation past the boundary.

A gain below `min_improvement` is not an improvement however positive it looks.
Without that floor a noisy benchmark manufactures improvements forever and the
plateau budget never fires, while a coverage run grinding its hard remainder
dies as a plateau.

Distinguish a failed iteration from failed infrastructure. A crash before any
edit, commit, or completed observation is charged to the retry allowance, not to
the iteration, plateau, or cost budgets. Once work exists — including a
collector artifact on disk — the iteration is real: reconcile and charge it.

## Integrity invariants (ratchet and search)

An evidence run cannot violate these; its equivalent obligations are the
collector's authority over results and the evaluator's over stopping.

**Count and ratio together.** A ratio improves by shrinking its denominator;
a count improves by inflating it with trivially executed lines. Require both.
Where possible restrict `scope.allow` so the denominator is not editable at all,
which defeats the whole class structurally rather than by inspection.

**Protect the enforcement.** Never edit settings, hooks, the containment unit,
`CLAUDE.md`, `AGENTS.md`, skill files, the charter, the components, or the
control plane except through the ledger writer.

**History append-only.** The last good commit stays an ancestor of HEAD.
Recovery reconciles forward and never rewrites.

**Honest measurement.** Goal check, regression check, collector, and their
configuration match the fingerprint — runner options, `conftest.py`, lockfiles,
tool versions. Test and assertion counts must not fall; coverage pragmas, skip,
xfail, and rerun markers must not be added or loosened; broad exception
swallowing must not enter tests; passing tests keep passing; benchmark inputs,
measured paths, and cache state must not be narrowed, hardcoded, or reused.
Confirm a performance gain on the held-out set and confirm green twice before
committing so a flaky pass cannot become a commit.

These raise the cost of gaming without making it impossible. Vacuous assertions
and tests exercising scaffolding stay reachable. The backstop is that nothing
publishes and a human reviews a deliberately small diff.

**Guard metrics.** Floors for metrics that are not the goal. Coverage bought
with doubled runtime is a regression.

## Stop conditions

Stop and report on: goal reached; iteration, wall-clock, plateau, cost, or
retry budget exhausted; consecutive-failure cap; integrity violation;
guard-metric breach; scope escape; undeclared external effect; containment
inactive; lease lost or contended; attempt-cap exhaustion; fingerprint mismatch;
missing or failing component; corrupt or unreconcilable ledger, worktree, or
branch; unreliable clock; unavailable route with no degrade rule; owner stop.

The sentinel and a terminal ledger record are both stop signals, and either
alone ends the run: the sentinel is how an owner stops one by hand, so a loop
that only recognises its own record will keep collecting through an owner stop.

Once stopping, take the terminal-only route. Validate what the terminal sequence
itself will use — it executes the charter's notification command — and nothing
else. Do not probe containment: a stopped run must not act on the world, and a
containment failure must never be able to trap a run that still owes a
notification. Then append the terminal record, notify, and raise the sentinel,
in that order, so nothing that must be recorded is locked out by it.

Re-check the sentinel immediately before accepting any observation, on every
path that can accept one — a fresh trial, a reconciled artifact, and a salvaged
one. Work produced after a stop is recorded as discarded and never adopted
later.

Reaching a budget is a legitimate outcome. Never extend a budget, weaken a
threshold, relax an invariant, or widen containment to keep running. Cost and
usage are budgets and observational evidence; they never change a quality gate.

## Router

Load each reference immediately before its transition, not on entry.

| Trigger | Destination |
| --- | --- |
| Writing or validating a charter; any ledger read or write | [`references/charter-and-ledger.md`](references/charter-and-ledger.md) |
| Installing, verifying, starting, stopping, or recovering the driver | [`references/driver-and-recovery.md`](references/driver-and-recovery.md) |
| Implementing an evidence-class component | `experiments/goal-loop-p4/loop/` |
| Implementation delegation | `opencode-delegate` |
| Code changes made in-loop | `karpathy-best-practices` |
| Owner wants the branch published | `phase-gated-implementation` |
| Resume or cross-repository work | `memory-continuity` |

## Handback

Report the stop reason, baseline and final measurements, iterations attempted
and committed, budget consumption, invariant and containment results, discarded
trials with the predicate that discarded them, and the branch and last good
commit. State that the evidence is operational, not audit-grade. Present the
branch as unreviewed work entering a normal phase — publication is
`phase-gated-implementation`'s decision, never pre-authorized by a run.
