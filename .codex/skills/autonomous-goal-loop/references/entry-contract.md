# Entry Contract

Read before approving unattended entry, changing entry or iteration-start code,
or explaining an entry refusal. The parent skill retains its repository-local
runtime status, authority, budgets, and stop rules; this extraction changes none.

## Unattended entry conditions

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
- Every selected-class component has a canonical source path and digest and
  anchor-held local execution evidence. An unattended run is not
  where you discover the invariant checker was never written.
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

## Iteration start (step 1 in full)

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
