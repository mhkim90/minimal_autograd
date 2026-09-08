# goal-loop P4: smallest working evidence loop

A real, running evidence-class loop, built after two independent reviews
(Codex Terra, OpenCode Sol) returned STOP on the specification-only version of
`autonomous-goal-loop`. It exists to replace prose with evidence.

> **Porting note (airgap-skillset).** This file is carried over from the
> original `experiments/goal-loop-p4/` implementation for historical and
> design context; most of what follows describes that original checkout and
> its verification history, not this package's own deployment. Three
> concrete differences in this package, detailed in `../PORTING.md`:
> the deployed unit example is `loop/goal-loop-p4.service.example`
> (templated placeholder paths, not a live unit — mentions below of
> `loop/goal-loop-p4.service` refer to the original); only
> `evidence/bundle1/` is included, kept as a parser-correctness test
> fixture with its `receipt_path` rewritten to this checkout (not evidence
> about this package's own containment); `evidence/bundle2/` and
> `evidence/bundle3/`, mentioned below, are not included — a receipt from
> one machine is not evidence for another, and no test in this package's
> suite needs them. Before any real unattended run from this package,
> regenerate a fresh receipt for the actual deployment via
> `loop/make_containment_receipt.py`; do not reuse bundle1's bytes as if
> they were evidence about a new deployment.

## Threat model

**The actors here are fallible, not adversarial.** It does not defend against a
deliberately dishonest actor or a privileged actor escaping its cgroup. Runtime
source is now digest-bound, but this is still not a general adversary-proof
attestation system.

What it defends against is exactly what a test in `loop/` demonstrates, and no
more. Narrowing the threat model licenses dropping audit-grade properties; it
never licenses a defect in a property that was kept, and a defence with no test
is a claim, not a property. Known limits are listed under *Not defended* below.

**Consequence, stated plainly:** output from this loop is operational evidence,
not audit-grade evidence. A number produced here cannot be cited as if it were
preregistered, independently collected measurement.

## Why no model in the loop

An evidence iteration is the collector plus the evaluator. Nothing in it needs
judgment, so nothing in it calls a model. That removes the entire question of
whether the model can be trusted to report honestly, which was the hardest
finding in both reviews, at the cost of nothing this class needed.

## Shape

```
subject/flaky.py      frozen subject; fails at a rate the loop does not know
loop/collector.py     runs one trial, writes one JSON artifact
loop/evaluator.py     Wilson interval; owns the stopping decision
loop/ledger.py        append-only JSONL, fsync, heals interrupted writes
loop/iterate.py       one iteration: reconcile, budget, trial, evaluate
loop/preflight.py     entry conditions as a check, not a paragraph
loop/clock.py         elapsed time across boots, and clock-tamper detection
loop/notifier.py      delivers the terminal record; failure is not silent
loop/run.sh           locks the ledger fd, then one iteration
loop/stop.sh          blocks on the ledger-fd lock, then a durable owner stop
loop/stop.py          the locked stop protocol; identical to the trial writer's
loop/testkit.py           test helpers that assert *why* a run stopped
loop/safety_test.py       invariants the loop is *for*; a change failing one is wrong
loop/bounds_test.py       process, budget, validation and containment bounds
loop/red_gate_test.py     tests written before their fixes, each seen failing first
loop/regression_test.py   the earlier suite, with its weak checks strengthened
loop/bundle_b4_b6_test.py B-4..B-6 durable entry, budget, cgroup and source gates
loop/security_bundle_test.py  supervised/unattended entry separation, probe targets
loop/ledger_fd_lock_test.py   ledger-fd lock and out-of-run-dir identity anchor
loop/driver_reserve_and_digest_test.py  required driver reserve; directory-aware digest
loop/receipt_declaration_test.py  the receipt digest and cgroup/unit provenance binding
loop/source_manifest.py   approval-bound declared command-source set
loop/negative_test.py     the four probes; contained and uncontained
loop/make_containment_receipt.py  emits one half of a receipt from probes run here
loop/runtime_binding_probe.py     prints the live systemd binding and its verdict
evidence/bundle1/         four-probe receipt, its two raw halves, and the
                        declaration that names and pins it
evidence/bundle2/         cgroup backstop kill receipt
evidence/bundle3/         live approved-unit binding receipt, runs A/B/C
.loop/<run-id>/       control plane: charter, ledger, results, STOP
<parent>/<run-id>.anchor  identity anchor, OUTSIDE the run dir so directory
                       replacement cannot recreate it
<parent>/<run-id>.source-manifest.json + commit.txt
                        approval-bound, canonical declared command-source evidence
```

Neither seeds nor arms are chosen. Arms rotate by trial index, and each arm
draws from its own seed stream keyed by `(run, arm, position in that arm)`.
Deriving the seed from the global index while rotating arms by that same index
confounds them: with two arms one gets every even seed and the other every odd
one, which manufactures an arm difference with no adversary involved.
The stopping decision is recomputed from the ledger every iteration
rather than read from a stored flag, so a crash between the deciding trial and
the stop record cannot buy an extra observation past the boundary.

## Verified by execution, not by review

| Property | How it was shown |
| --- | --- |
| Lease excludes a second actor | held the lock externally; the iteration declined and exited 0 |
| A crash after a trial completes does not lose it | an artifact valid for the current index is adopted before any new trial starts, covering both a crash and a collector that published and then failed |
| A crash before a trial produces nothing charges a retry | injected `iteration_start` with no artifact; the run recorded `infra_retry` |
| An interrupted write does not corrupt the ledger | injected a truncated line mid-run; 24 bytes were healed and all 72 records still parse |
| A corrupt ledger halts instead of guessing | the poisoned ledger stopped the run and raised the sentinel; preserved at `.loop/_evidence/` |
| The terminal record survives the stop | `stop` is appended before the sentinel is raised, confirmed by sequence and mtime |
| The stopping rule actually fires | 35 trials, stopped on `ci_width satisfied`; estimate 0.314 against a true rate of 0.30 |
| A stop reaches a human | an ordinary validated terminal stop sends the run id, reason, and summary as JSON to the charter's `notify.command`, recorded as a `notify` event; source or containment-receipt binding refusals are an explicit unnotified residual |
| A failed notification is not silent | an unreachable channel and an undeclared channel both printed `UNRESOLVED`, recorded the error, and exited 3 rather than 0 |
| The defects reviews found stay fixed | `loop/red_gate_test.py` (10) and `loop/regression_test.py` (11), both written or corrected before their fixes and observed failing first |
| Owner stop is serialized and durable | `loop/safety_test.py`, `loop/ledger_fd_lock_test.py`. `run.sh` and `stop.sh` both take an exclusive flock on the same object they write — the ledger file descriptor itself — so a stop cannot interleave a trial: trial-first fsyncs before the stop can lock; stop-first is seen under the lock and no trial is appended. A separate lease file no longer exists. A bare `touch STOP` is a pre-start hint only. |
| Ledger identity is bound to an out-of-run-dir anchor | `loop/ledger_fd_lock_test.py`. Charter approval (`preflight.approve`) is the one durable bootstrap: it creates the ledger, then writes an anchor OUTSIDE the run directory recording the ledger's (dev, ino). Every continuation validates fd == anchor == path under the lock, before and after the append; any replacement or partial bootstrap fails closed as corruption. This defends against a run dir or ledger being replaced by an operator or cleanup **mistake** — not against an adversary who can hardlink or swap the same inode back. |
| Budget decision is linearized before the trial append | `loop/bounds_test.py`. Exactly one elapsed-time read makes the decision; a deadline crossing during write or fsync does not revoke an accepted trial |
| Process, budget, validation and containment checks | `loop/bounds_test.py` (41). Named checks, not a claim that all bounds are hard — see *Not defended* |
| Unattended entry contract | `loop/bundle_b4_b6_test.py` — risk, scope/frozen subject digest, anchor-owned local observed execution records, explicit model-free route, effects, resource correspondence, static containment receipt validation before approval effects, active unit binding, and documented residuals refuse with durable reasons |
| Whole-iteration and whole-tree trial bounds | `loop/bundle_b4_b6_test.py`, `loop/bounds_test.py`; the runtime uses the remaining absolute budget, and the unit uses `TimeoutStartSec` plus `KillMode=control-group` |
| Approval-bound declared command-source set | `loop/bundle_b4_b6_test.py`; approval writes `commit.txt` and a self-describing manifest, then every unattended iteration entry verifies the declared command targets and all listed digests while holding the ledger lock |
| Containment probe implementation | approval reads the exact bytes of the charter-named, host-produced bundle1 receipt, parses and anchor-binds the four outcomes, contained execution provenance, unit/source digests, and targets without live probes; every entry re-reads and re-hashes that receipt before any effect, while preflight still runs the four-probe set at unattended entry |
| Current privilege control | `unshare --user --map-root-user -- id -u` succeeded uncontained on this machine and verified UID 0. Permission denial is classified as blocked; ambiguous, nonzero, timeout, and missing-executable results are inconclusive |
| Supervised waiver has a distinct approved entry | `run.sh` is always unattended and clears `GOAL_LOOP_SUPERVISED`; only `run-supervised.sh` passes internal supervised mode, which requires an approved `execution_mode: supervised` charter and refuses `INVOCATION_ID` |
| Filesystem probe targets are charter-bound | the two filesystem targets are exact, absolute canonical charter entries, checked against every canonical writable path with component-safe and symlink-resolving comparisons; live probes receive those entries directly |

## Why the green numbers changed meaning

The suites once reported 51 passing while two of them never read the artifacts
they were named for. They passed because a fixture's hardcoded timestamp looked
like a tampered clock, the run stopped for that unrelated reason, and the checks
only asked whether it had stopped.

`loop/testkit.py` exists to make that impossible: `stopped_because` fails when a
run stops for anything other than the reason under test, and `stamp()` builds
fixtures whose wall and monotonic components agree. Applying it turned green
checks red immediately, which is what a suite is for.

## Containment evidence

`evidence/v6/` is historical evidence for an earlier implementation. It collects
the commit, charter and digest, unit and timer, systemd status, journal,
notification, ledger, and `SHA256SUMS`, but its negative test used the retired
setuid-bit probe and ambient filesystem targets. It must not be cited as current
B-1..B-3 probe evidence. It is also a structured self-report, not an independent
attestation.

The historical v6 negative test recorded each probe with both outcomes (not a
current receipt):

| Probe | Uncontained | Contained |
| --- | --- | --- |
| write outside permitted paths | succeeded | blocked |
| write into the repo outside the run dir | succeeded | blocked |
| undeclared network connection | succeeded | blocked |
| retired setuid-bit probe (historical only) | succeeded | blocked |

The current privilege probe actually attempts user-namespace UID 0 and verifies
the resulting output. The host-produced `evidence/bundle1/negative_test_receipts.json`
is the current receipt: control records 4/4 succeeded and contained records 4/4
blocked, with the identical probe digest and contained cgroup, invocation ID,
and unit provenance. It is operator-supplied evidence; approval binds and parses
it but does not independently execute or attest it. The table above remains
historical and used the retired probe.

The current unit is `loop/goal-loop-p4.service`. It uses the default unattended
`run.sh`, clears the legacy supervision variable, sets `RestrictNamespaces=yes`,
declares the writable state path, and sets `TimeoutStartSec=600`,
`TimeoutStartFailureMode=kill`, `TimeoutStopSec=1s` for intentional owner-stop
paths, `KillMode=control-group`, `MemoryMax=512M`, `CPUQuota=50%`, and
`TasksMax=64`. The approved driver deadline is strictly greater than the
collector's outer deadline. Before launch, the driver computes an absolute hard
deadline from the iteration start, then gives the trial that deadline minus the
declared termination reserve. The collector's outer deadline is the trial
deadline plus that reserve, still clipped to the hard deadline, so startup
overhead is charged and collector cleanup runs before the unit backstop. An
explicit supervised charter uses its declared reserve; older supervised
charters without one use a bounded one-second compatibility default. Unattended
start expiry still uses immediate cgroup kill rather than the owner-stop bound.
The controller's `evidence/bundle3/runtime_binding_receipt.txt` supplies the
approved-unit binding evidence through Runs A, B, and C; the receipt explains
which unit-name, invocation, and cgroup checks each run discriminates.

Aggregate counts are no longer accepted: `blocked == total == control_succeeded`
was satisfied by zero probes. The charter names an exact receipt file; approval
reads and hashes its exact bytes, parses each canonical probe and both outcomes,
and records the receipt path, exact-byte digest, targets, timestamp, parsed
outcomes, and contained execution provenance in the external anchor. The
host-produced `evidence/bundle1/negative_test_receipts.json` is operator-supplied
evidence, and `evidence/bundle1/receipt-declaration.json` is the charter fragment
that names it and pins its digest. That file is deliberately not a run charter —
it declares no execution mode, budgets, arms or components, and approval will not
accept it as one. Its only job is to state which receipt bytes a charter means.

`negative_test.receipt_sha256` is **required**, and checked before the receipt is
parsed. Binding the bytes into the anchor stops the receipt changing after
approval; it does not stop a different receipt being in place at approval, since
whatever sits at `receipt_path` is what gets bound. The declared digest is the
charter committing in advance to which file it meant. It was decoration for one
round — present in the evidence file and absent from every line of `loop/`, which
is the shape a commitment takes when nothing reads it.

Approval parses and binds the recorded provenance, but does not
independently execute or attest the receipt. If inline outcomes remain in the
charter, they must match the receipt and cannot mint evidence. Before approval,
preflight hashes and path-binds the canonical probe implementation and unit and
checks the receipt's source digest, timestamp, and sufficient contained
execution provenance. Every entry re-verifies the receipt digest, parsed
outcomes, and anchor binding before a probe, trial, notification, or other
effect. Approval does not run a live probe or inspect runtime binding; those
remain separate live entry gates.

## Not defended

- **privileged_cgroup_escape** — privileged actors may escape the unit cgroup
- **unavailable_cgroup_controllers** — resource controllers may be unavailable
- **pre_existing_external_daemons** — pre-existing external daemons are outside this run
- **post_verification_source_toctou** — post-verification source mutation is outside this fallible-actor contract
- **notification_exactly_once** — notification is at-least-once rather than exactly once
- **reboot_wallclock_accounting** — reboot gaps rely on wall-clock accounting
- **probe_target_toctou** — probe target validation and use are not one no-follow kernel operation
- **operator_supplied_containment_control** — the operator-supplied receipt records host-observed four-probe outcomes and contained execution provenance; approval binds and parses its bytes but does not independently execute or attest them
- **source_or_receipt_binding_refusal_unnotified** — source or containment-receipt binding refusal intentionally does not invoke the charter notifier because that notifier is governed by the failed source contract; iteration refusal writes a locked durable stop and fsynced STOP, while stop.sh refusal writes a fsynced STOP only, and neither path recovers, reconciles, probes, or notifies
- **approval_child_effects_not_rolled_back** — approval detects drift in the tracked control-plane tree, declared source/component inputs, and subject (tracked-tree/source drift) and refuses without isolating or rolling them back; only approval-owned durable state is absent
- **post_verification_tampering** — an actor may alter the verifier or anchor after verification
- **bare_stop_during_iteration** — a bare STOP during a trial is only converted on the next entry
- **post_decision_budget_crossing** — a deadline may pass during trial append or fsync after the decision
- **clock_anomaly_granularity** — backward wall-clock and wall-monotonic disagreement share one anomaly check
- **artifact_extra_fields** — unknown extra artifact fields are carried through unexamined
- **ledger_event_value_domains** — recognised ledger event value domains are not fully checked
- **inherited_environment** — the trial environment is inherited wholesale
- **undeclared_runtime_imports** — Python runtime imports beyond declared command sources are not generically closed
- **spawned_executables** — executables spawned by a declared command are not generically closed
- **shell_interpreter_loader** — shell, interpreter, and dynamic-loader bindings are not a complete runtime closure
- **commit_marker_limitations** — commit.txt proves the recorded repository marker only, not every runtime byte
- **timer_behavior** — the timer is not disabled on stop
- **local_approval_not_external_attestation** — approval is local execution evidence, not an independent external attestation
- **measurement_command_semantics** — the measured command's inherited runtime semantics are not independently attested
- **component_exercise_coverage** — registered component exercises cover the fixed profiles, not arbitrary implementations

The hermetic `_unattended` test fixture executes only the control half and
synthesizes contained `blocked` values and provenance so unrelated approval
gates can be exercised. Those fields are test scaffolding, not evidence. The
positive anchor gate instead points at the host-produced bundle1 receipt and
binds its exact bytes, parsed outcomes, and contained provenance.

- **Exactly-once notification.** The guarantee is at-least-once. `notifier`
  delivers and then records; a crash between those delivers twice on recovery.
  Closing it needs an idempotency key the channel honours.
- **Reboot accounting without trusting the wall clock.** Monotonic readings are
  comparable only within one boot, so the gap across a reboot is measured with
  wall timestamps. Downtime is now counted rather than skipped, but a clock reset
  that still lands slightly later undercounts it.
- **Containment's control side is operator-supplied and remains residual.** The
  host-produced bundle1 receipt records all four control outcomes, all four
  contained outcomes, and contained cgroup, invocation ID, and unit provenance.
  Approval binds and parses its exact bytes but does not independently execute
  or attest them and does not require a live user bus, so the receipt is not
  independent attestation.
  The live contained side still runs the four canonical probes and refuses
  unless all four are blocked; unrelated errors are inconclusive. Bundle3 is
  unit-binding evidence, not FOUR-PROBE NEGATIVE-TEST control evidence.
- **Source or receipt binding refusal is intentionally unnotified.** The charter
  notifier is governed by the failed source contract, so these refusals never
  invoke it: iteration refusal produces a locked durable stop plus fsynced
  `STOP`, while `stop.sh` refusal produces fsynced `STOP` only. Neither path
  recovers, reconciles, probes, or notifies.
- **Approval child effects are not rolled back.** Approval detects drift in the
  tracked control-plane tree, declared source/component inputs, and subject
  (tracked-tree/source drift) and refuses without isolating or rolling them
  back. Only approval-owned durable state is required to be absent after
  refusal.
- **Probe-target TOCTOU.** Targets are canonicalized and checked before use, but
  validation and the filesystem write are not one no-follow kernel operation;
  an adversary swapping paths between them is outside this fallible-actor model.
- **Privileged/adversarial escape.** Bundle3 live binding Runs A/B/C provide
  approved-unit acceptance and rejection controls in this checkout. They do not
  cover privileged escape, negative-probe completeness, an actor able to alter
  the verifier or anchor after verification, pre-existing external daemons, or
  unavailable cgroup controllers.
- **A bare STOP raised during an iteration.** `touch STOP` is only a pre-start or
  iteration-entry hint. If it appears during a trial, that current trial may be
  accepted; the next entry converts the sentinel to the durable owner-stop
  record. Use `loop/stop.sh` when the owner needs lease-serialized authority.
- **Budget at durability after the decision point.** The owner stop is
  serialized by the lease, and the budget decision is the elapsed-time read just
  before the trial append. A deadline can pass during that append or its fsync;
  the contract makes no promise that it had not passed at durability time.
- **Clock-anomaly granularity.** A backward wall clock and a wall/monotonic
  disagreement are caught by the same comparison, not by separate rules.
- **Artifact structure beyond identity and result shape.** Unknown extra fields
  are carried through unexamined.
- **A recognised-but-wrong ledger event.** Event names and per-event required
  fields are checked, but not value domains within them.
- **The trial environment is inherited wholesale** rather than fixed by the
  charter.
- **The timer is not disabled on stop.** A stopped run is cheap but still fires.

Every remaining deliberate risk must also be an object with `risk` and `reason`
in the approved charter's `unenforced_risks`; B-4 refuses a charter whose
README entry is missing. The list above is the rationale for the accepted
rootless and operational limits, not a claim that they are defended.

### Systemd evidence boundary

`evidence/bundle3/runtime_binding_receipt.txt` records Runs A, B, and C: A
accepts the approved unit binding, B refuses when `SYSTEMD_UNIT` is removed,
and C refuses an otherwise identical process outside the unit. Those runs
discriminate the unit-name, invocation, and cgroup checks. They do not prove
in-process collector ordering. `evidence/bundle2/cgroup_kill_receipt.txt`
proves only that the systemd backstop kills a `setsid()` descendant that ignores
SIGTERM; the driver reserve is a separate approval check.

## Withdrawn claims

An independent review found that several rows of the table above were not true
of the code, and that some evidence cited for them came from an earlier build.
They are withdrawn rather than quietly reworded:

- **"Containment holds, 4/4."** This project's own rule is that every probe must
  succeed without containment before a contained result means anything. Only
  3 of 4 did — `sudo` was blocked by the environment, not by the sandbox — so
  the control was incomplete and the result overstated. The probe also reports
  "blocked" when a write succeeds but its cleanup fails.
- **"Runs unattended end to end, 78 trials."** That run predates the current
  implementation: it used `min_trials`, not `min_trials_per_arm`, and its
  records carry no boot id. It is evidence about an earlier build.
- **"The sentinel outlasts the timer."** True of the timer, but the sentinel was
  checked only at iteration entry, so a stop raised during a long trial did not
  prevent that trial's result from being accepted.
- **"Elapsed time survives a reboot."** The boot id was recorded and never used;
  elapsed time came from wall timestamps with the timezone stripped. `clock.py`
  now segments by boot and detects wall/monotonic disagreement, but cross-boot
  gaps still rest on wall timestamps, so the original phrasing overstated it and
  there is no test for reboot segmentation itself.
- **bench A/B medians** were printed alongside the result and invited a
  performance reading. The evaluator stopped on a pass/fail confidence interval
  and made no duration comparison at all.

## What the unattended test found

Two assumptions in the written driver guidance were wrong, and only running it
showed that:

- `Linger=no` was the machine's actual state. An overnight run started without
  checking would have died silently at logout.
- `ProtectHome=tmpfs` cannot work when the worktree, control plane, or
  interpreter live under `/home`: replacing home with a tmpfs removes the paths
  `ReadWritePaths=` must bind, and the unit fails with `status=226/NAMESPACE`.
  `ProtectHome=read-only` is correct here. A user service also does not inherit
  the login shell's `PATH`, so it must be set explicitly.

The negative test was run without containment first and 3 of 4 probes
succeeded. Without that control, the contained 4/4 result would not have
distinguished real isolation from a probe that never tried anything.

## Real targets

Two runs against real subjects rather than the toy, both driven by the same
loop with arms scheduled as `index mod len(arms)`:

- **MCP flaky hunt** — 32 trials over `usage_mcp`, `memory_mcp`, `claude_mcp`,
  `opencode_mcp`. Zero failures; per-arm 95% CI `[0, 0.324]` at n=8.
- **bench A/B** — 30 trials over `with_skills` and `without_skills`. Zero
  failures on both arms.

Both ran on the previous implementation, and their charters omitted containment,
scope, and approval fields that the skill lists as entry conditions. That they
ran at all is the finding: the conditions existed only as prose. `preflight.py`
now refuses such a charter, and both runs would need repeating before their
numbers mean anything about the current code.

Neither result means "not flaky". Zero failures in eight trials bounds the rate
below 0.324, and ruling out a practically interesting 1–5% flake rate needs
hundreds of trials per arm. `min_trials_per_arm: 8` was chosen to exercise the
loop, not to answer the question. The evaluator reporting an interval instead of
folding zero failures into "safe" is the behaviour that matters here.

What only the real targets exposed:

- `pytest` writes `.pytest_cache` into the repository under test, violating the
  evidence class's no-write-to-subject rule. Observation tools have side
  effects; frozen has to be verified, not assumed.
- Running the same command from one directory up turned three of four green
  suites into collection errors. Without `cwd` in the trial spec the run would
  have recorded 32 confident, entirely wrong observations.
- Arm runtimes differ sevenfold (1.6s to 11.7s), so a trial-count budget does
  not predict wall clock; the slowest arm paces the run.

## The bug that mattered

Both reviews examined a written rule that a truncated final line must be treated
as absent and never repaired. Neither caught that appending after such a line
splices the two fragments into one syntactically complete but corrupt record,
poisoning every later read. One real run found it in minutes.

The written rule was also wrong. Truncating back to the last newline is not
fabrication: an unterminated tail can only be a write that never completed, so
nothing acknowledged is lost. `ledger.py` now heals on append and records how
many bytes it dropped.
