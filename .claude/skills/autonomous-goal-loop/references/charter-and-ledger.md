# Charter and Ledger

Read immediately before writing or validating a charter, or before any ledger
read or write. Return evidence to the core; this decides no stop and no
publication.

`experiments/goal-loop-p4/` is the working reference implementation of
everything below for the evidence class. Where prose and that code disagree, the
code is what actually runs — reconcile deliberately rather than assuming the
prose is right. It has been wrong before.

## Layout

```
<control-plane>/<run-id>/     never under the worktree
  charter.json                approved envelope, immutable after approval
  iterations.jsonl            append-only, ledger writer only
  results/                    collector artifacts, collector only
  lease                       exclusive advisory lock (flock)
  STOP                        sentinel; present means no process may act
<worktree>/                   dedicated worktree on branch loop/<run-id>
```

The control plane sits outside the worktree so restoring the worktree cannot
destroy the record. Containment grants exactly these paths.

## Charter

```jsonc
{
  "run_id": "<slug>",
  "class": "evidence | ratchet | search",
  "model": "null for evidence; a closed allowed value for modelful classes",
  "subject": "<frozen artifact (evidence) or paths under study>",
  "subject_frozen": true,
  "containment": {
    "required": true, "unit": "<path>", "unit_sha256": "<digest>",
    "writable_paths": ["<worktree>", "<control plane>"],
    "probe_source": "<canonical loop/negative_test.py>",
    "probe_source_sha256": "<digest>",
    "probe_targets": {"<canonical filesystem probe>": "<canonical target>"},
    "negative_test": {"receipt_path": "<canonical exact receipt path>", "receipt_sha256": "<required; the exact bytes the charter commits to>"},
    "external_effects": ["<network hosts, GPU, daemons, shared caches>"],
    "verified_at": "<timestamp of the negative test>"
  },
  "resources": {"MemoryMax": "<finite unit value>", "CPUQuota": "<finite percent>", "TasksMax": "<finite count>"},
  "source_manifest": {"manifest_path": "<approval-sidecar>", "commit_path": "<approval-sidecar>", "files": ["<declared canonical paths>"]},
  "measurement": {"argv": ["<exact>"], "cwd": "<canonical existing directory>", "timeout_s": 30, "parser": {"id": "json-value-count-v1", "field": "value", "count_field": "count"}, "source_paths": ["<canonical existing files>"]},
  "baseline": {"value": null, "count": null},
  "components": {"<required name>": {"path": "<canonical source>", "sha256": "<digest>"}},
  "notify": {"command": ["<structured argv>"], "source_paths": ["<explicit dependencies>"]},

  // evidence only — both expressions fixed before the first trial
  "arms": {"<name>": {"argv": ["<exact>"], "cwd": "<exact working directory>", "source_paths": ["<explicit dependencies>"]}},
  "stopping_rule": {"min_trials_per_arm": 20, "max_trials": 400, "ci_width": 0.15},
  "discard_predicate": "<machine-evaluable, never a post-hoc judgment>",

  // ratchet and search
  "goal": {
    "command": "<exact>", "artifact": "<path the collector reads>",
    "field": "<field>", "type": "<expected type>",
    "metric": "<name>", "direction": "increase | decrease",
    "threshold": "<stop condition only>",
    "count_metric": "<count an iteration is judged on>",
    "ratio_floor": "<ratio that must hold while the count rises>"
  },
  "measurement": {
    "repeats": 1, "aggregate": "median", "noise_band": "<baseline spread>",
    "correctness_oracle": "<command, required for a performance goal>",
    "tolerance": "<approved>", "held_out": "<never optimized against>"
  },
  "guard_metrics": [{"metric": "<name>", "command": "<exact>", "floor": null}],
  "fingerprint": {"files": ["<configs, conftest, lockfiles>"], "tools": ["<name==version>"], "sha256": "<hash>"},
  "scope": {"allow": ["<globs>"], "deny": ["<globs>"]},
  "regression_check": "<command that must stay green>",
  "allowed_commands": ["<exact>"],
  "unenforced_risks": [{"risk": "<canonical residual id>", "reason": "<nonempty README-matched reason>"}],
  "search": {"space": "<parameters and ranges>", "best_so_far": {"commit": "<sha>", "value": null}},

  "budgets": {
    "max_iterations": 0, "max_wallclock_s": 0, "max_plateau": 0,
    "min_improvement": 0, "max_consecutive_failures": 0,
    "attempts_per_iteration": 0, "max_iteration_duration_s": 0,
    "max_trial_duration_s": 0, "termination_duration_s": 0, "driver_timeout_s": 0, "max_diff_lines_per_iteration": 0,
    "max_cost_usd": 0, "max_cost_per_iteration": 0,
    "assumed_cost_on_loss": 0, "max_infra_retries": 0
  },
  "route": {"mode": "headless", "primary": null, "degrade": "stop"},
  "risk_level": "L1 | L2",
  "approved_by": "<owner>", "approved_at": "<ts>"
}
```

`scope.deny` must always include the goal and regression configuration,
`conftest.py`, lockfiles, the components, the containment unit, `.claude/**`,
`AGENTS.md`, `CLAUDE.md`, and the control plane. That is the run's own
enforcement surface.

Validate before approval: the baseline is an instruction, never a result or
  receipt; its structured command, canonical cwd, finite bounded timeout, fixed
parser, and explicit source paths resolve. Approval executes it in a scratch cwd
with a new process group, pipes, and the timeout, then derives success from
return code and parsed output. Validate before the first iteration: the parse contract resolves to a real
artifact and field on a trial run; baseline measured at a real commit and not
already past threshold; count and ratio floor both present for a ratio goal;
oracle, tolerance, and held-out set present for a performance goal; `repeats`
above one wherever the metric is noisy; `min_improvement` at least the noise
band; mandatory `scope.deny` entries present; regression check and guard metrics
green; every needed command in `allowed_commands` and every needed effect in
`external_effects`; `driver_timeout_s` equals the unit's `TimeoutStartSec` and
is strictly greater than the collector's outer deadline; the finite positive trial and termination durations fit inside
the start timeout, `termination_duration_s` equals `TimeoutStopSec` for
intentional owner-stop paths, and `TimeoutStartFailureMode=kill` prevents an
uncharged start-failure grace; matching finite positive semantic `MemoryMax`,
`CPUQuota`, and `TasksMax`; every component's canonical source and digest is
bound to anchor-held local execution evidence with a successful result and
test ID; the complete residual registry is documented in README; the declared
  command-source set and every listed digest are verified; containment is verified
  by a recorded negative-test receipt whose unit, probe source, exact probe
  set/outcomes/targets, timestamp, and contained execution provenance bindings
  are statically checked before approval effects -- provenance meaning the
  contained cgroup, invocation id and unit are present *and* the cgroup ends in
  its own unit, since three unrelated non-empty strings are not a binding; and
  the fingerprint is reproducible.

Approval binds the deterministic subject digest, current commit, command and
source digests, and before/after component and subject digests. It reads the
charter-named containment receipt's exact bytes, parses the canonical four
  outcomes and contained execution provenance, and records the path, exact-byte
  digest, parsed outcomes, and provenance in the external anchor; inline outcomes
  must match if present. The receipt is host-observed, operator-supplied
  evidence: approval parses and binds its recorded provenance but does not
  independently execute or attest it. It executes every fixed registry
exercise and stores the observed baseline and exercise records directly in the external anchor before writing the canonical declared
command-source manifest and `commit.txt` beside it. Legacy baseline or component
  receipt files are not evidence. Every iteration verifies runtime identity
immediately after locked FD identity validation, then the manifest and receipt
before results mkdir, STOP recovery, notification, probes, reconciliation, or a
  trial. Binding failure is durably recorded without running notification or
  another post-verification component. Source or containment-receipt binding
  refusal intentionally never invokes the charter notifier because that
  notifier is governed by the failed source contract: iteration refusal writes
  a locked durable stop plus fsynced `STOP`, while `stop.sh` refusal writes
  fsynced `STOP` only; neither recovers, reconciles, probes, or notifies. A
  valid terminal STOP reconciles and notifies without live containment. Approval
  detects drift in the tracked control-plane tree, declared source/component
  inputs, and subject (tracked-tree/source drift), but does not isolate or roll
  them back; only approval-owned durable state is absent after refusal.

Unattended approval does not run the negative test or inspect runtime binding. It
requires `containment.required` to be exactly `true` and validates the recorded
unit digest, canonical `loop/negative_test.py` path and digests, and the exact
receipt bytes named by `negative_test.receipt_path` and pinned by the required
`negative_test.receipt_sha256`. Require that digest and check it before parsing:
binding the bytes into the anchor stops the receipt changing after approval, but
whatever sits at `receipt_path` at approval is what gets bound, so without a
declared digest the charter never said which receipt it meant. The receipt's canonical
probe set, outcomes, targets, timestamp, and sufficient contained execution
provenance are parsed before baseline, component, or source-manifest work. The
host-observed receipt is operator-supplied evidence; approval parses and binds
its recorded provenance but does not independently execute or attest it. Every
entry re-reads and re-hashes the receipt and compares its parsed outcomes and
provenance with the anchor before any probe, trial, or other effect. The live
containment and runtime-binding checks remain separate entry gates.

Approval resolves every structured command to its exact canonical script,
module/package, shell, or executable target and requires that target in
`source_paths` and the manifest. `python -c`, `sh -c`, eval, dynamic namespace,
zip, and opaque loaders are refused.

This source contract does not claim generic closure of runtime imports, inherited
environment, spawned executables, interpreters, or dynamic loaders; those
residuals must be structured in `unenforced_risks` and stated in README.

Evidence additionally: the subject is in `scope.deny` and never `scope.allow`;
`results/` is writable only by the collector; the protocol yields independent
trials rather than reusing warm state; and `stopping_rule` and
`discard_predicate` are expressions the evaluator computes.

**Fix the trial's environment, not just its command.** A trial is an argv *and*
a working directory, and both belong in the charter. Run the exact spec once
before approving it: a command that is green from a repository root can fail to
even collect from one directory up, and recording those failures as observations
would be a confident, entirely wrong result.

**Confirm the trial does not write to its subject.** Ordinary tools mutate what
they observe — `pytest` creates `.pytest_cache` in the repository under test
unless given `-p no:cacheprovider`. Frozen means verified frozen, not assumed.

**A subject digest must cover directories, not just files.** A file-only walk
cannot see an empty directory appear or vanish, and it skips a directory symlink
entirely — the link is correctly not followed, but it was also not hashed, so it
could be swung at a different tree while the freeze check reported no change.
Hash every directory by name and every directory symlink by its link target.

**Derive the arm schedule from the trial index**, as with seeds: arm
`index mod len(arms)`. Nothing then chooses an arm, so no separate preregistered
schedule is needed and no favourable ordering is reachable. Budget by wall clock
rather than trial count when arms differ in cost; a round-robin over arms whose
runtimes differ several-fold is paced by its slowest arm.

Search additionally: `best_so_far` matches a real commit and value, and the
space stays inside `scope.allow`.

Size budgets to the run, not a template. `min_improvement` is load-bearing —
a real count for a discrete metric, a multiple of the noise band for a measured
one — because `max_plateau` means nothing without it. Keep
`max_diff_lines_per_iteration` genuinely small: the final safeguard is a human
reading the branch in the morning, and a night of large commits destroys it.

## Ledger

The writer appends one complete JSON object per line while holding the lease,
with a single write and an fsync. Records carry `at`, `mono`, and a boot
identifier so monotonic readings are never compared across boots. The gap
spanning a reboot is still measured with wall timestamps, because nothing else
survives one: reboot accounting is bounded by how far the wall clock can be
trusted, not independent of it.

Events: `run_start`, `iteration_start`, `trial` (evidence), `discarded`,
`explored` (search), `iteration_end`, `infra_retry`, `notify`, `stop`. Verdict fields
(`stopping_rule_met`, `outcome`, `cleared_min_improvement`) are the evaluator's;
the model may not author them.

**Interrupted and corrupt writes.** A final line without a newline is an
interrupted write. Do not append after it: the two fragments splice into one
syntactically complete but corrupt record and poison every later read. Truncate
back to the last newline first and record how many bytes were dropped. This is
not fabrication — an unterminated tail is a write that never completed, so
nothing acknowledged is lost.

A *complete* line that does not parse is different: the ledger is already
corrupt. Do not append, do not guess, do not repair. Raise the sentinel with the
reason and stop for a human.

*(This corrects an earlier rule that forbade repair outright. Two reviews passed
that rule; one real run corrupted a ledger in minutes. See
`experiments/goal-loop-p4/README.md`.)*

## Resume

Resume from disk alone; never from conversation history, never by inventing a
measurement.

Take the lease first. Held by a live process on this boot means another
iteration is running — exit without acting. Held by a dead process or an earlier
boot is stale and may be reclaimed, recording that it was.

Then: if the sentinel exists or the ledger holds a terminal record, take the
terminal-only route. Validate the charter — the terminal sequence executes its
notification command — complete whatever of that sequence is still owed, and
collect nothing. Do not probe containment on this route: a stopped run must not
act on the world, and a containment failure must never be able to trap a run
that still owes a notification.

Otherwise verify the charter hash, the fingerprint, and that containment is
active. Any mismatch stops.

Classify by the last complete record:

| Last record | Action |
| --- | --- |
| none | validate the charter, measure the baseline |
| `run_start`, `iteration_end` (result not `invariant_violation`), `trial` | proceed |
| `iteration_end` with `invariant_violation` | stop; the previous iteration tried to game its gate |
| `infra_retry` below cap | back off, proceed |
| `infra_retry` at cap | stop |
| `stop` | do not run |
| `iteration_start` or `explored` | reconcile below |

Reconcile against the result artifacts, HEAD, **and** the worktree — never the
worktree alone. A clean worktree does not mean nothing happened.

| State | Action |
| --- | --- |
| the artifact for this attempt exists, and no stop is pending | adopt it as the observation and charge it. A crash must not delete a completed trial, or crash timing biases the result. Bind the artifact to an unpredictable attempt id issued at `iteration_start`: run id, index, arm and seed are all computable in advance, so matching them proves nothing about which attempt produced the file, and a stale or already-discarded artifact would satisfy them |
| no artifact, HEAD == base, clean | nothing landed; `infra_retry`, charge only the retry allowance |
| no artifact, HEAD == base, dirty | edits never committed; restore the worktree, record `reverted`, charge the budgets |
| HEAD is one commit ahead, base is its parent | evidence class: stop — an evidence run must never commit. Ratchet/search: restore the worktree first, then re-measure and run the invariant checker at HEAD; on pass record `adopted` charging `assumed_cost_on_loss`, on fail stop and leave the commit for a human |
| HEAD ahead by more, not a descendant, or base missing | stop as unrecoverable |

Restoring means discarding uncommitted changes inside the worktree, never
rewriting history. `git reset`, `--amend`, and force updates stay denied in
recovery too; that is why an unrecorded commit is adopted rather than erased.

Recompute elapsed time, cost, consecutive failures, and the plateau streak from
the ledger, not from process uptime. Count a plateau step only where
`cleared_min_improvement` is false. Derive elapsed from `mono` within a boot and
`at` across boots; a backward wall-clock jump beyond the charter's tolerance, or
an `at` sequence inconsistent with `mono` inside one boot, stops the run as
unreliable clock rather than guessing which reading to trust.

Never restart a run by deleting its ledger.
