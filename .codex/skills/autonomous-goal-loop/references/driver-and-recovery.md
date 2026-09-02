# Driver and Recovery

Read immediately before installing, verifying, starting, stopping, or recovering
a driver. Return evidence to the core; this decides no stop and no publication.

`experiments/goal-loop-p4/loop/run.sh` is the working lease-and-sentinel entry
point. Reuse its shape rather than reinventing it.

## What the driver is

The driver is the only thing that survives the session. It fires one iteration
at a time under containment; that iteration takes the lease, reads the ledger,
does at most one unit of work, and exits. State lives in the ledger, never in
the driver and never in a conversation.

Do not use in-session schedulers for an unattended run. `CronCreate` is
in-memory and dies with the session; `/loop` with `ScheduleWakeup` needs the
session alive. Both are fine for supervised runs of a few minutes; neither
survives a closed terminal.

## Containment

Containment, not the permission allowlist, is the trust boundary. An
allow-listed test runs real code with the owner's privileges; the allowlist
governs which commands the model invokes, not what they reach once running.

```ini
[Service]
Type=oneshot
WorkingDirectory=<worktree>
TimeoutStartSec=<driver_timeout_s; greater than the collector outer deadline>
TimeoutStopSec=<termination_duration_s>
TimeoutStartFailureMode=kill
KillMode=control-group
SendSIGKILL=yes
ExecStart=<control-plane>/<run-id>/run.sh <control-plane>/<run-id>

ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=<worktree> <control-plane>/<run-id>
Environment=PATH=<toolchain bin>:/usr/bin:/bin
PrivateTmp=yes
NoNewPrivileges=yes
RestrictSUIDSGID=yes
PrivateDevices=yes          # relax explicitly for a declared GPU run
PrivateNetwork=yes          # see below before relaxing
MemoryMax=<ceiling>
CPUQuota=<ceiling>
TasksMax=<ceiling>
```

`ProtectHome=tmpfs` looks stricter and does not work when the worktree, control
plane, or toolchain live under `/home`: replacing home with a tmpfs removes the
very paths `ReadWritePaths=` must bind, and the unit fails to start with
`status=226/NAMESPACE`. Use `read-only`, which leaves home readable, blocks
writes everywhere in it, and lets `ReadWritePaths=` punch through for the run.
`ProtectSystem=strict` alone is not enough — it does not cover `/home` at all.

Set `Environment=PATH=` explicitly. A user service does not inherit the login
shell's PATH, so an interpreter installed under home is otherwise not found.

Where systemd cannot express what a run needs, invoke `bwrap` from the entry
script with the same allow-and-deny shape. Either way: the writable set is
enumerated, network and devices are denied unless the charter names them, and
resource ceilings are real so a runaway iteration cannot take the machine down.

`RestrictAddressFamilies` limits protocol families, **not destination hosts**.
Once networking is on, undeclared hosts stay reachable without separate egress
filtering. Prefer `PrivateNetwork=yes`; if a run needs specific hosts, filter
egress explicitly and record that the local-and-reversible guarantee no longer
covers those effects.

Verify containment with a negative test before declaring a run unattended: an
undeclared write, an undeclared network call, and a privilege escalation must
each fail through the same invocation path the driver uses. Record the result. A
unit file that reads correctly is not evidence it is loaded and effective.

Run the same probes **without** containment first and require them to succeed.
Without that control, an all-blocked result cannot be distinguished from a probe
that never tried anything, and the test proves nothing.
`experiments/goal-loop-p4/loop/negative_test.py` is the working probe set.

For unattended approval, the charter names a receipt path. Approval reads its
exact bytes, hashes those bytes, parses the canonical four probe outcomes,
targets, and contained execution provenance, and stores the path, exact-byte
digest, parsed outcomes, and provenance in the external anchor. The
host-observed receipt is operator-supplied evidence: approval parses and binds
its recorded provenance but does not independently execute or attest it.
It binds the unit and canonical probe implementation by path and digest, and
requires `containment.verified_at` to equal the receipt timestamp and the receipt
to contain sufficient contained execution provenance. Inline outcomes, if
retained, must match the receipt and cannot mint results. These
checks precede any approval-owned subprocess or durable write; approval does
not run the probes or read runtime/systemd binding. Every entry re-reads and
re-hashes the bound receipt before any probe, trial, or other effect. Those live
checks remain separate entry gates.

## Timer

```ini
[Timer]
OnBootSec=2min
OnUnitInactiveSec=<gap between iterations>
AccuracySec=1s

[Install]
WantedBy=timers.target
```

`OnUnitInactiveSec` measures from when the last iteration finished, so the gap
is real and iterations pace themselves against their own duration; `OnCalendar`
fires on wall-clock and degrades as iterations grow. Set `AccuracySec`
explicitly — it defaults to one minute, which silently swallows shorter gaps.

`Type=oneshot` serializes starts of this unit and nothing else. It is not mutual
exclusion: a manual run, a duplicated unit, or a process outliving its timeout
still produces two actors. **The lease provides exclusion; the timer only
provides cadence.** Every entry point takes the lease, including a human running
an iteration by hand.

Choose the gap from the goal. Coverage or correctness runs should be effectively
continuous, around 30 seconds — an iteration already takes minutes and idle time
buys nothing, while cost is bounded directly by the cost budget. Performance
runs need a settling gap, roughly 60 to 120 seconds, so clocks and thermals
return to a stable state; that is measurement validity, not a throughput
tradeoff, since measuring on the previous iteration's heat turns drift into
apparent improvement.

Keep `driver_timeout_s` and `TimeoutStartSec` equal, and make the driver deadline
strictly greater than the collector outer deadline. The field is **required**,
never defaulted: an omitted reserve once fell back to the in-process hard
deadline, which put the cgroup kill at the same instant as the in-process kill
and record, so leaving the field out silently selected the racing configuration.
Omission must be a refusal, not a default. Reserve
`termination_duration_s` inside that start timeout; the trial/work deadline is
no later than the remaining usable interval. `TimeoutStartFailureMode=kill`
makes start expiry an immediate cgroup kill with no uncharged stop grace.
`TimeoutStopSec` is the same finite bound for intentional owner-stop paths. The
in-process process-group kill is best effort. The existing controller receipt at
`experiments/goal-loop-p4/evidence/bundle2/cgroup_kill_receipt.txt` proves only
that the cgroup backstop kills a `setsid()` descendant that ignores SIGTERM; it
  does not prove in-process ordering. Bundle3's controller receipt records Runs
  A/B/C and discriminates approved-unit acceptance, missing `SYSTEMD_UNIT`, and
  an outside-unit control. The collector and driver add no uncharged timeout
  grace period.

Enable with `systemctl --user enable --now loop-<run-id>.timer` and run
`loginctl enable-linger $USER`, or the units stop at logout — which defeats the
entire purpose. Verify linger is active before calling a run unattended.

## Headless invocation (ratchet and search only)

An evidence iteration needs no model. Only classes that edit the subject invoke
one:

```sh
claude -p "$PROMPT" --model <charter model> --output-format json \
  --max-turns <cap> --permission-mode acceptEdits \
  --add-dir <worktree> --settings <worktree>/.claude/settings.json
```

Pin the model from the charter; never inherit a configured default. Keep the
prompt small and stateless — it names the run ID and control-plane path and
nothing else. History, measurements, and approval come from the ledger, the
charter, and the components.

Parse `session_id` and `total_cost_usd` and pass both to the ledger writer;
cumulative cost comes from summing the ledger, which is why it survives a crash.
When a crash loses the reported cost, charge `assumed_cost_on_loss` and mark the
source as assumed, so a crash loop cannot spend without appearing to.

## Permissions

Permissions are the inner layer. Translate `allowed_commands` into
`permissions.allow`, and deny `git push`, PR and merge operations,
`git commit --amend`, `git reset`, force updates, any write outside the writable
set, and any write to `scope.deny` — which includes the settings file itself,
the components, and the containment unit. A run able to edit its own permissions
has no permissions.

An evidence run gets a strictly tighter set: the trial command only, with no
write access to the subject and no direct write to the ledger or results, which
belong to the ledger writer and the collector.

Never use `--dangerously-skip-permissions`. An insufficient allowlist blocks the
iteration, which is the correct failure: fail closed, record the blocked
command, wait for a human.

## Stopping and recovery

To stop: append the terminal record, raise the sentinel, then
`systemctl --user disable --now loop-<run-id>.timer`. That order matters — the
ledger writer refuses to append once the sentinel exists, so raising it first
makes the required record impossible. The sentinel is re-checked immediately
before any commit or accepted observation, so a long iteration cannot land work
after the owner asked it to stop.

Notify through the charter's channel on every ordinary validated stop. Source or
containment-receipt binding refusal intentionally never invokes the charter
notifier because that notifier is governed by the failed source contract:
iteration refusal writes a locked durable stop plus fsynced `STOP`, while
`stop.sh` refusal writes fsynced `STOP` only. Neither path recovers, reconciles,
probes, or notifies. A stop that cannot be notified otherwise stays unresolved
rather than exiting quietly.

For recovery the ledger is authoritative and the reconciliation table in
[charter-and-ledger.md](charter-and-ledger.md) is the only permitted procedure.
Never resume by deleting the ledger, editing a past record, re-hashing a
mismatched fingerprint, or rewriting history. If worktree, branch, and ledger
cannot be reconciled, stop as unrecoverable and hand it to a human.

Before removing a finished run's worktree, units, and control plane, confirm the
branch is handed off. The branch holds the only copy of the work.
