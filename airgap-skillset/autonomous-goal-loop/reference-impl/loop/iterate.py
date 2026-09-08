"""One iteration of an evidence-class loop.

Threat model: the model and this code are fallible, not adversarial. This
defends against crashes, runaway budgets, silent stopping, and lost
observations. It does not defend against a deliberately dishonest actor, so its
output is operational evidence and not audit-grade evidence.

The ledger FD is the lock. It remains held from iteration acceptance through
the durable append, so an owner stop cannot pass a trial writer.
"""
import json
import os
import secrets
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock
import evaluator
import ledger
import notifier
import preflight

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUPERVISED_TERMINATION_DURATION_S = 1.0


def boot_id():
    with open("/proc/sys/kernel/random/boot_id") as fh:
        return fh.read().strip()


def now():
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mono": round(time.monotonic(), 3), "boot": boot_id()}


def sentinel(run_dir):
    return os.path.join(run_dir, "STOP")


def halt(run_dir, reason, trusted=False):
    """Fail closed; malformed preflight inputs get only a local refusal sentinel."""
    if not trusted:
        # Before charter shape/identity can be read there is no safe ledger
        # append, but an existing run directory can still durably remember the
        # named refusal for the timer and testkit.
        try:
            if os.path.isdir(run_dir):
                with open(sentinel(run_dir), "w") as fh:
                    fh.write(reason + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        except OSError:
            pass
        print(f"STOP: {reason}", file=sys.stderr)
        raise SystemExit(2)
    with open(sentinel(run_dir), "w") as fh:
        fh.write(reason + "\n")
    print(f"STOP: {reason}")
    raise SystemExit(2)


def append_locked(fd, ledger_path, anchor_path, run_id, record):
    """Check immediately around the one append operation."""
    ledger.validate_identity(fd, ledger_path, anchor_path, run_id)
    result = ledger.append_fd(fd, ledger_path, record)
    ledger.validate_identity(fd, ledger_path, anchor_path, run_id)
    return result


def _finish_locked(run_dir, ledger_path, charter, reason, summary, fd):
    """Drive the terminal sequence to completion, from wherever it stopped.

    Crashing partway through a stop used to be unrecoverable in both
    directions: a crash after the stop record left a run that never notified
    and never raised its sentinel, and a crash after notifying could notify
    twice. Deriving each remaining step from the ledger makes the sequence
    idempotent, so any number of interrupted attempts converge on one stop, one
    notification, and one sentinel.
    """
    records = ledger.read_fd(fd, ledger_path)
    stop_record = next((r for r in records if r["event"] == "stop"), None)

    if stop_record is None:
        append_locked(fd, ledger_path,
                      preflight.anchor_path(run_dir, charter["run_id"]),
                      charter["run_id"], {"event": "stop", "reason": reason,
                                          "summary": summary, **now()})
    else:
        reason, summary = stop_record["reason"], stop_record["summary"]

    if any(r["event"] == "notify" and r.get("delivered") for r in records):
        delivery = {"delivered": True}
    else:
        delivery = notifier.notify(charter, reason, summary)
        append_locked(fd, ledger_path,
                      preflight.anchor_path(run_dir, charter["run_id"]),
                      charter["run_id"], {"event": "notify", **delivery, **now()})

    if not os.path.exists(sentinel(run_dir)):
        with open(sentinel(run_dir), "w") as fh:
            fh.write(str(reason) + "\n")

    print(f"STOP: {reason}")
    if not delivery["delivered"]:
        # Stopped but unresolved: nobody has been told. Exit nonzero so this is
        # distinguishable from a clean finish in the driver's own status.
        print(f"UNRESOLVED: notification failed: {delivery.get('error')}")
        raise SystemExit(3)


def _refuse_source_binding_locked(run_dir, ledger_path, charter, reason, fd):
    """Record a source refusal without invoking any post-verification code."""
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    refusal = f"preflight refused: {reason}"
    append_locked(fd, ledger_path, anchor, charter["run_id"],
                  {"event": "stop", "reason": refusal, "summary": {}, **now()})
    with open(sentinel(run_dir), "w") as fh:
        fh.write(refusal + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    preflight._sync_directory(sentinel(run_dir))
    print(f"STOP: {refusal}")
    raise SystemExit(2)


def finish(run_dir, ledger_path, charter, reason, summary, fd=None):
    """Complete the terminal sequence on the caller's locked ledger FD."""
    if fd is not None:
        return _finish_locked(run_dir, ledger_path, charter, reason, summary, fd)
    with ledger.locked(ledger_path,
                       preflight.anchor_path(run_dir, charter["run_id"]),
                       charter["run_id"]) as owned_fd:
        return _finish_locked(run_dir, ledger_path, charter, reason, summary, owned_fd)


def artifact_path(run_dir, index, attempt):
    """One path per attempt. Reusing a path across attempts is what let a
    discarded or stale file be picked up by a later one."""
    return os.path.join(run_dir, "results", f"trial-{index}-{attempt}.json")


def remaining_budget(charter, spent, iteration_elapsed=0.0):
    """Clip a trial to what is left of the wall-clock budget.

    Starting a full-length trial with seconds remaining lets a run overrun its
    budget by an entire trial plus the orchestrator's own allowance, which makes
    the budget a suggestion rather than a bound.
    """
    budgets = charter["budgets"]
    usable = (budgets.get("max_iteration_duration_s", float("inf"))
              - budgets.get("termination_duration_s", 0) - iteration_elapsed)
    return min(budgets["max_trial_duration_s"], usable,
               budgets["max_wallclock_s"] - spent - iteration_elapsed)


def termination_duration(charter, supervised=False):
    """Return the bounded collector cleanup margin for this entry mode."""
    value = charter["budgets"].get("termination_duration_s")
    if value is None and supervised:
        return DEFAULT_SUPERVISED_TERMINATION_DURATION_S
    return value or 0.0


def trial_deadlines(charter, spent, iteration_started, supervised=False):
    """Return absolute trial and collector deadlines.

    The trial gets the declared cleanup reserve before the collector's outer
    deadline. Both deadlines are anchored to iteration_started, so startup and
    ledger/preflight work cannot extend either the iteration or wall-clock
    budget.
    """
    budgets = charter["budgets"]
    reserve = termination_duration(charter, supervised=supervised)
    launched_at = time.monotonic()
    hard_limits = [iteration_started + budgets["max_wallclock_s"] - spent]
    if "max_iteration_duration_s" in budgets:
        hard_limits.append(iteration_started + budgets["max_iteration_duration_s"])
    hard_deadline = min(hard_limits)
    trial_deadline = min(launched_at + budgets["max_trial_duration_s"],
                         hard_deadline - reserve)
    collector_deadline = min(trial_deadline + reserve, hard_deadline)
    return trial_deadline, collector_deadline


def _kill_collector(proc, cleanup_timeout):
    """Hard-stop a collector that did not honor its own trial deadline."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=max(0.0, cleanup_timeout))
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=max(0.0, cleanup_timeout))
        except subprocess.TimeoutExpired:
            pass


def trials_from(records):
    """Observations already in the ledger, validated on the way out.

    Only fresh artifacts used to be shape-checked, so a malformed result that
    reached the ledger raised inside the evaluator on this and every later
    firing — a stop that could never stop.
    """
    out = []
    resolved = set()
    for record in records:
        if record["event"] != "trial":
            continue
        pair = (record.get("trial_index"), record.get("attempt"))
        if pair in resolved:
            raise ledger.Corrupt(f"duplicate resolution pair {pair!r}")
        resolved.add(pair)
        if "result" not in record or not isinstance(record["result"], dict):
            raise ledger.Corrupt(
                f"trial {record.get('trial_index')} has no result object")
        problem = check_result_shape(record["result"])
        if problem:
            raise ledger.Corrupt(
                f"trial {record.get('trial_index')} has a malformed result: {problem}")
        out.append(record["result"])
    return out


RESULT_SCHEMA = {"status": str, "passed": bool, "duration_s": (int, float),
                 "failures": list, "tail": list}


def check_result_shape(result):
    """Identity is not enough. A result with the right ids and no `passed` used
    to be appended and then raise inside the evaluator on this and every later
    firing, with no stop and no notification."""
    for field, want in RESULT_SCHEMA.items():
        if field not in result:
            return f"result is missing {field}"
        if not isinstance(result[field], want):
            return f"result field {field} has type {type(result[field]).__name__}"
    if result["status"] == "ok" and not isinstance(result.get("exit_code"), int):
        return "an ok result must carry an integer exit_code"
    return None


def read_artifact(path, expected):
    """Return the artifact only if it is complete and is the trial we expect.

    Trusting the filename alone adopts a stale or wrong-arm result, and an
    unparseable file used to raise through every timer firing forever without
    stopping or notifying anyone.
    """
    try:
        with open(path) as fh:
            result = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"artifact unreadable: {exc}"
    for field, want in expected.items():
        if result.get(field) != want:
            return None, (f"artifact identity mismatch on {field}: "
                          f"{result.get(field)!r} != {want!r}")
    problem = check_result_shape(result)
    if problem:
        return None, f"artifact is malformed: {problem}"
    return result, None


def pending_start(records):
    """The last iteration_start with no later resolution, or None.

    Looking only at the final record missed a start left behind a stop record,
    which a crash-mid-trial-then-stop produces: its artifact stayed unclassified
    for ever.
    """
    # Resolve by attempt, not by index. Two attempts can share an index (a
    # crash-retry), and matching on index alone let an earlier attempt's
    # infra_retry mark a newer attempt's completed artifact as already resolved.
    resolved = set()
    for record in records:
        if record["event"] not in ("trial", "discarded", "infra_retry"):
            continue
        attempt = record.get("attempt")
        if not isinstance(attempt, str) or not attempt:
            raise ledger.Corrupt(
                f"{record['event']} resolution has no non-empty attempt")
        resolved.add((record["trial_index"], attempt))
    for record in reversed(records):
        if (record["event"] == "iteration_start"
                and (record["trial_index"], record.get("attempt")) not in resolved):
            return record
    return None


def reconcile(records, run_dir, ledger_path, charter, summary_now, fd):
    """Resolve an iteration that started but never recorded its trial.

    Returns (records, problem, done). `done` is True when adoption terminalised
    the run — because the artifact landed outside budget, a stop is pending, or
    the stopping rule is now met — so the caller stops rather than starting more.
    """
    start = pending_start(records)
    if start is None:
        return records, None, False
    index = start["trial_index"]
    attempt = start.get("attempt")
    if not isinstance(attempt, str) or not attempt:
        return records, (f"iteration_start for trial {index} carries no attempt id; "
                         "its artifact cannot be shown to belong to it"), False
    artifact = artifact_path(run_dir, index, attempt)

    if not os.path.exists(artifact):
        append_locked(fd, ledger_path,
                      preflight.anchor_path(run_dir, charter["run_id"]),
                      charter["run_id"], {"event": "infra_retry", "trial_index": index,
                                          "attempt": attempt,
                                          "cause": "crash before trial completed", **now()})
        return ledger.read_fd(fd, ledger_path), None, False

    result, problem = read_artifact(artifact, {
        "run_id": charter["run_id"], "trial_index": index,
        "arm": start.get("arm"), "seed": start.get("seed"), "attempt": attempt})
    if problem:
        return records, f"cannot reconcile trial {index}: {problem}", False

    # Adopt through the same gate as a fresh result. Appending trial directly
    # would let a recovered artifact bypass the budget and durable stop record.
    accept(run_dir, ledger_path, charter, index, start.get("arm"), attempt,
           result, sorted(charter["arms"]), summary_now, fd=fd)
    records = ledger.read_fd(fd, ledger_path)
    return records, None, any(r["event"] == "stop" for r in records)


def main():
    if len(sys.argv) == 2:
        supervised, run_dir = False, sys.argv[1]
    elif len(sys.argv) == 3 and sys.argv[1] == "--supervised":
        supervised, run_dir = True, sys.argv[2]
    else:
        print("usage: iterate.py [--supervised] <run-dir>", file=sys.stderr)
        return 2
    try:
        iterate(run_dir, supervised=supervised)
    except ledger.IdentityCorrupt as exc:
        return halt(run_dir, f"ledger identity corruption, human required: {exc}")
    except ledger.Corrupt as exc:
        return halt(run_dir, f"ledger corrupt, human required: {exc}", trusted=True)
    except Exception as exc:
        # Fail closed. Whatever raised, the timer keeps firing, so exiting on a
        # bare traceback would loop into the same crash for ever. Leave a durable
        # sentinel and stop.
        import traceback
        return halt(run_dir, f"unexpected error, human required: "
                             f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    trusted=True)


def iterate(run_dir, supervised=False):
    iteration_started = time.monotonic()
    ledger_path = os.path.join(run_dir, "iterations.jsonl")
    try:
        charter = json.load(open(os.path.join(run_dir, "charter.json")))
    except (json.JSONDecodeError, OSError) as exc:
        return halt(run_dir, f"charter is unreadable: {exc}")
    # Shape first. Every line below assumes a mapping with the fields it names,
    # and a malformed charter would otherwise raise somewhere with nothing to
    # stop cleanly — a run that cannot even refuse.
    shape = preflight.check_schema(charter)
    if shape:
        return halt(run_dir, f"charter is unusable: {shape}")

    anchor_path = preflight.anchor_path(run_dir, charter["run_id"])
    with ledger.locked(ledger_path, anchor_path, charter["run_id"]) as fd:
        mode_refusal = preflight.check_execution_mode(charter, supervised=supervised)
        if mode_refusal:
            return halt(run_dir, f"execution mode refused: {mode_refusal}")
        # This is the first action after locked-FD identity validation.  It must
        # precede mkdir, STOP conversion, reconciliation, notification, probes,
        # and trials on every unattended entry path.
        binding_refusal, binding_record = preflight.check_runtime_binding(charter)
        if binding_refusal:
            return halt(run_dir, f"runtime identity refused: {binding_refusal}")

        source_refusal = (None if supervised else
                          preflight.check_source_binding(charter, run_dir))
        if source_refusal:
            return _refuse_source_binding_locked(run_dir, ledger_path, charter,
                                                 source_refusal, fd)

        if not supervised:
            receipt_refusal = preflight.check_containment_evidence(charter, run_dir)
            if receipt_refusal:
                return _refuse_source_binding_locked(
                    run_dir, ledger_path, charter, receipt_refusal, fd)

        # No replacement-directory mkdir or result artifact exists before the
        # descriptor/anchor identity and source contract have been checked.
        os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
        arms = sorted(charter["arms"])
        records = ledger.read_fd(fd, ledger_path)

        # STOP is only a pre-start hint. Under the ledger lock, turn it into the
        # authoritative record before any reconciliation or acceptance can occur.
        if (os.path.exists(sentinel(run_dir)) and
                not any(r["event"] == "stop" for r in records)):
            invalid = preflight.validate_charter(charter, run_dir)
            if invalid:
                append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                              {"event": "stop",
                               "reason": f"preflight refused: {invalid}",
                               "summary": {}, **now()})
                print(f"STOP: preflight refused: {invalid}")
                return
            append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                          {"event": "stop", "reason": "owner stop",
                           "summary": {}, **now()})
            records = ledger.read_fd(fd, ledger_path)

        def summary_now():
            """Never raise; a summary must not prevent a durable stop."""
            try:
                return evaluator.should_stop(
                    trials_from(ledger.read_fd(fd, ledger_path)),
                    charter["stopping_rule"], arms)[2]
            except Exception as exc:
                return {"unavailable": f"{type(exc).__name__}: {exc}"}

        # Only the durable record is authoritative. A bare STOP was converted
        # above while holding the ledger lock.
        stopping = any(r["event"] == "stop" for r in records)

        if stopping:
            invalid = preflight.validate_charter(charter, run_dir)
            if invalid:
                print(f"stopped, and the charter cannot be validated: {invalid}")
                return
            records, problem, done = reconcile(records, run_dir, ledger_path,
                                               charter, summary_now, fd)
            if problem:
                return halt(run_dir, problem, trusted=True)
            if done:
                return
            return finish(run_dir, ledger_path, charter, "owner stop", summary_now(), fd)

        # Static validation and recorded evidence must pass before reading live
        # identity.  The live probe itself is last, after that identity refusal
        # point, so an outside-unit process cannot probe first.
        refusal = preflight.run_static(charter, run_dir, supervised=supervised)
        if refusal:
            return finish(run_dir, ledger_path, charter,
                          f"preflight refused: {refusal}", summary_now(), fd)

        subject_refusal = preflight.check_subject_binding(charter, run_dir)
        if subject_refusal:
            return finish(run_dir, ledger_path, charter,
                          f"preflight refused: {subject_refusal}", summary_now(), fd)

        refusal = preflight.run_live(charter, supervised=supervised)
        if refusal:
            return finish(run_dir, ledger_path, charter,
                          f"preflight refused: {refusal}", summary_now(), fd)

        records, problem, done = reconcile(records, run_dir, ledger_path,
                                           charter, summary_now, fd)
        if problem:
            return finish(run_dir, ledger_path, charter, problem, summary_now(), fd)
        if done:
            return

        if not any(r["event"] == "run_start" for r in records):
            append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                          {"event": "run_start", "charter": charter,
                           "runtime_binding": binding_record, **now()})
            records = ledger.read_fd(fd, ledger_path)

        spent, anomaly = clock.elapsed(records)
        if anomaly:
            return finish(run_dir, ledger_path, charter,
                          f"unreliable clock: {anomaly}", summary_now(), fd)
        if spent >= charter["budgets"]["max_wallclock_s"]:
            return finish(run_dir, ledger_path, charter,
                          "wallclock budget exhausted", summary_now(), fd)

        retries = sum(1 for r in records if r["event"] == "infra_retry")
        if retries >= charter["budgets"]["max_infra_retries"]:
            return finish(run_dir, ledger_path, charter,
                          "infra retries exhausted", summary_now(), fd)

        # Recomputed from the ledger every time rather than read from a stored
        # flag, so a crash between the deciding trial and the stop record cannot
        # let the run take an extra observation past its stopping boundary.
        done, reason, summary = evaluator.should_stop(
            trials_from(records), charter["stopping_rule"], arms)
        if done:
            return finish(run_dir, ledger_path, charter, reason, summary, fd)

        trials = trials_from(records)
        index = len(trials)
        arm = evaluator.arm_for(index, arms)
        arm_position = sum(1 for t in trials if t.get("arm") == arm)
        seed = evaluator.seed_for(charter["run_id"], arm, arm_position)
        # Run id, index, arm and seed are all computable in advance, so matching
        # them proves nothing about which attempt produced a file. The attempt id
        # is unpredictable and issued here, which makes an artifact evidence
        # about this attempt rather than about the identity it claims.
        attempt = secrets.token_hex(8)

        append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                      {"event": "iteration_start", "trial_index": index,
                       "arm": arm, "seed": seed, "attempt": attempt,
                       **now()})

        spec = dict(charter["arms"][arm], arm=arm, run_id=charter["run_id"],
                    trial_index=index, attempt=attempt)
        artifact = artifact_path(run_dir, index, attempt)
        expected = {"run_id": charter["run_id"], "trial_index": index,
                    "arm": arm, "seed": seed, "attempt": attempt}
        trial_deadline, collector_deadline = trial_deadlines(
            charter, spent, iteration_started, supervised=supervised)
        trial_timeout = trial_deadline - time.monotonic()
        if trial_timeout <= 0 or collector_deadline <= time.monotonic():
            return finish(run_dir, ledger_path, charter,
                          "wallclock budget exhausted", summary_now(), fd)
        collector_argv = [sys.executable, os.path.join(HERE, "collector.py"),
                          json.dumps(spec), str(seed), str(trial_timeout), artifact,
                          str(trial_deadline)]
        collector = subprocess.Popen(
            collector_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        try:
            collector.communicate(
                timeout=max(0.0, collector_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_collector(collector, termination_duration(
                charter, supervised=supervised))
            # The collector may have published before it died. Charging a retry
            # without looking drops a completed observation. A trial deadline is
            # terminal, rather than an infrastructure retry: retrying a hung
            # command would permit an unbounded series of deadline overruns.
            salvaged, _ = read_artifact(artifact, expected)
            if salvaged:
                if salvaged["status"] == "timeout":
                    return finish(run_dir, ledger_path, charter,
                                  "trial deadline exceeded", summary_now(), fd)
                # Through the same gate as everything else. accept() handles the
                # stop and over-budget cases, so salvage cannot bypass it.
                return accept(run_dir, ledger_path, charter, index, arm, attempt,
                              salvaged, arms, summary_now, fd=fd)
            return finish(run_dir, ledger_path, charter,
                          "trial deadline exceeded", summary_now(), fd)
        if collector.returncode != 0:
            append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                          {"event": "infra_retry", "trial_index": index,
                           "attempt": attempt,
                           "cause": "collector failed: nonzero exit",
                           **now()})
            print(f"trial {index} [{arm}]: collector failed, charged to retries")
            return

        result, problem = read_artifact(artifact, expected)
        if problem:
            return finish(run_dir, ledger_path, charter,
                          f"collector produced an unusable artifact: {problem}",
                          summary_now(), fd)
        if result["status"] == "timeout":
            return finish(run_dir, ledger_path, charter,
                          "trial deadline exceeded", summary_now(), fd)

        accept(run_dir, ledger_path, charter, index, arm, attempt, result, arms,
               summary_now, fd=fd)


def _accept_locked(run_dir, ledger_path, charter, index, arm, attempt, result,
                   arms, summary_now, fd):
    """Decide what becomes of a completed result: trial, discarded, or terminal.

    The one place a result is turned into a ledger record, so the normal path
    and the salvage path cannot diverge. The durable owner-stop record is the
    only stop gate here. The elapsed-time read immediately before the trial
    append is the budget decision; it is not rechecked after that decision.
    """
    anchor_path = preflight.anchor_path(run_dir, charter["run_id"])
    subject_refusal = preflight.check_subject_binding(charter, run_dir)
    if subject_refusal:
        return finish(run_dir, ledger_path, charter,
                      f"preflight refused: {subject_refusal}", summary_now(), fd)
    records = ledger.read_fd(fd, ledger_path)
    stop_pending = any(r["event"] == "stop" for r in records)
    if stop_pending:
        append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                      {"event": "discarded", "trial_index": index,
                       "arm": arm, "attempt": attempt, "result": result,
                       "cause": "owner stop arrived during the trial", **now()})
        print(f"stop requested during trial {index}; observation recorded as discarded")
        return finish(run_dir, ledger_path, charter, "owner stop", summary_now(), fd)

    trial_record = {"event": "trial", "trial_index": index,
                    "attempt": attempt, "result": result, **now()}
    spent_now, anomaly_now = clock.elapsed(records)
    if anomaly_now:
        return finish(run_dir, ledger_path, charter,
                      f"unreliable clock: {anomaly_now}", summary_now(), fd)
    if spent_now >= charter["budgets"]["max_wallclock_s"]:
        append_locked(fd, ledger_path, anchor_path, charter["run_id"],
                      {"event": "discarded", "trial_index": index,
                       "arm": arm, "attempt": attempt, "result": result,
                       "cause": "completed after the wall-clock budget", **now()})
        return finish(run_dir, ledger_path, charter,
                      "wallclock budget exhausted", summary_now(), fd)

    # The decision read above is deliberately the last budget operation before
    # this sole trial append. A deadline crossing during write is not a reason to
    # revoke a decision already made.
    append_locked(fd, ledger_path, anchor_path, charter["run_id"], trial_record)

    records = ledger.read_fd(fd, ledger_path)
    done, reason, summary = evaluator.should_stop(
        trials_from(records), charter["stopping_rule"], arms)
    if not evaluator.is_observation(result):
        verdict = f"discarded ({result['status']})"
    elif result["passed"]:
        verdict = "pass"
    else:
        verdict = f"FAIL {result['failures'] or result['tail']}"
    print(f"trial {index} [{arm}]: {verdict} ({result['duration_s']}s) "
          f"n={summary['overall']['n']}")
    if done:
        finish(run_dir, ledger_path, charter, reason, summary, fd)


def accept(run_dir, ledger_path, charter, index, arm, attempt, result, arms,
           summary_now, fd=None):
    """Accept only while the caller's ledger FD remains exclusively locked."""
    if fd is not None:
        return _accept_locked(run_dir, ledger_path, charter, index, arm, attempt,
                              result, arms, summary_now, fd)
    with ledger.locked(ledger_path,
                       preflight.anchor_path(run_dir, charter["run_id"]),
                       charter["run_id"]) as owned_fd:
        return _accept_locked(run_dir, ledger_path, charter, index, arm, attempt,
                              result, arms, summary_now, owned_fd)


if __name__ == "__main__":
    sys.exit(main() or 0)
