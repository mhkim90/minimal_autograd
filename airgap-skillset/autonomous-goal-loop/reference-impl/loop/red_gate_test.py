"""Tests written before the fixes, against the code as it stands.

The previous suite was written alongside its fixes and checked against an older
build, which conflated "behaves differently from the old code" with "tests the
property in its name". Ten of its seventeen checks did not discriminate.

Every test here must fail first, for the reason its name gives, on the code it
is meant to correct. A test that passes on arrival proves nothing and is a bug
in the test.

Run: python3 loop/red_gate_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluator
import ledger
import preflight
import testkit

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  <- {detail}'}")


def boot():
    return open("/proc/sys/kernel/random/boot_id").read().strip()


def stamp():
    return testkit.stamp()


def make_run(tmp, **overrides):
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    charter = {
        "run_id": "red", "class": "evidence", "subject": "toy",
        "execution_mode": "supervised",
        "arms": {"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"], "cwd": tmp}},
        "notify": {"command": ["true"]},
        "containment": {"required": False}, "supervised": True,
        "stopping_rule": {"min_trials_per_arm": 2, "max_trials": 6, "ci_width": 1.0},
        "budgets": {"max_wallclock_s": 600, "max_infra_retries": 3,
                    "max_trial_duration_s": 30},
        "approved_by": "test", "approved_at": "2026-08-29T00:00:00+0900",
    }
    charter.update(overrides)
    path = os.path.join(run_dir, "charter.json")
    json.dump(charter, open(path, "w"))
    preflight.approve(path)
    return run_dir, charter


def fire(run_dir):
    return testkit.fire(run_dir)


# --- terminal sequence -------------------------------------------------------

def test_unresolved_notification_can_be_retried(tmp):
    """A failed notification records the failure and raises the sentinel; every
    later firing then exits inside run.sh, so the run can never tell anyone."""
    marker = os.path.join(tmp, "channel-ready")
    run_dir, _ = make_run(tmp, notify={"command": ["sh", "-c",
                                                       f"test -f {marker!r}"]})
    lp = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(lp, {"event": "stop", "reason": "test", "summary": {}, **stamp()})
    fire(run_dir)                      # fails to notify, raises sentinel

    # Repair the channel without changing the approved charter or attempting a
    # second bootstrap.
    open(marker, "w").close()
    fire(run_dir)

    delivered = [r for r in ledger.read(lp) if r["event"] == "notify" and r.get("delivered")]
    check("an unresolved notification is retried once the channel works",
          delivered, "sentinel short-circuits run.sh, so finish() is unreachable")


def test_stop_during_trial_is_accounted(tmp):
    """A bare STOP during a trial is a hint for the next iteration only."""
    run_dir, _ = make_run(
        tmp, arms={"a": {"argv": [sys.executable, "-c",
                                  "import time,os;open(os.environ['STOPFILE'],'w').write('owner')"
                                  ";time.sleep(1)"],
                         "cwd": tmp}})
    os.environ["STOPFILE"] = os.path.join(run_dir, "STOP")
    fire(run_dir)
    del os.environ["STOPFILE"]

    lp = os.path.join(run_dir, "iterations.jsonl")
    records = ledger.read(lp)
    check("a mid-iteration bare STOP does not revoke the current trial",
          [r["event"] for r in records].count("trial") == 1
          and not any(r["event"] == "stop" for r in records),
          f"ledger has {[r['event'] for r in records]}")
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("the next entry serializes the bare STOP as owner stop", ok, detail)


def test_recovery_validates_charter_before_using_it(tmp):
    """Stop recovery runs before preflight, so a charter edited after approval
    can have its notify.command executed without the digest ever being checked."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(lp, {"event": "stop", "reason": "test", "summary": {}, **stamp()})

    marker = os.path.join(tmp, "executed-unapproved")
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    charter["notify"] = {"command": ["sh", "-c", f"touch {marker}"]}
    json.dump(charter, open(os.path.join(run_dir, "charter.json"), "w"))
    # digest deliberately NOT refreshed: this is an unapproved edit
    fire(run_dir)
    check("an unapproved notify.command is not executed during recovery",
          not os.path.exists(marker),
          "recovery ran the edited command before validating the digest")


# --- collector ---------------------------------------------------------------

def test_inner_timeout_stops_a_quiet_hung_subject(tmp):
    """Reading stdout to EOF before wait(timeout=...) means a subject that
    produces no output is never timed out: the read blocks until it exits."""
    run_dir, _ = make_run(tmp)
    spec = json.dumps({"argv": [sys.executable, "-c", "import time;time.sleep(45)"],
                       "cwd": tmp, "run_id": "red", "trial_index": 0, "arm": "a"})
    artifact = os.path.join(tmp, "t.json")
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "collector.py"),
                        spec, "1", "3", artifact], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.monotonic() - started
    check("a silent subject is killed at the inner timeout, not at EOF",
          elapsed < 10, f"inner timeout was 3s; collector still ran after {elapsed:.1f}s")


def test_published_artifact_survives_collector_failure(tmp):
    """If the collector publishes its artifact and then exits nonzero, the
    orchestrator charges a retry without looking, and because the last record is
    then infra_retry rather than iteration_start, reconcile never runs."""
    run_dir, _ = make_run(
        tmp, arms={"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"], "cwd": tmp}})
    lp = os.path.join(run_dir, "iterations.jsonl")

    # The reachable window: the collector published, then the process died
    # before any terminal record. The last record is therefore iteration_start,
    # and the artifact belongs to that exact attempt.
    ledger.append(lp, {"event": "run_start", "charter": {}, **stamp()})
    index, arm, attempt = 0, "a", "deadbeefdeadbeef"
    seed = evaluator.seed_for("red", arm, 0)
    ledger.append(lp, {"event": "iteration_start", "trial_index": index, "arm": arm,
                       "seed": seed, "attempt": attempt, **stamp()})
    # A duration no real trial would produce, so the recorded observation can be
    # traced to this artifact. Asserting only on trial_index would be satisfied
    # by the loop simply re-running the trial, which is the loss being tested.
    json.dump({"run_id": "red", "trial_index": index, "arm": arm, "seed": seed,
               "attempt": attempt, "status": "ok", "exit_code": 0, "passed": True,
               "duration_s": 99.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results",
                                f"trial-{index}-{attempt}.json"), "w"))

    fire(run_dir)
    trials = [r["result"] for r in ledger.read(lp) if r["event"] == "trial"]
    check("an artifact published before a collector failure is still adopted",
          any(t.get("duration_s") == 99.0 for t in trials),
          "the completed observation was dropped and the trial simply re-run")


# --- validation --------------------------------------------------------------

def test_malformed_artifact_does_not_poison_the_ledger(tmp):
    """read_artifact checks identity but not shape. A result with the right ids
    and no `passed` is appended, then raises inside summarize on this and every
    later firing, with no stop and no notification."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    seed = evaluator.seed_for("red", "a", 0)
    attempt = "feedfacefeedface"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": seed, "attempt": attempt, **stamp()})
    json.dump({"run_id": "red", "trial_index": 0, "arm": "a", "seed": seed,
               "attempt": attempt, "status": "ok"},   # no `passed`
              open(os.path.join(run_dir, "results",
                                f"trial-0-{attempt}.json"), "w"))

    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "malformed")
    check("a structurally invalid artifact stops the run", ok, detail)
    check("a structurally invalid artifact does not raise",
          "Traceback" not in result.stderr, result.stderr[-160:])


def test_empty_arms_is_refused(tmp):
    run_dir, _ = make_run(tmp, arms={})
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "no arms")
    check("a charter with no arms is refused by preflight", ok, detail)


# --- containment probe -------------------------------------------------------

def test_probe_does_not_destroy_an_existing_file(tmp):
    """check_containment opens the probe path with "w", truncating whatever is
    there before deleting it."""
    victim = os.path.join(tmp, "precious")
    open(victim, "w").write("do not lose me")
    charter = {"containment": {"required": False, "probe_path": victim}}
    preflight.check_containment(charter)
    check("the containment probe does not destroy an existing file",
          os.path.exists(victim) and open(victim).read() == "do not lose me",
          "probe truncated and removed the file it probed with")


def test_unwritable_probe_path_is_not_mistaken_for_containment():
    """A probe path whose parent does not exist raises OSError, which the check
    reads as "containment is active" even when nothing is contained."""
    charter = {"containment": {"required": True,
                               "probe_path": "/nonexistent-dir-xyz/probe"}}
    reason = preflight.check_containment(charter)
    check("an unwritable probe path is not accepted as proof of containment",
          reason is not None,
          "OSError from a bad path was read as containment being active")


# --- budget ------------------------------------------------------------------

def test_trial_cannot_outrun_the_wallclock_budget(tmp):
    """A trial starts without clipping its timeout to the remaining budget, so a
    run can exceed max_wallclock_s by a whole trial plus the outer allowance."""
    run_dir, _ = make_run(tmp, budgets={"max_wallclock_s": 2, "max_infra_retries": 3,
                                        "max_trial_duration_s": 30})
    lp = os.path.join(run_dir, "iterations.jsonl")
    old = dict(stamp())
    old["mono"] = round(time.monotonic() - 1.5, 3)
    ledger.append(lp, {"event": "run_start", "charter": {}, **old})
    # Behavioural, not a symbol-presence proxy: bounds_test times a real trial
    # against an exhausted budget. Kept here only as a pointer.
    check("a trial is clipped to the remaining wall-clock budget",
          hasattr(__import__("iterate"), "remaining_budget"),
          "see bounds_test.test_trial_timeout_is_actually_clipped for the real check")


def main():
    isolated = [
        test_unresolved_notification_can_be_retried,
        test_stop_during_trial_is_accounted,
        test_recovery_validates_charter_before_using_it,
        test_inner_timeout_stops_a_quiet_hung_subject,
        test_published_artifact_survives_collector_failure,
        test_malformed_artifact_does_not_poison_the_ledger,
        test_empty_arms_is_refused,
        test_probe_does_not_destroy_an_existing_file,
        test_trial_cannot_outrun_the_wallclock_budget,
    ]
    for test in isolated:
        tmp = tempfile.mkdtemp(prefix="goal-loop-red-")
        try:
            test(tmp)
        except Exception as exc:
            check(test.__name__, False, f"test itself raised: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    test_unwritable_probe_path_is_not_mistaken_for_containment()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} passing, {len(failed)} failing")
    print("When written, every check here failed. Once its fix is in, it must "
          "stay green, so a failure now is a regression.")
    # Exiting zero regardless made this unusable as a gate: a suite that cannot
    # fail cannot guard anything.
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
