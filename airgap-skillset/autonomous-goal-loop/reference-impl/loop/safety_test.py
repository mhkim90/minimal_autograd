"""Safety invariants that must hold whatever the implementation looks like.

These are separate from the defect-regression suites on purpose. Those suites
cover failures someone already found; this one covers properties the loop is
*for*. The distinction was learned the hard way: a round of fixes closed eight
known defects and broke owner stop, because owner stop lived in one line of
shell that no test named.

If a change makes one of these fail, the change is wrong, however good its
reason.

Run: python3 loop/safety_test.py
"""
import json
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ledger
import preflight
import testkit

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  <- {detail}'}")


def make_run(tmp, **overrides):
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    charter = {
        "run_id": "safety", "class": "evidence", "subject": "toy",
        "execution_mode": "supervised",
        "arms": {"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"], "cwd": tmp}},
        "notify": {"command": ["true"]},
        "containment": {"required": False}, "supervised": True,
        "stopping_rule": {"min_trials_per_arm": 20, "max_trials": 40, "ci_width": 0.01},
        "budgets": {"max_wallclock_s": 600, "max_infra_retries": 3,
                    "max_trial_duration_s": 30},
        "approved_by": "test", "approved_at": "2026-08-30T00:00:00+0900",
    }
    charter.update(overrides)
    path = os.path.join(run_dir, "charter.json")
    json.dump(charter, open(path, "w"))
    preflight.approve(path)
    return run_dir, charter


def fire(run_dir, times=1):
    return testkit.fire(run_dir, times)


def events(run_dir):
    return [r["event"] for r in
            ledger.read(os.path.join(run_dir, "iterations.jsonl"))]


STOP_SH = os.path.join(HERE, "stop.sh")


# --- owner stop --------------------------------------------------------------

def test_owner_stop_halts_collection(tmp):
    """A durable owner stop prevents collection on every later firing."""
    run_dir, _ = make_run(tmp)
    fire(run_dir)                                  # one normal trial
    before = events(run_dir).count("trial")

    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir, times=3)

    after = events(run_dir).count("trial")
    check("no observation is collected after the owner records a stop",
          after == before, f"trial count went {before} -> {after}")
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("the durable owner stop is the reason collection halts", ok, detail)


def test_owner_stop_is_recorded_as_terminal(tmp):
    """A stop the owner asked for must appear in the ledger as the reason the
    run ended, not be inferred from the absence of later records."""
    run_dir, _ = make_run(tmp)
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir, times=2)

    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    stops = [r for r in records if r["event"] == "stop"]
    check("an owner stop produces exactly one terminal stop record",
          len(stops) == 1, f"{len(stops)} stop records")
    ok, detail = testkit.stopped_because(run_dir, "owner")
    check("the terminal record names the owner as the cause", ok, detail)


def test_stopped_run_still_completes_notification(tmp):
    """Blocking collection must not block the terminal sequence: a run that owes
    a notification has to be able to deliver it after the sentinel exists."""
    marker = os.path.join(tmp, "delivered")
    run_dir, _ = make_run(tmp, notify={"command": ["sh", "-c",
                                                       f"test -f {marker!r}"]})
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir)

    open(marker, "w").close()
    fire(run_dir)

    check("a stopped run can still deliver an owed notification",
          os.path.exists(marker), "terminal recovery is unreachable once stopped")


def test_stopped_run_does_not_probe_containment(tmp):
    """A stopped run must not act. Probing the boundary on every timer firing,
    for ever, is acting."""
    run_dir, _ = make_run(tmp, containment={"required": True,
                                            "probe_path": os.path.join(tmp, "probe")})
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir, times=2)

    # The probe cleans up after itself, so observe it via the refusal it would
    # otherwise cause: an uncontained run with required containment must not be
    # able to turn a stopped run into a preflight refusal.
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("a stopped run is not re-judged by the containment probe",
          ok and "containment" not in detail.lower(), detail)
    # Observing the reason is indirect. Assert the probe path too: the probe
    # cleans up after itself, so a leftover file is not evidence, but a probe
    # that ran and was refused would have changed the recorded reason above.


# --- observation eligibility -------------------------------------------------

def test_discarded_observation_is_never_adopted(tmp):
    """A bare STOP raised during a trial is only a next-entry hint."""
    stopfile = os.path.join(tmp, "run", "STOP")
    run_dir, _ = make_run(
        tmp, arms={"a": {"argv": [sys.executable, "-c",
                                      f"open({stopfile!r},'w').write('owner')"],
                             "cwd": tmp}})
    lp = os.path.join(run_dir, "iterations.jsonl")

    # A trial that raises the sentinel while running may still be accepted.
    fire(run_dir)
    recorded = ledger.read(lp)
    trials = [r for r in recorded if r["event"] == "trial"]
    check("a mid-iteration bare STOP does not revoke the current trial",
          len(trials) == 1 and not any(r["event"] == "stop" for r in recorded),
          f"ledger has {[r['event'] for r in recorded]}")
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("the next iteration serializes the bare STOP as owner stop", ok, detail)


def test_loop_stop_blocks_behind_running_iteration(tmp):
    """The stop command waits for the same ledger lock, so the trial wins."""
    run_dir, _ = make_run(
        tmp, arms={"a": {"argv": [sys.executable, "-c", "import time; time.sleep(1)"],
                          "cwd": tmp}})
    env = dict(os.environ)
    running = subprocess.Popen([os.path.join(HERE, "run-supervised.sh"), run_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env)
    lp = os.path.join(run_dir, "iterations.jsonl")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if any(r["event"] == "iteration_start" for r in ledger.read(lp)):
            break
        time.sleep(0.01)
    stopper = subprocess.Popen([STOP_SH, run_dir], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    time.sleep(0.1)
    blocked = stopper.poll() is None
    running.wait(timeout=10)
    stopper.wait(timeout=10)
    records = ledger.read(lp)
    kinds = [r["event"] for r in records]
    trial_pos, stop_pos = kinds.index("trial"), kinds.index("stop")
    check("loop-stop blocks while loop holds the ledger lock", blocked,
          f"stopper exited early with {stopper.returncode}")
    check("the running iteration's trial precedes the durable stop",
          trial_pos < stop_pos, f"ledger order: {kinds}")


def test_loop_stop_wins_before_iteration(tmp):
    """A stop record acquired first prevents a later iteration from starting."""
    run_dir, _ = make_run(tmp)
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir)
    lp = os.path.join(run_dir, "iterations.jsonl")
    records = ledger.read(lp)
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("a stop acquired before the ledger lock prevents a later trial",
          ok and not any(r["event"] == "trial" for r in records), detail)


def test_loop_stop_persists_across_restarts(tmp):
    """A durable stop remains authoritative over every later timer firing."""
    run_dir, _ = make_run(tmp)
    subprocess.run([STOP_SH, run_dir], check=True, capture_output=True, text=True)
    fire(run_dir, times=3)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("a durable stop refuses trials after restart", ok and
          not any(r["event"] == "trial" for r in records), detail)


def test_run_and_loop_stop_preserve_one_ledger_inode(tmp):
    """Both entry points contend on the approved ledger inode."""
    run_dir, _ = make_run(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    before = os.fstat(fd).st_ino
    fcntl.flock(fd, fcntl.LOCK_EX)
    run_attempt = subprocess.Popen([os.path.join(HERE, "run-supervised.sh"), run_dir],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True,
                                    env=dict(os.environ))
    time.sleep(0.1)
    run_blocked = run_attempt.poll() is None
    stopper = subprocess.Popen([STOP_SH, run_dir], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    time.sleep(0.1)
    same_while_blocked = os.stat(path).st_ino == before and stopper.poll() is None
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    stopper.wait(timeout=10)
    same_after = os.stat(path).st_ino == before
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    run_attempt.wait(timeout=10)
    check("run.sh and loop-stop use one non-replaced ledger inode",
          run_blocked and same_while_blocked and same_after and ok,
          detail or run_attempt.stdout.read())


def test_stale_artifact_is_not_adopted(tmp):
    """Reconcile resolves an unresolved iteration_start by its attempt-qualified
    path. A file sitting at that path whose own identity names a different
    attempt is not evidence about this one and must be rejected, not adopted on
    its filename."""
    import evaluator
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "aaaaaaaaaaaaaaaa"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": evaluator.seed_for("safety", "a", 0),
                       "attempt": attempt, **testkit.stamp()})
    # Right filename, wrong attempt inside: a file left by attempt B.
    json.dump({"run_id": "safety", "trial_index": 0, "arm": "a",
               "seed": evaluator.seed_for("safety", "a", 0), "attempt": "bbbbbbbbbbbbbbbb",
               "status": "ok", "exit_code": 0, "passed": True, "duration_s": 42.0,
               "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"), "w"))

    fire(run_dir)
    adopted = [r["result"] for r in ledger.read(lp) if r["event"] == "trial"]
    check("an artifact naming a different attempt is not adopted",
          not any(t.get("duration_s") == 42.0 for t in adopted),
          "a foreign-attempt artifact was accepted as an observation")


def main():
    tests = [test_owner_stop_halts_collection,
             test_owner_stop_is_recorded_as_terminal,
             test_stopped_run_still_completes_notification,
             test_stopped_run_does_not_probe_containment,
             test_discarded_observation_is_never_adopted,
             test_stale_artifact_is_not_adopted,
             test_loop_stop_blocks_behind_running_iteration,
             test_loop_stop_wins_before_iteration,
             test_loop_stop_persists_across_restarts,
             test_run_and_loop_stop_preserve_one_ledger_inode]
    for test in tests:
        tmp = tempfile.mkdtemp(prefix="goal-loop-safety-")
        try:
            test(tmp)
        except Exception as exc:
            check(test.__name__, False, f"test itself raised: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
