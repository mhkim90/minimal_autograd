"""Regression tests for the defects an independent review found.

Each test names the failure it reproduces. They exist because prose review
passed all of these and one real run did not: a claim about this loop is worth
what its test is worth.

Run: python3 loop/regression_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clock
import evaluator
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
        "run_id": "regress", "class": "evidence", "subject": "toy",
        "execution_mode": "supervised",
        "arms": {"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"], "cwd": tmp}},
        "notify": {"command": ["true"]},
        "containment": {"required": False}, "supervised": True,
        "stopping_rule": {"min_trials_per_arm": 2, "max_trials": 4, "ci_width": 1.0},
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


def test_partial_artifact_does_not_livelock(tmp):
    """A truncated artifact used to raise JSONDecodeError on every timer firing,
    with no stop, no notification, and an exit status that looked like success."""
    run_dir, charter = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "cafebabecafebabe"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": 1, "attempt": attempt, **testkit.stamp()})
    # The attempt-qualified path the recovery actually looks at. Planting at the
    # old shared path meant this file was never read and the check passed on an
    # unrelated stop.
    open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"),
         "w").write('{"passed": tr')

    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "artifact")
    check("partial artifact stops the run instead of looping forever", ok, detail)
    check("partial artifact does not leave a traceback",
          "Traceback" not in result.stderr, result.stderr[-200:])


def test_foreign_artifact_rejected(tmp):
    """An artifact from another arm or trial must not be adopted by filename."""
    run_dir, charter = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "0badc0de0badc0de"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": 1, "attempt": attempt, **testkit.stamp()})
    json.dump({"run_id": "regress", "trial_index": 0, "arm": "OTHER", "seed": 1,
               "attempt": attempt, "passed": True, "status": "ok", "exit_code": 0,
               "duration_s": 77.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"), "w"))

    fire(run_dir)
    trials = [r["result"] for r in ledger.read(lp) if r["event"] == "trial"]
    # Assert on the planted artifact's fingerprint. Asserting that no trial
    # exists at all was satisfied by the loop simply running a fresh one.
    check("artifact from the wrong arm is not adopted",
          not any(t.get("duration_s") == 77.0 for t in trials),
          "the foreign-arm artifact was accepted as an observation")


def test_stop_sequence_is_idempotent(tmp):
    """A crash after the stop record used to leave a run that never notified and
    never raised its sentinel; a crash after notifying could notify twice."""
    run_dir, charter = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(lp, {"event": "stop", "reason": "test", "summary": {},
                       **testkit.stamp()})

    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "test")
    check("stop record without a sentinel completes the sequence",
           ok and os.path.exists(os.path.join(run_dir, "STOP")), detail)
    notifies = [r for r in ledger.read(lp) if r["event"] == "notify"]
    check("interrupted stop notifies exactly once", len(notifies) == 1,
          f"{len(notifies)} notify records")

    fire(run_dir)
    notifies = [r for r in ledger.read(lp) if r["event"] == "notify"]
    check("repeating a completed stop does not notify again", len(notifies) == 1,
          f"{len(notifies)} notify records")
    # Note the limit of this pair: it counts ledger records, not deliveries. A
    # crash between notifier.notify() returning and its record being appended
    # still delivers twice. The guarantee is at-least-once, not exactly-once,
    # and closing that needs an idempotency key the channel honours.


def test_preflight_refuses_incomplete_charter(tmp):
    """Two real charters missing containment, scope and approval fields ran to
    completion, which is how we learned the entry conditions were decorative."""
    run_dir, _ = make_run(tmp)
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    del charter["containment"]
    json.dump(charter, open(os.path.join(run_dir, "charter.json"), "w"))
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "missing required fields")
    check("charter missing a required field is refused", ok, detail)


def test_preflight_detects_an_edit_that_keeps_the_schema_valid(tmp):
    """Deleting a field trips the schema check before the digest is consulted,
    so it proves nothing about approval binding. Change a value instead."""
    run_dir, _ = make_run(tmp)
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    charter["stopping_rule"]["max_trials"] = 9999      # schema still valid
    json.dump(charter, open(os.path.join(run_dir, "charter.json"), "w"))
    # digest deliberately not refreshed: this is an edit after approval
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "changed after approval")
    check("a schema-valid edit after approval is still refused", ok, detail)


def test_preflight_refuses_unbuilt_class(tmp):
    run_dir, _ = make_run(tmp, **{"class": "ratchet"})
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "not implemented")
    check("unbuilt ratchet class is refused at runtime", ok, detail)


def test_seed_streams_are_independent_per_arm():
    """seed=i with arm=i mod k gave arm A every even seed and arm B every odd
    one, manufacturing an arm difference with no adversary.

    Disjointness was already true of the broken scheme, so asserting it proves
    nothing. What the old scheme lacked was each arm covering the whole space.
    """
    a = [evaluator.seed_for("r", "a", k) for k in range(200)]
    b = [evaluator.seed_for("r", "b", k) for k in range(200)]
    for residue in (2, 3, 5):
        check(f"each arm's seeds span every residue class mod {residue}",
              len({s % residue for s in a}) == residue
              and len({s % residue for s in b}) == residue,
              "an arm is confined to a subset of the seed space")
    check("seeds are deterministic",
          a == [evaluator.seed_for("r", "a", k) for k in range(200)])


def test_harness_errors_are_not_subject_failures():
    trials = [{"status": "ok", "passed": True}, {"status": "timeout", "passed": False},
              {"status": "error", "passed": False}]
    s = evaluator.summarize(trials)
    check("timeouts and collector errors are discarded, not counted as failures",
          s["n"] == 1 and s["failures"] == 0 and s["discarded"] == 2, str(s))


def test_width_threshold_is_not_rounded():
    """Rounding before comparing let a true width just above the threshold stop
    the run. Exercise should_stop, not just the summary it consumes."""
    trials = [{"status": "ok", "passed": True, "arm": "a"}] * 40
    s = evaluator.summarize(trials)
    threshold = round(s["_width_exact"], 4)
    if threshold >= s["_width_exact"]:
        threshold = s["_width_exact"] - 1e-9      # just below the true width
    done, _, _ = evaluator.should_stop(
        trials, {"min_trials_per_arm": 1, "max_trials": 999, "ci_width": threshold},
        ["a"])
    check("a width fractionally above the threshold does not stop the run",
          not done, "rounded comparison accepted a width above ci_width")


def test_unexpected_arm_stops_the_run():
    """One stray arm also fails a cardinality check, so it does not discriminate.
    Keep the count right and swap a declared arm for an undeclared one."""
    trials = ([{"status": "ok", "passed": True, "arm": "a"}] * 5
              + [{"status": "ok", "passed": True, "arm": "ghost"}] * 5)
    done, reason, _ = evaluator.should_stop(
        trials, {"min_trials_per_arm": 1, "max_trials": 99, "ci_width": 1.0}, ["a", "b"])
    check("an undeclared arm stops the run even when the arm count matches",
          done and "unexpected arm" in str(reason), str(reason))


def test_clock_detects_backward_jump():
    boot = "b1"
    good = [{"at": "2026-08-29T00:00:00+0900", "mono": 0.0, "boot": boot},
            {"at": "2026-08-29T00:10:00+0900", "mono": 600.0, "boot": boot}]
    _, anomaly = clock.elapsed(good)
    check("consistent clocks report no anomaly", anomaly is None, str(anomaly))

    tampered = [{"at": "2026-08-29T00:00:00+0900", "mono": 0.0, "boot": boot},
                {"at": "2026-08-29T00:00:10+0900", "mono": 600.0, "boot": boot}]
    _, anomaly = clock.elapsed(tampered)
    check("wall clock disagreeing with monotonic time is detected",
          anomaly is not None)

    rolled = [{"at": "2026-08-29T01:00:00+0900", "mono": 0.0, "boot": boot},
              {"at": "2026-08-29T00:00:00+0900", "mono": 600.0, "boot": boot}]
    _, anomaly = clock.elapsed(rolled)
    check("wall clock moved backwards inside one boot is detected",
          anomaly is not None)

    tz = [{"at": "2026-08-29T00:00:00+0900", "mono": 0.0, "boot": boot},
          {"at": "2026-08-28T15:10:00+0000", "mono": 600.0, "boot": boot}]
    _, anomaly = clock.elapsed(tz)
    check("timezone offsets are honoured rather than stripped", anomaly is None,
          str(anomaly))


def main():
    isolated = [test_partial_artifact_does_not_livelock, test_foreign_artifact_rejected,
                test_stop_sequence_is_idempotent, test_preflight_refuses_incomplete_charter,
                test_preflight_refuses_unbuilt_class,
                test_preflight_detects_an_edit_that_keeps_the_schema_valid]
    for test in isolated:
        tmp = tempfile.mkdtemp(prefix="goal-loop-regress-")
        try:
            test(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    for test in (test_seed_streams_are_independent_per_arm,
                 test_harness_errors_are_not_subject_failures,
                 test_width_threshold_is_not_rounded,
                 test_unexpected_arm_stops_the_run,
                 test_clock_detects_backward_jump):
        test()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
