"""Red-first witnesses for the ledger-FD lock and anchor contract."""
import json
import contextlib
import io
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

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  <- {detail}'}")


def make_charter(tmp, run_id="fd-lock", **overrides):
    run_dir = os.path.join(tmp, "run")
    os.makedirs(run_dir)
    charter = {
        "run_id": run_id, "class": "evidence", "subject": "toy",
        "execution_mode": "supervised",
        "arms": {"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"],
                         "cwd": tmp}},
        "notify": {"command": ["true"]},
        "containment": {"required": False}, "supervised": True,
        "stopping_rule": {"min_trials_per_arm": 20, "max_trials": 40,
                           "ci_width": 0.01},
        "budgets": {"max_wallclock_s": 600, "max_infra_retries": 3,
                    "max_trial_duration_s": 30},
        "approved_by": "test", "approved_at": "2026-08-31T00:00:00+0000",
    }
    charter.update(overrides)
    path = os.path.join(run_dir, "charter.json")
    json.dump(charter, open(path, "w"))
    return run_dir, charter, path


def approve(tmp, **overrides):
    run_dir, charter, path = make_charter(tmp, **overrides)
    preflight.approve(path)
    return run_dir, charter


def test_approve_bootstraps_ledger_and_external_anchor(tmp):
    run_dir, charter, path = make_charter(tmp)
    preflight.approve(path)
    ledger_path = os.path.join(run_dir, "iterations.jsonl")
    anchor_path = preflight.anchor_path(run_dir, charter["run_id"])
    anchor = json.load(open(anchor_path))
    stat = os.stat(ledger_path)
    check("approve creates the ledger", os.path.isfile(ledger_path), ledger_path)
    check("approve creates an external anchor", os.path.dirname(anchor_path) == tmp
          and os.path.isfile(anchor_path), anchor_path)
    check("anchor binds the approved run and ledger inode",
          anchor == {"run_id": charter["run_id"],
                     "ledger": {"st_dev": stat.st_dev, "st_ino": stat.st_ino}},
           repr(anchor))


def test_path_like_run_id_cannot_escape_anchor_parent(tmp):
    run_dir, charter, _ = make_charter(tmp, run_id="../escape/absolute-like")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    check("path-like run_id stays in the intended anchor parent",
          os.path.dirname(os.path.abspath(anchor)) == os.path.abspath(tmp)
          and os.path.basename(anchor).endswith(".anchor"), anchor)


def test_first_identity_validation_happens_after_flock(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    original_validate = ledger.validate_identity
    original_flock = ledger.fcntl.flock
    events = []

    def checked_validate(fd, ledger_path, anchor_path, run_id=None):
        events.append("identity")
        return original_validate(fd, ledger_path, anchor_path, run_id)

    def checked_flock(fd, operation):
        if operation == ledger.fcntl.LOCK_EX:
            events.append("flock")
        return original_flock(fd, operation)

    ledger.validate_identity = checked_validate
    ledger.fcntl.flock = checked_flock
    try:
        with ledger.locked(path, anchor, charter["run_id"]):
            pass
    finally:
        ledger.validate_identity = original_validate
        ledger.fcntl.flock = original_flock
    check("first identity validation follows exclusive flock",
          events[:2] == ["flock", "identity"], repr(events))


def test_replacement_between_open_and_flock_is_detected_after_lock(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    original_flock = ledger.fcntl.flock
    replaced = [False]

    def replacing_flock(fd, operation):
        if operation == ledger.fcntl.LOCK_EX and not replaced[0]:
            replaced[0] = True
            os.rename(path, path + ".stale")
            open(path, "w").close()
        return original_flock(fd, operation)

    ledger.fcntl.flock = replacing_flock
    try:
        try:
            with ledger.locked(path, anchor, charter["run_id"]):
                pass
        except ledger.IdentityCorrupt:
            classified = True
        else:
            classified = False
    finally:
        ledger.fcntl.flock = original_flock
    check("replacement between open and flock is caught post-lock", classified)


def test_exceptional_body_replacement_is_classified_and_released(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    original_flock = ledger.fcntl.flock
    original_close = ledger.os.close
    events = []

    def checked_flock(fd, operation):
        events.append(("flock", operation))
        return original_flock(fd, operation)

    def checked_close(fd):
        events.append(("close", fd))
        return original_close(fd)

    ledger.fcntl.flock = checked_flock
    ledger.os.close = checked_close
    raised = None
    try:
        try:
            with ledger.locked(path, anchor, charter["run_id"]):
                os.rename(path, path + ".stale")
                open(path, "w").close()
                raise RuntimeError("body failure after replacement")
        except BaseException as exc:
            raised = exc
            classified = isinstance(exc, ledger.IdentityCorrupt)
    finally:
        ledger.fcntl.flock = original_flock
        ledger.os.close = original_close
    check("replacement during exceptional body wins over body exception", classified,
          f"raised {type(raised).__name__}: {raised}")
    check("exceptional cleanup unlocks and closes the ledger FD",
          any(kind == "flock" and op == ledger.fcntl.LOCK_UN
              for kind, op in events)
          and any(kind == "close" for kind, _ in events), repr(events))


def test_finish_error_replacement_is_classified_by_context_cleanup(tmp):
    import iterate
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    original_notify = iterate.notifier.notify

    def failing_notify(*args):
        os.rename(path, path + ".stale")
        open(path, "w").close()
        raise RuntimeError("finish notification failure after replacement")

    iterate.notifier.notify = failing_notify
    raised = None
    try:
        try:
            with ledger.locked(path, anchor, charter["run_id"]) as fd:
                iterate.finish(run_dir, path, charter, "test", {}, fd=fd)
        except BaseException as exc:
            raised = exc
            classified = isinstance(exc, ledger.IdentityCorrupt)
    finally:
        iterate.notifier.notify = original_notify
    check("finish/error replacement cannot evade context cleanup", classified,
          f"raised {type(raised).__name__}: {raised}")


def test_continuation_without_anchor_fails_without_side_effects(tmp):
    run_dir, _, path = make_charter(tmp)
    # A stale/partial ledger without its anchor is not an initializer either.
    open(os.path.join(run_dir, "iterations.jsonl"), "w").close()
    result = testkit.fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "anchor", result.stderr,
                                         returncode=result.returncode)
    created = [name for name in ("lease", "STOP", "results")
               if os.path.exists(os.path.join(run_dir, name))]
    check("virgin continuation halts for missing anchor", ok, detail or result.stderr)
    check("virgin continuation creates no run side effects", not created,
          f"created {created}; charter was {path}")


def test_lease_file_is_not_used(tmp):
    run_dir, _ = approve(tmp)
    result = testkit.fire(run_dir)
    check("continuation does not create the removed lease", 
          not os.path.exists(os.path.join(run_dir, "lease")), result.stdout)


def test_stop_before_first_iteration_is_durable(tmp):
    run_dir, _ = approve(tmp)
    stop = subprocess.run([os.path.join(HERE, "stop.sh"), run_dir],
                          capture_output=True, text=True)
    later = testkit.fire(run_dir)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    ok, detail = testkit.stopped_because(run_dir, "owner stop",
                                         later.stderr + later.stdout)
    check("approved stop before launch is durable", stop.returncode == 0 and ok,
          detail or stop.stderr)
    check("durable pre-launch stop prevents a trial",
          not any(r["event"] == "trial" for r in records), repr(records))


def _append_duplicate(run_dir, first, second):
    path = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(path, first)
    ledger.append(path, second)
    return testkit.fire(run_dir)


def test_duplicate_iteration_start_is_corruption(tmp):
    run_dir, charter = approve(tmp)
    common = {"trial_index": 0, "arm": "a", "seed": 1}
    first = {"event": "iteration_start", "attempt": "a", **common,
             **testkit.stamp()}
    second = {"event": "iteration_start", "attempt": "a", **common,
              **testkit.stamp()}
    result = _append_duplicate(run_dir, first, second)
    ok, detail = testkit.stopped_because(run_dir, "duplicate", result.stderr,
                                         returncode=result.returncode)
    check("duplicate iteration_start pair is corruption", ok,
          detail or result.stdout)


def test_duplicate_resolution_pairs_are_corruption(tmp):
    for label, event1, event2 in (
        ("trial/trial", "trial", "trial"),
        ("discarded/discarded", "discarded", "discarded"),
        ("trial/discarded", "trial", "discarded"),
    ):
        case = os.path.join(tmp, label.replace("/", "-"))
        os.makedirs(case)
        run_dir, charter, path = make_charter(case, run_id=label.replace("/", "-"))
        preflight.approve(path)
        result = {"status": "ok", "exit_code": 0, "passed": True,
                  "duration_s": 0.1, "failures": [], "tail": [],
                  "run_id": label, "trial_index": 0, "arm": "a", "seed": 1,
                  "attempt": "a"}
        base = {"trial_index": 0, "attempt": "a", "result": result}
        if event1 == "trial":
            one = {"event": event1, **base, **testkit.stamp()}
        else:
            one = {"event": event1, **base, "cause": "test", **testkit.stamp()}
        if event2 == "trial":
            two = {"event": event2, **base, **testkit.stamp()}
        else:
            two = {"event": event2, **base, "cause": "test", **testkit.stamp()}
        fired = _append_duplicate(run_dir, one, two)
        ok, detail = testkit.stopped_because(run_dir, "duplicate", fired.stderr,
                                             returncode=fired.returncode)
        check(f"duplicate {label} resolution pair is corruption", ok,
              detail or fired.stdout)


def _run_in_process(run_dir, replace_at=None, replace_on_unlock=False):
    import iterate
    original = ledger.validate_identity
    original_flock = ledger.fcntl.flock
    calls = [0]
    replaced = [False]

    def checked(fd, path, anchor, run_id=None):
        calls[0] += 1
        result = original(fd, path, anchor, run_id)
        if replace_at is not None and calls[0] == replace_at:
            stale = path + ".stale"
            os.rename(path, stale)
            open(path, "w").close()
        return result

    def checked_flock(fd, operation):
        if (replace_on_unlock and operation == ledger.fcntl.LOCK_UN
                and not replaced[0]):
            replaced[0] = True
            os.rename(os.path.join(run_dir, "iterations.jsonl"),
                      os.path.join(run_dir, "iterations.jsonl.stale"))
            open(os.path.join(run_dir, "iterations.jsonl"), "w").close()
        return original_flock(fd, operation)

    old_argv = sys.argv
    old_supervised = os.environ.get("GOAL_LOOP_SUPERVISED")
    os.environ["GOAL_LOOP_SUPERVISED"] = "1"
    stdout, stderr = io.StringIO(), io.StringIO()
    ledger.validate_identity = checked
    ledger.fcntl.flock = checked_flock
    try:
        sys.argv = ["iterate.py", "--supervised", run_dir]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                iterate.main()
            except SystemExit as exc:
                return exc.code, stdout.getvalue(), stderr.getvalue(), calls[0]
    finally:
        sys.argv = old_argv
        if old_supervised is None:
            os.environ.pop("GOAL_LOOP_SUPERVISED", None)
        else:
            os.environ["GOAL_LOOP_SUPERVISED"] = old_supervised
        ledger.validate_identity = original
        ledger.fcntl.flock = original_flock
    return 0, stdout.getvalue(), stderr.getvalue(), calls[0]


def test_replacement_after_append_fsync_is_untrusted(tmp):
    run_dir, _ = approve(tmp)
    code, out, err, calls = _run_in_process(run_dir, replace_at=4)
    ok, detail = testkit.stopped_because(run_dir, "identity", out + err,
                                         returncode=code)
    check("replacement after append fsync is detected before unlock",
          code != 0 and ok and calls >= 5, detail or err)
    check("identity corruption does not write a sentinel in the run path",
          not os.path.exists(os.path.join(run_dir, "STOP")), out + err)


def test_replacement_between_final_check_and_unlock_is_untrusted(tmp):
    run_dir, _ = approve(tmp)
    code, out, err, calls = _run_in_process(run_dir, replace_on_unlock=True)
    testkit.LAST_RETURN_CODE = code
    ok, detail = testkit.stopped_because(run_dir, "identity", out + err,
                                         returncode=code)
    check("replacement between final check and unlock is classified corrupt",
          code != 0 and ok and calls >= 5, detail or err)


def test_whole_directory_replacement_creates_nothing(tmp):
    run_dir, charter = approve(tmp)
    old = run_dir + ".old"
    os.rename(run_dir, old)
    os.mkdir(run_dir)
    shutil.copy(os.path.join(old, "charter.json"),
                os.path.join(run_dir, "charter.json"))
    open(os.path.join(run_dir, "iterations.jsonl"), "w").close()
    result = testkit.fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "identity", result.stderr,
                                         returncode=result.returncode)
    created = [name for name in ("results", "STOP", "lease")
               if os.path.exists(os.path.join(run_dir, name))]
    check("whole-directory replacement halts on anchor mismatch", ok,
          detail or result.stderr)
    check("whole-directory replacement creates no continuation artifacts",
          not created, f"replacement contains {created}; charter={charter}")


def test_partial_bootstrap_is_not_repaired(tmp):
    run_dir, charter = approve(tmp)
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    os.unlink(anchor)
    before = open(os.path.join(run_dir, "iterations.jsonl"), "rb").read()
    try:
        preflight.approve(os.path.join(run_dir, "charter.json"))
    except RuntimeError as exc:
        raised = "unrecoverable" in str(exc)
    else:
        raised = False
    check("approval does not repair ledger-without-anchor partial state", raised)
    check("partial bootstrap leaves the original ledger untouched",
          open(os.path.join(run_dir, "iterations.jsonl"), "rb").read() == before)


def test_path_disappearance_is_identity_corruption(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    original_stat = ledger.os.stat
    try:
        def disappear(target):
            if os.path.abspath(os.fspath(target)) == os.path.abspath(path):
                os.unlink(path)
            return original_stat(target)
        ledger.os.stat = disappear
        try:
            ledger.validate_identity(fd, path, anchor, charter["run_id"])
        except Exception as exc:
            classified = isinstance(exc, ledger.IdentityCorrupt)
        else:
            classified = False
    finally:
        ledger.os.stat = original_stat
        os.close(fd)
    check("ledger path disappearance is classified as identity corruption",
          classified)


def test_identity_replacement_matrix_has_no_valid_stop_then_trial(tmp):
    # The integer stages drive replacement at successive validate_identity calls
    # inside _run_in_process; they are labelled by which identity check catches
    # them, not by a literal syscall boundary. Every case must halt before any
    # trial is accepted, which is the property under test.
    stages = (
        ("ledger missing entirely", "missing"),
        ("replaced at identity check 1", 1),
        ("replaced at identity check 2", 2),
        ("replaced at identity check 3", 3),
        ("replaced at identity check 4", 4),
        ("replaced at unlock check", "unlock"),
    )
    for name, stage in stages:
        case = os.path.join(tmp, name.replace(" ", "-"))
        os.makedirs(case)
        run_dir, _ = approve(case)
        if stage == "missing":
            os.unlink(os.path.join(run_dir, "iterations.jsonl"))
            result = testkit.fire(run_dir)
            code, output = result.returncode, result.stdout + result.stderr
        else:
            code, stdout, stderr, _ = _run_in_process(
                run_dir, replace_at=stage if isinstance(stage, int)
                else None, replace_on_unlock=stage == "unlock")
            output = stdout + stderr
        expected = "ledger" if stage == "missing" else "identity"
        ok, detail = testkit.stopped_because(run_dir, expected, output,
                                             returncode=code)
        current = os.path.join(run_dir, "iterations.jsonl")
        events = ledger.read(current) if os.path.exists(current) else []
        check(f"replacement {name} is classified before a trial is accepted",
              code != 0 and ok and not any(r["event"] == "trial"
                                           for r in events), detail or output)


def test_stale_inode_fsync_crash_fails_closed_on_restart(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    anchor = preflight.anchor_path(run_dir, charter["run_id"])
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    try:
        ledger.fcntl.flock(fd, ledger.fcntl.LOCK_EX)
        ledger.append_fd(fd, path, {"event": "run_start", "charter": charter,
                                    **testkit.stamp()})
        os.rename(path, path + ".stale")
        open(path, "w").close()
    finally:
        ledger.fcntl.flock(fd, ledger.fcntl.LOCK_UN)
        os.close(fd)
    result = testkit.fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "identity", result.stderr,
                                         returncode=result.returncode)
    check("restart detects replacement after stale-inode fsync crash", ok,
          detail or result.stderr)
    check("restart does not initialize the replacement ledger",
          open(path, "rb").read() == b"" and
          not os.path.exists(os.path.join(run_dir, "STOP")),
          "replacement ledger or sentinel was changed")


def test_ambiguous_recovery_adopts_nothing_and_trials_from_counts_once(tmp):
    run_dir, charter = approve(tmp)
    path = os.path.join(run_dir, "iterations.jsonl")
    result = {"run_id": charter["run_id"], "trial_index": 0, "arm": "a",
              "seed": 1, "attempt": "ambiguous", "status": "ok",
              "exit_code": 0, "passed": True, "duration_s": 1.0,
              "failures": [], "tail": []}
    record = {"trial_index": 0, "attempt": "ambiguous", "result": result,
              **testkit.stamp()}
    ledger.append(path, {"event": "iteration_start", **record})
    ledger.append(path, {"event": "trial", **record})
    ledger.append(path, {"event": "discarded", **record, "cause": "ambiguous"})
    before = open(path, "rb").read()
    fired = testkit.fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "duplicate", fired.stderr,
                                         returncode=fired.returncode)
    after = open(path, "rb").read()
    check("ambiguous recovery halts for duplicate resolution", ok,
          detail or fired.stderr)
    check("ambiguous recovery adopts no additional record", before == after,
          "ledger changed during ambiguous recovery")
    try:
        import iterate
        iterate.trials_from([{"event": "trial", **record},
                             {"event": "trial", **record}])
    except ledger.Corrupt:
        counted_once = True
    else:
        counted_once = False
    check("trials_from does not double-count an ambiguous pair", counted_once)


def test_stopped_because_rejects_success_output_without_failure_evidence(tmp):
    testkit.LAST_OUTPUT = "STOP: identity mismatch"
    testkit.LAST_RETURN_CODE = 0
    no_false_positive, _ = testkit.stopped_because(tmp, "identity")
    testkit.LAST_RETURN_CODE = 2
    ledger_path = os.path.join(tmp, "iterations.jsonl")
    open(ledger_path, "w").close()
    ledger.append(ledger_path, {"event": "stop", "reason": "identity mismatch",
                                "summary": {}, **testkit.stamp()})
    with_failure, _ = testkit.stopped_because(tmp, "identity")
    check("stopped_because requires failing STOP evidence",
          not no_false_positive and with_failure)


def main():
    tests = [test_approve_bootstraps_ledger_and_external_anchor,
             test_path_like_run_id_cannot_escape_anchor_parent,
             test_first_identity_validation_happens_after_flock,
             test_replacement_between_open_and_flock_is_detected_after_lock,
             test_exceptional_body_replacement_is_classified_and_released,
             test_finish_error_replacement_is_classified_by_context_cleanup,
             test_continuation_without_anchor_fails_without_side_effects,
             test_lease_file_is_not_used,
             test_stop_before_first_iteration_is_durable,
             test_duplicate_iteration_start_is_corruption,
             test_replacement_after_append_fsync_is_untrusted,
             test_replacement_between_final_check_and_unlock_is_untrusted,
             test_whole_directory_replacement_creates_nothing,
             test_partial_bootstrap_is_not_repaired,
             test_path_disappearance_is_identity_corruption,
             test_identity_replacement_matrix_has_no_valid_stop_then_trial,
             test_stale_inode_fsync_crash_fails_closed_on_restart,
             test_ambiguous_recovery_adopts_nothing_and_trials_from_counts_once,
             test_stopped_because_rejects_success_output_without_failure_evidence]
    for test in tests:
        tmp = tempfile.mkdtemp(prefix="goal-loop-fd-red-")
        try:
            test(tmp)
        except Exception as exc:
            check(test.__name__, False, f"test itself raised: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(prefix="goal-loop-fd-red-duplicates-")
    try:
        test_duplicate_resolution_pairs_are_corruption(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
