"""Process, budget, validation and containment bounds — written before their fixes.

Same discipline as the other red suites: every check here was observed failing,
for the reason in its name, against the code it corrects.

Run: python3 loop/bounds_test.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clock
import ledger
import preflight
import testkit

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  <- {detail}'}")


def boot():
    return open("/proc/sys/kernel/random/boot_id").read().strip()


def stamp(**over):
    base = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mono": round(time.monotonic(), 3), "boot": boot()}
    base.update(over)
    return base


def make_run(tmp, **overrides):
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    charter = {
        "run_id": "bounds", "class": "evidence", "subject": "toy",
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


def fire(run_dir):
    return testkit.fire(run_dir)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# --- process tree ------------------------------------------------------------

def test_descendant_ignoring_sigterm_is_killed(tmp):
    """_kill_tree sends SIGTERM to the group, then returns as soon as the direct
    child exits. A descendant that ignores SIGTERM survives, because SIGKILL is
    never sent once the leader is gone."""
    marker = os.path.join(tmp, "child.pid")
    subject = (
        "import os,subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',"
        f"\"import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);\"\n"
        f" \"open({marker!r},'w').write(str(os.getpid()));time.sleep(120)\"])\n"
        "time.sleep(0.5)\n"                      # leader exits early, child lives on
    )
    spec = json.dumps({"argv": [sys.executable, "-c", subject], "cwd": tmp,
                       "run_id": "bounds", "trial_index": 0, "arm": "a",
                       "attempt": "aaaa"})
    artifact = os.path.join(tmp, "t.json")
    subprocess.run([sys.executable, os.path.join(HERE, "collector.py"),
                    spec, "1", "3", artifact], capture_output=True, timeout=60)
    time.sleep(1)

    if not os.path.exists(marker):
        return check("a descendant ignoring SIGTERM is killed", False,
                     "test could not start the descendant")
    pid = int(open(marker).read())
    survived = alive(pid)
    if survived:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    check("a descendant ignoring SIGTERM is killed with the group", not survived,
          f"pid {pid} outlived the trial")


# --- budget ------------------------------------------------------------------

def test_trial_timeout_is_actually_clipped(tmp):
    """The existing budget check only asserts that a helper symbol exists. Time a
    real trial against a nearly exhausted budget instead."""
    run_dir, _ = make_run(tmp, budgets={"max_wallclock_s": 3, "max_infra_retries": 3,
                                        "max_trial_duration_s": 120},
                          arms={"a": {"argv": [sys.executable, "-c",
                                               "import time;time.sleep(120)"],
                                      "cwd": tmp}})
    started = time.monotonic()
    fire(run_dir)
    elapsed = time.monotonic() - started
    # A generous ceiling passes substantial overruns. The budget is 3s; allow
    # the orchestrator's own overhead and nothing like a full trial.
    check("a trial is stopped by the remaining wall-clock budget, not its own limit",
          elapsed < 15, f"trial ran {elapsed:.0f}s against a 3s remaining budget")


def test_reboot_downtime_is_accounted(tmp):
    """clock.elapsed adds nothing between the last record of a previous boot and
    now, so a run can restart with its budget apparently untouched after hours of
    downtime."""
    records = [
        {"at": "2026-08-29T00:00:00+0900", "mono": 10.0, "boot": "previous-boot"},
        {"at": "2026-08-29T00:00:30+0900", "mono": 40.0, "boot": "previous-boot"},
    ]
    spent, anomaly = clock.elapsed(records)
    # Those records are from another boot and hours in the past. Either the gap
    # to now is counted, or the run must refuse to guess.
    check("downtime since a previous boot is counted or refused",
          anomaly is not None or spent > 3600,
          f"elapsed={spent:.0f}s anomaly={anomaly}")


# --- fail-closed validation --------------------------------------------------

def test_malformed_ledger_record_does_not_poison(tmp):
    """Only fresh artifacts are shape-checked. A malformed result already in the
    ledger raises inside the evaluator on every later firing."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(lp, {"event": "trial", "trial_index": 0,
                       "attempt": "malformed-result",
                       "result": {"status": "ok"}, **testkit.stamp()})  # no `passed`
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "malformed")
    check("a malformed ledger record stops the run", ok, detail)
    check("a malformed ledger record does not raise",
          "Traceback" not in result.stderr, result.stderr[-160:])


def test_notify_command_type_is_checked(tmp):
    """notify.command is documented as an argv list but only checked for
    truthiness, so a string or integer reaches subprocess and raises."""
    run_dir, _ = make_run(tmp, notify={"command": 42})
    # Force the terminal sequence, or notify is never reached and the check
    # passes without exercising anything.
    with open(os.path.join(run_dir, "STOP"), "w") as fh:
        fh.write("owner stop\n")
    result = fire(run_dir)
    check("a non-list notify.command is refused before use",
          "Traceback" not in result.stderr, result.stderr[-160:])
    ok, detail = testkit.stopped_because(run_dir, "notify.command")
    check("a non-list notify.command stops with a naming reason", ok, detail)


# --- containment -------------------------------------------------------------

def test_waived_containment_is_refused_when_unattended(tmp):
    """required:false is an unrestricted waiver. Nothing ties it to a supervised
    run, so an unattended run can simply switch containment off."""
    charter = {"containment": {"required": False}, "supervised": False}
    reason = preflight.check_containment(charter)
    check("containment cannot be waived for an unattended run",
          reason is not None, "an unattended run waived containment")


# --- containment evidence bound to the charter --------------------------------

def _unit(tmp, body=None):
    path = os.path.join(tmp, "loop.service")
    if body is None:
        writable = os.path.realpath(os.path.join(tmp, "writable"))
        body = ("[Service]\nProtectHome=read-only\nRestrictNamespaces=yes\n"
                f"ExecStart={os.path.join(HERE, 'run.sh')} {os.path.join(tmp, 'run')}\n"
                "Environment=GOAL_LOOP_SUPERVISED=\n"
                f"ReadWritePaths={writable}\n")
    open(path, "w").write(body)
    return path


PROBE_SRC = os.path.join(HERE, "negative_test.py")


def _good_probes():
    import negative_test
    return {name: {"control": "succeeded", "contained": "blocked"}
            for name in negative_test.PROBES}


def _contained(tmp, **over):
    import negative_test
    unit = _unit(tmp)
    src = preflight.file_digest(PROBE_SRC)
    writable = os.path.realpath(os.path.join(tmp, "writable"))
    targets = {
        negative_test.HOME_PROBE: os.path.realpath(os.path.join(tmp, "home-target")),
        negative_test.REPO_PROBE: os.path.realpath(os.path.join(tmp, "repo-target")),
    }
    c = {"required": True, "probe_path": os.path.join(tmp, "probe"),
         "writable_paths": [writable], "probe_targets": targets,
         "unit": unit, "unit_sha256": preflight.file_digest(unit),
         "probe_source": PROBE_SRC, "probe_source_sha256": src,
          "verified_at": "2026-08-30T00:00:00+0900",
          "negative_test": {"at": "2026-08-30T00:00:00+0900",
                           "probe_source_sha256": src, "targets": targets,
                           "probes": _good_probes()}}
    c.update(over)
    return c


def test_unit_must_match_the_one_that_was_tested(tmp):
    """The negative test says something about the unit that was loaded when it
    ran. Editing the unit afterwards silently invalidates that evidence."""
    containment = _contained(tmp)
    open(containment["unit"], "a").write("PrivateNetwork=no\n")   # edited after approval
    reason = preflight.check_containment_evidence(
        {"containment": containment, "arms": {}}, tmp)
    check("an edited containment unit invalidates its negative test",
          reason is not None, "the unit changed after the evidence was recorded")


def test_unblocked_probe_is_refused(tmp):
    """A probe that ran uncontained but was not blocked under containment is an
    escape, not evidence. Keep a full, well-formed probes map so the generic
    'needs a probes map' guard cannot pass this by accident."""
    src = preflight.file_digest(PROBE_SRC)
    probes = _good_probes()
    probes["undeclared network connection"]["contained"] = "succeeded"   # escaped
    reason = preflight.check_containment_evidence(
        {"containment": _contained(tmp, negative_test={
            "at": "x", "probe_source_sha256": src, "probes": probes}), "arms": {}}, tmp)
    check("a probe not blocked under containment is refused",
          reason is not None and "not blocked" in reason, str(reason))


def test_probe_without_a_control_is_refused(tmp):
    """A probe blocked under containment but that never succeeded uncontained
    demonstrates nothing — it might always fail."""
    src = preflight.file_digest(PROBE_SRC)
    probes = _good_probes()
    probes["undeclared network connection"]["control"] = "failed"        # no control
    reason = preflight.check_containment_evidence(
        {"containment": _contained(tmp, negative_test={
            "at": "x", "probe_source_sha256": src, "probes": probes}), "arms": {}}, tmp)
    check("a probe with no uncontained control is refused",
          reason is not None and "uncontained" in reason, str(reason))


def test_run_probes_classifies_unrelated_errors_inconclusive(tmp):
    """An unrelated errno is inconclusive, not evidence of containment."""
    import negative_test
    original = negative_test.PROBES

    def unrelated():
        raise OSError(28, "no space left on device")

    negative_test.PROBES = {"injected unrelated errno": unrelated}
    try:
        live = negative_test.run_probes({})
    finally:
        negative_test.PROBES = original
    check("an unrelated probe errno is classified inconclusive",
          live == {"injected unrelated errno": "inconclusive"}, str(live))


def test_evidence_requires_the_canonical_probe_set(tmp):
    """One invented probe with matching digests used to satisfy the check."""
    src = preflight.file_digest(PROBE_SRC)
    reason = preflight.check_containment_evidence(
        {"containment": _contained(tmp, negative_test={
            "at": "x", "probe_source_sha256": src,
            "probes": {"foo": {"control": "succeeded", "contained": "blocked"}}}),
         "arms": {}}, tmp)
    check("evidence naming the wrong probe set is refused",
          reason is not None and "probe" in reason.lower(), str(reason))


def test_control_plane_inside_the_subject_is_refused(tmp):
    """The control plane must survive the subject being restored or removed."""
    subject = os.path.join(tmp, "subject")
    os.makedirs(subject, exist_ok=True)
    run_dir = os.path.join(subject, ".loop")
    os.makedirs(run_dir, exist_ok=True)
    reason = preflight.check_containment_evidence(
        {"containment": _contained(tmp), "arms": {"a": {"argv": ["true"], "cwd": subject}}},
        run_dir)
    check("a control plane inside the subject is refused",
          reason is not None, "the ledger lives inside what the run observes")


# --- findings an independent review reproduced --------------------------------

def test_empty_negative_test_is_refused(tmp):
    """The evidence check compares aggregates, so zero probes satisfies
    blocked == total == control_succeeded."""
    reason = preflight.check_containment_evidence(
        {"containment": _contained(tmp, negative_test={
            "at": "x", "probe_source_sha256": preflight.file_digest(PROBE_SRC),
            "probes": {}}), "arms": {}}, tmp)
    check("a negative test with no probes is refused",
          reason is not None, "an empty probes map was accepted")


def test_start_without_an_attempt_is_refused(tmp):
    """A legacy or malformed iteration_start with no attempt id must not adopt
    trial-<n>-None.json. It is now caught as corruption on read."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    # Raw write to bypass append's own validation and mimic a corrupt ledger.
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "iteration_start", "seq": 1, "trial_index": 0,
                             "arm": "a", "seed": 1, **testkit.stamp()}) + "\n")
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": 1,
               "attempt": None, "status": "ok", "exit_code": 0, "passed": True,
               "duration_s": 55.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", "trial-0-None.json"), "w"))
    fire(run_dir)
    # The ledger still holds the corrupt record, so read defensively.
    try:
        trials = [r for r in ledger.read(lp) if r["event"] == "trial"]
    except ledger.Corrupt:
        trials = []
    ok, detail = testkit.stopped_because(run_dir, "iteration_start record")
    check("a start record without an attempt id is not reconciled into a trial",
           not any(t.get("result", {}).get("duration_s") == 55.0 for t in trials),
           "an artifact with a null attempt was adopted")
    check("a start record without an attempt id stops the run",
          ok, detail)


def test_ledger_record_without_seq_is_handled(tmp):
    """append() derives the next seq from the last record, so a record missing
    seq raises KeyError with no terminal handling."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "run_start", "charter": {},
                             **testkit.stamp()}) + "\n")       # no seq
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "seq")
    check("a ledger record missing seq stops as corruption", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_supervised_is_not_self_attested(tmp):
    """The charter cannot attest to how it was invoked. A charter that waives
    containment and marks itself supervised passes the whole entry gate."""
    run_dir, _ = make_run(tmp, containment={"required": False}, supervised=True)
    os.environ.pop("GOAL_LOOP_SUPERVISED", None)
    reason = preflight.run(json.load(open(os.path.join(run_dir, "charter.json"))), run_dir)
    check("supervision cannot be claimed by the charter alone",
          reason is not None,
          "a self-declared supervised charter waived containment")


def test_accept_over_budget_records_discarded_not_trial(tmp):
    """Both the normal and salvage acceptance paths must refuse an over-budget
    result. The normal path was fixed; salvage adopted the artifact directly."""
    import iterate, ledger as L
    run_dir, charter = make_run(
        tmp, budgets={"max_wallclock_s": 1, "max_infra_retries": 3,
                      "max_trial_duration_s": 30})
    lp = os.path.join(run_dir, "iterations.jsonl")
    # A run that started two seconds ago, budget is one second.
    L.append(lp, {"event": "run_start", "charter": charter, **testkit.stamp(ago_s=2)})
    result = {"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": 1,
              "attempt": "x", "status": "ok", "exit_code": 0, "passed": True,
              "duration_s": 0.1, "failures": [], "tail": []}
    iterate.accept(run_dir, lp, charter, 0, "a", "x", result, ["a"], lambda: {})
    kinds = [r["event"] for r in L.read(lp)]
    check("an over-budget result is recorded as discarded, not a trial",
          "trial" not in kinds and "discarded" in kinds, f"ledger has {kinds}")


def test_budget_read_is_the_acceptance_decision(tmp):
    """A deadline crossing after the decision read does not revoke the trial."""
    import iterate
    run_dir, charter = make_run(
        tmp, budgets={"max_wallclock_s": 1, "max_infra_retries": 3,
                      "max_trial_duration_s": 30})
    lp = os.path.join(run_dir, "iterations.jsonl")
    result = {"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": 1,
              "attempt": "budget-gap", "status": "ok", "exit_code": 0,
              "passed": True, "duration_s": 0.1, "failures": [], "tail": []}
    calls = []
    trial_appends = []
    original_elapsed = iterate.clock.elapsed
    original_append = iterate.ledger.append_fd

    def decision_read(records):
        calls.append("elapsed")
        return 0.5, None

    def delayed_append(fd, path, record):
        if record["event"] == "trial":
            calls.append("trial_append")
            trial_appends.append(record)
            time.sleep(1.1)
        return original_append(fd, path, record)

    iterate.clock.elapsed = decision_read
    iterate.ledger.append_fd = delayed_append
    try:
        iterate.accept(run_dir, lp, charter, 0, "a", "budget-gap", result,
                       ["a"], lambda: {})
    finally:
        iterate.clock.elapsed = original_elapsed
        iterate.ledger.append_fd = original_append
    records = ledger.read(lp)
    check("the budget decision read precedes the trial append exactly once",
          calls == ["elapsed", "trial_append"], f"acceptance calls: {calls}")
    check("acceptance performs exactly one trial append", len(trial_appends) == 1,
          f"trial append count: {len(trial_appends)}")
    check("a trial accepted before the deadline remains valid after append delay",
          any(r["event"] == "trial" for r in records),
          f"ledger has {[r['event'] for r in records]}")


def test_crash_at_trial_and_stop_write_fsync_does_not_reorder_or_merge(tmp):
    """Torn writes heal before a later append, including a failed fsync."""
    run_dir, charter = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    ledger.append(lp, {"event": "run_start", "charter": charter,
                       **testkit.stamp()})
    result = {"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": 1,
              "attempt": "torn-trial", "status": "ok", "exit_code": 0,
              "passed": True, "duration_s": 0.1, "failures": [], "tail": []}
    original_write = ledger.os.write
    failed = [False]

    def tear_once(fd, data):
        if not failed[0]:
            failed[0] = True
            original_write(fd, data[:len(data) // 2])
            raise OSError("injected crash during trial write")
        return original_write(fd, data)

    ledger.os.write = tear_once
    try:
        try:
            ledger.append(lp, {"event": "trial", "trial_index": 0,
                               "attempt": "torn-trial", "result": result,
                               **testkit.stamp()})
        except OSError:
            pass
    finally:
        ledger.os.write = original_write

    subprocess.run([os.path.join(HERE, "stop.sh"), run_dir], check=True,
                   capture_output=True, text=True)
    ledger.append(lp, {"event": "notify", "delivered": True, **testkit.stamp()})
    records = ledger.read(lp)
    kinds = [r["event"] for r in records]
    check("a torn trial tail is healed before the stop append",
          kinds == ["run_start", "stop", "notify"], f"ledger has {kinds}")
    check("a later valid append is not merged into partial trial JSON",
           all(r["event"] != "trial" for r in records)
           and records[-1]["event"] == "notify", f"ledger has {records}")
    stop_index = kinds.index("stop")
    check("a valid stop has no later valid trial", not any(
        r["event"] == "trial" for r in records[stop_index + 1:]),
          f"ledger has {kinds}")

    run3, charter = make_run(os.path.join(tmp, "torn-stop"))
    lp3 = os.path.join(run3, "iterations.jsonl")
    ledger.append(lp3, {"event": "run_start", "charter": charter,
                        **testkit.stamp()})
    original_write = ledger.os.write
    failed = [False]

    def tear_stop_once(fd, data):
        if not failed[0]:
            failed[0] = True
            original_write(fd, data[:len(data) // 2])
            raise OSError("injected crash during stop write")
        return original_write(fd, data)

    ledger.os.write = tear_stop_once
    try:
        try:
            ledger.append(lp3, {"event": "stop", "reason": "owner stop",
                                "summary": {}, **testkit.stamp()})
        except OSError:
            pass
    finally:
        ledger.os.write = original_write
    ledger.append(lp3, {"event": "notify", "delivered": True, **testkit.stamp()})
    healed = ledger.read(lp3)
    check("a torn stop tail is ignored before the next valid append",
          [r["event"] for r in healed] == ["run_start", "notify"]
          and healed[-1].get("healed_bytes", 0) > 0,
          f"ledger has {healed}")
    subprocess.run([os.path.join(HERE, "stop.sh"), run3], check=True,
                   capture_output=True, text=True)
    fire(run3)
    healed = ledger.read(lp3)
    stop_index = [r["event"] for r in healed].index("stop")
    ok, detail = testkit.stopped_because(run3, "owner stop")
    check("a later valid append after a torn stop is parseable and terminal",
          ok and not any(r["event"] == "trial" for r in healed[stop_index + 1:]),
          detail)

    run2, charter = make_run(os.path.join(tmp, "fsync"))
    lp2 = os.path.join(run2, "iterations.jsonl")
    original_fsync = ledger.os.fsync
    failed = [False]

    def fail_stop_fsync(fd):
        if not failed[0]:
            failed[0] = True
            raise OSError("injected crash during stop fsync")
        return original_fsync(fd)

    ledger.os.fsync = fail_stop_fsync
    try:
        try:
            ledger.append(lp2, {"event": "stop", "reason": "owner stop",
                                "summary": {}, **testkit.stamp()})
        except OSError:
            pass
    finally:
        ledger.os.fsync = original_fsync
    records2 = ledger.read(lp2)
    fire(run2)
    records2 = ledger.read(lp2)
    ok, detail = testkit.stopped_because(run2, "owner stop")
    check("a complete stop write remains ordered before any later trial",
          ok and not any(r["event"] == "trial" for r in records2), detail)


def test_first_ledger_creation_fsyncs_parent_directory(tmp):
    """Bootstrap must persist the new ledger file AND its parent directory entry.

    The old form instrumented a post-approval append, by which point the ledger
    already existed, so removing the parent fsync from bootstrap still passed.
    Instrument preflight.approve — the actual first-creation path.
    """
    import preflight
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "results"))
    charter_path = os.path.join(run_dir, "charter.json")
    json.dump(testkit.charter(tmp, "firstcreate"), open(charter_path, "w"))
    lp = os.path.join(run_dir, "iterations.jsonl")

    original_open = preflight.os.open
    original_fsync = preflight.os.fsync
    opened, synced = {}, []

    def tracked_open(path, flags, mode=0o777, *a, **k):
        fd = original_open(path, flags, mode, *a, **k)
        opened[fd] = os.path.abspath(os.fspath(path))
        return fd

    def tracked_fsync(fd):
        synced.append(opened.get(fd))
        return original_fsync(fd)

    preflight.os.open = tracked_open
    preflight.os.fsync = tracked_fsync
    try:
        preflight.approve(charter_path)
    finally:
        preflight.os.open = original_open
        preflight.os.fsync = original_fsync
    check("bootstrap fsyncs the new ledger file", lp in synced,
          f"fsynced paths: {synced}")
    check("bootstrap fsyncs the ledger's parent directory",
          os.path.abspath(run_dir) in synced, f"fsynced paths: {synced}")


def test_idempotent_stop_retries_directory_durability(tmp):
    """A failed parent sync is retried before an existing stop is acknowledged."""
    import stop
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    original_open = ledger.os.open
    original_fsync = ledger.os.fsync
    opened = {}
    synced = []
    failed_parent = [False]

    def tracked_open(path, flags, mode=0o777, *args, **kwargs):
        fd = original_open(path, flags, mode, *args, **kwargs)
        opened[fd] = os.path.abspath(os.fspath(path))
        return fd

    def tracked_fsync(fd):
        synced.append(opened.get(fd))
        return original_fsync(fd)

    def fail_parent_fsync(fd):
        path = opened.get(fd)
        synced.append(path)
        if path == run_dir and not failed_parent[0]:
            failed_parent[0] = True
            raise OSError("injected crash during parent directory fsync")
        return original_fsync(fd)

    ledger.os.open = tracked_open
    ledger.os.fsync = fail_parent_fsync
    try:
        try:
            ledger.append(lp, {"event": "stop", "reason": "owner stop",
                               "summary": {}, **testkit.stamp()})
        except OSError:
            pass
    finally:
        ledger.os.open = original_open
        ledger.os.fsync = original_fsync

    initial_synced = list(synced)
    synced = []
    opened = {}
    ledger.os.open = tracked_open
    ledger.os.fsync = tracked_fsync
    old_argv = sys.argv
    try:
        sys.argv = ["stop.py", run_dir]
        stop.main()
    finally:
        sys.argv = old_argv
        ledger.os.open = original_open
        ledger.os.fsync = original_fsync
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    records = ledger.read(lp)
    check("idempotent stop syncs the existing file and parent directory",
          failed_parent[0] and lp in initial_synced and ok
          and len([r for r in records if r["event"] == "stop"]) == 1
          and lp in synced and run_dir in synced,
          detail or f"fsynced paths: {synced}")


def test_append_does_not_acknowledge_short_successful_write(tmp):
    """A short successful write must complete or reject, never acknowledge."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    original_write = ledger.os.write
    shortened = [False]

    def short_once(fd, data):
        if not shortened[0]:
            shortened[0] = True
            return original_write(fd, data[:len(data) // 2])
        return original_write(fd, data)

    result = error = None
    ledger.os.write = short_once
    try:
        try:
            result = ledger.append(lp, {"event": "stop", "reason": "owner stop",
                                       "summary": {}, **testkit.stamp()})
        except OSError as exc:
            error = exc
    finally:
        ledger.os.write = original_write
    records = ledger.read(lp)
    completed = result is not None and len(records) == 1 and records[0]["event"] == "stop"
    rejected = error is not None and not records
    check("a short successful write is completed or rejected without acknowledgement",
          completed or rejected,
          f"returned={result!r}, error={error!r}, records={records!r}")


def test_malformed_ledger_event_is_refused(tmp):
    """Event records themselves are never validated. An event with no `event`
    key raises while the run is classifying its own history."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"seq": 1, "note": "not an event",
                             **testkit.stamp()}) + "\n")
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "event")
    check("a ledger record with no event type stops as corruption", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_non_mapping_charter_is_refused(tmp):
    """check_schema assumes a mapping and nested mappings."""
    run_dir, _ = make_run(tmp)
    with open(os.path.join(run_dir, "charter.json"), "w") as fh:
        json.dump(["not", "a", "charter"], fh)
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "mapping")
    check("a non-mapping charter stops with a naming reason", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_stop_before_launch_runs_no_trial(tmp):
    """A bare STOP present at entry is serialized before any new trial."""
    run_dir, _ = make_run(tmp)
    fire(run_dir)                                   # one clean trial, index 0
    before = len([r for r in ledger.read(os.path.join(run_dir, "iterations.jsonl"))
                  if r["event"] == "trial"])
    # Write the sentinel between firings, exactly as an owner would.
    with open(os.path.join(run_dir, "STOP"), "w") as fh:
        fh.write("owner stop\n")
    fire(run_dir)
    after = len([r for r in ledger.read(os.path.join(run_dir, "iterations.jsonl"))
                  if r["event"] == "trial"])
    ok, detail = testkit.stopped_because(run_dir, "owner stop")
    check("a sentinel present at entry prevents a new trial",
           after == before and ok, detail or f"trial count went {before} -> {after}")


def test_trial_record_without_result_is_corruption(tmp):
    """A trial event with no result reaches record['result'] in trials_from and
    raises, on this and every later firing."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "trial", "seq": 1, "trial_index": 0,
                             **testkit.stamp()}) + "\n")   # no result
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "result")
    check("a trial record with no result stops as corruption", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_null_arm_spec_is_refused(tmp):
    """arms: {'a': null} reaches 'argv' not in spec and raises."""
    run_dir, _ = make_run(tmp, arms={"a": None})
    result = fire(run_dir)
    check("a null arm spec does not raise",
           "Traceback" not in result.stderr, result.stderr[-160:])
    ok, detail = testkit.stopped_because(run_dir, "arm")
    check("a null arm spec is refused with a reason", ok, detail)


def test_string_seq_is_corruption(tmp):
    """A record whose seq is a string reaches last['seq'] + 1 and raises."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "run_start", "seq": "one",
                             **testkit.stamp()}) + "\n")
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "seq")
    check("a non-integer seq stops as corruption", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_any_unexpected_exception_fails_closed(tmp):
    """Whatever raises, the run must not exit with a bare traceback and keep the
    timer firing into the same crash. It must leave a durable sentinel."""
    run_dir, _ = make_run(tmp)
    import iterate
    original = iterate.iterate

    def explode(_, supervised=False):
        raise RuntimeError("injected unexpected")

    iterate.iterate = explode
    old_argv = sys.argv
    try:
        sys.argv = ["iterate.py", "--supervised", run_dir]
        try:
            iterate.main()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv
        iterate.iterate = original
    ok, detail = testkit.stopped_because(
        run_dir, "unexpected error, human required")
    check("a genuine unexpected exception uses the generic catch-all",
          ok and "RuntimeError: injected unexpected" in detail, detail)


def test_string_budget_is_refused(tmp):
    """max_trials: '40' reaches an int/str comparison in the evaluator."""
    run_dir, _ = make_run(
        tmp, stopping_rule={"min_trials_per_arm": 20, "max_trials": "40",
                            "ci_width": 0.01})
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "number")
    check("a non-numeric stopping-rule value is refused, not compared", ok,
          f"{detail} / {result.stderr[-120:]}")


def test_unknown_event_type_is_corruption(tmp):
    """Any string is accepted as an event; a reader classifying by type meets an
    event it does not know and behaves undefined."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "wat", "seq": 1, **testkit.stamp()}) + "\n")
    result = fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "unknown event")
    check("an unknown ledger event type is treated as corruption",
          "Traceback" not in result.stderr and ok, detail)


def test_terminal_entry_classifies_a_pending_artifact(tmp):
    """On restart with the sentinel present, the terminal route runs before
    reconcile, so a completed artifact from an interrupted iteration_start is
    never classified."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "cccccccccccccccc"
    import evaluator
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": evaluator.seed_for("bounds", "a", 0),
                       "attempt": attempt, **testkit.stamp()})
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a",
               "seed": evaluator.seed_for("bounds", "a", 0), "attempt": attempt,
               "status": "ok", "exit_code": 0, "passed": True, "duration_s": 0.1,
               "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"), "w"))
    with open(os.path.join(run_dir, "STOP"), "w") as fh:
        fh.write("owner stop\n")
    fire(run_dir)
    kinds = [r["event"] for r in ledger.read(lp)]
    check("a pending artifact is discarded when stopping, not adopted",
          "discarded" in kinds and "trial" not in kinds,
          f"expected discarded, got: {kinds}")


def test_newer_same_index_attempt_is_recovered(tmp):
    """pending_start resolved by trial_index alone, so an infra_retry for a
    crashed attempt marked the index resolved and a newer attempt that published
    then crashed was skipped."""
    import evaluator
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    seed = evaluator.seed_for("bounds", "a", 0)
    ledger.append(lp, {"event": "run_start", "charter": {}, **testkit.stamp()})
    # Attempt A: crashed with no artifact -> recorded infra_retry.
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": seed, "attempt": "a"*16, **testkit.stamp()})
    ledger.append(lp, {"event": "infra_retry", "trial_index": 0,
                       "attempt": "a"*16, "cause": "crash", **testkit.stamp()})
    # Attempt B: published, then crashed before acceptance.
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": seed, "attempt": "b"*16, **testkit.stamp()})
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": seed,
               "attempt": "b"*16, "status": "ok", "exit_code": 0, "passed": True,
               "duration_s": 63.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{'b'*16}.json"), "w"))
    fire(run_dir)
    trials = [r["result"] for r in ledger.read(lp) if r["event"] == "trial"]
    check("a newer same-index attempt's artifact is recovered, not skipped",
           any(t.get("duration_s") == 63.0 for t in trials),
           "attempt B's completed artifact was skipped")


def test_late_resolution_is_qualified_by_index_and_attempt(tmp):
    """A late resolution for A must not resolve B when both identities share an
    attempt token; B's still-unresolved artifact is the one to adopt."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    token = "shared-attempt"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0,
                       "arm": "a", "seed": 11, "attempt": token,
                       **testkit.stamp()})
    ledger.append(lp, {"event": "iteration_start", "trial_index": 1,
                       "arm": "a", "seed": 12, "attempt": token,
                       **testkit.stamp()})
    # This resolution arrived after B started and qualifies only A.
    ledger.append(lp, {"event": "infra_retry", "trial_index": 0,
                       "attempt": token, "cause": "late A resolution",
                       **testkit.stamp()})
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a", "seed": 11,
               "attempt": token, "status": "ok", "exit_code": 0, "passed": True,
               "duration_s": 63.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{token}.json"), "w"))
    json.dump({"run_id": "bounds", "trial_index": 1, "arm": "a", "seed": 12,
               "attempt": token, "status": "ok", "exit_code": 0, "passed": True,
               "duration_s": 64.0, "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-1-{token}.json"), "w"))
    fire(run_dir)
    trials = [r["result"] for r in ledger.read(lp) if r["event"] == "trial"]
    check("a late resolution selects the correct index-attempt start",
          sum(t.get("duration_s") == 64.0 for t in trials) == 1,
          "B was treated as resolved by A's late resolution")
    check("a late-resolved attempt is not double-adopted",
          sum(t.get("duration_s") == 63.0 for t in trials) == 0
          and [t.get("duration_s") for t in trials].count(64.0) == 1,
          f"adopted fingerprints: {[t.get('duration_s') for t in trials]}")


def test_resolution_without_attempt_stops_as_ledger_corruption(tmp):
    """A resolution record without its qualifying attempt is corruption, not an
    unresolved record that pending_start may silently skip."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "run_start", "seq": 1, "charter": {},
                             **testkit.stamp()}) + "\n")
        fh.write(json.dumps({"event": "infra_retry", "seq": 2,
                             "trial_index": 0, "cause": "crash",
                             **testkit.stamp()}) + "\n")
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "infra_retry record is missing")
    check("a resolution without an attempt stops for named ledger corruption",
          ok, detail)


def test_notify_delivered_must_be_boolean(tmp):
    """A string 'false' is truthy, so a notify record claiming delivered='false'
    suppressed the retry."""
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    with open(lp, "w") as fh:
        fh.write(json.dumps({"event": "stop", "seq": 1, "reason": "x", "summary": {},
                             **testkit.stamp()}) + "\n")
        fh.write(json.dumps({"event": "notify", "seq": 2, "delivered": "false",
                             **testkit.stamp()}) + "\n")
    fire(run_dir)
    ok, detail = testkit.stopped_because(
        run_dir, "notify.delivered must be a boolean")
    check("a non-boolean notify.delivered stops for that corruption reason",
          ok, detail)


def test_append_validates_before_writing(tmp):
    """append() must reject a malformed record, not defer to a later read, or the
    'corruption at write' contract is false."""
    import ledger as L
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    before = (os.path.exists(lp), open(lp, "rb").read() if os.path.exists(lp) else b"")
    raised = False
    detail = ""
    try:
        L.append(lp, {"event": "iteration_start", "trial_index": 0,
                      **testkit.stamp()})  # omit only attempt
    except L.Corrupt as exc:
        raised = True
        detail = str(exc)
    after = (os.path.exists(lp), open(lp, "rb").read() if os.path.exists(lp) else b"")
    check("append names the missing iteration_start attempt requirement",
          raised and "iteration_start record is missing ['attempt']" in detail,
          detail)
    check("append validation leaves ledger bytes unchanged", after == before,
          f"ledger changed from {before!r} to {after!r}")


def test_reconcile_over_budget_is_not_adopted(tmp):
    """Recovery adopting a pending artifact must go through the same budget gate
    as a fresh result; reconcile appended trial directly."""
    import evaluator
    run_dir, _ = make_run(
        tmp, budgets={"max_wallclock_s": 1, "max_infra_retries": 3,
                      "max_trial_duration_s": 30})
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "dddddddddddddddd"
    ledger.append(lp, {"event": "run_start", "charter": {}, **testkit.stamp(ago_s=5)})
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": evaluator.seed_for("bounds", "a", 0),
                       "attempt": attempt, **testkit.stamp(ago_s=4)})
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a",
               "seed": evaluator.seed_for("bounds", "a", 0), "attempt": attempt,
               "status": "ok", "exit_code": 0, "passed": True, "duration_s": 0.1,
               "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"), "w"))
    fire(run_dir)
    kinds = [r["event"] for r in ledger.read(lp)]
    check("a reconciled over-budget artifact is not adopted as a trial",
          "trial" not in kinds, f"ledger has {kinds}")


def test_terminal_reconcile_with_existing_stop_record(tmp):
    """A pending iteration_start followed by a stop record (no sentinel) must
    still have its artifact classified; reconcile looked only at the last
    record."""
    import evaluator
    run_dir, _ = make_run(tmp)
    lp = os.path.join(run_dir, "iterations.jsonl")
    attempt = "eeeeeeeeeeeeeeee"
    ledger.append(lp, {"event": "iteration_start", "trial_index": 0, "arm": "a",
                       "seed": evaluator.seed_for("bounds", "a", 0),
                       "attempt": attempt, **testkit.stamp()})
    json.dump({"run_id": "bounds", "trial_index": 0, "arm": "a",
               "seed": evaluator.seed_for("bounds", "a", 0), "attempt": attempt,
               "status": "ok", "exit_code": 0, "passed": True, "duration_s": 0.1,
               "failures": [], "tail": []},
              open(os.path.join(run_dir, "results", f"trial-0-{attempt}.json"), "w"))
    ledger.append(lp, {"event": "stop", "reason": "owner stop", "summary": {},
                       **testkit.stamp()})
    fire(run_dir)
    kinds = [r["event"] for r in ledger.read(lp)]
    check("a pending artifact behind a stop record is discarded",
          "discarded" in kinds and "trial" not in kinds,
          f"expected discarded, got: {kinds}")


def test_nan_budget_is_refused(tmp):
    """NaN passes isinstance(float) and makes every budget comparison false."""
    run_dir, _ = make_run(
        tmp, budgets={"max_wallclock_s": float("nan"), "max_infra_retries": 3,
                      "max_trial_duration_s": 30})
    fire(run_dir)
    ok, detail = testkit.stopped_because(run_dir, "finite")
    check("a non-finite budget is refused", ok, detail)


def test_negative_test_json_exits_nonzero_on_escape():
    """The probe's --json mode always exited zero, even reporting an escape,
    contradicting its own contract."""
    import subprocess
    # Uncontained, probes succeed (escape). --json must exit nonzero.
    r = subprocess.run([sys.executable, os.path.join(HERE, "negative_test.py"), "--json"],
                       capture_output=True, text=True, timeout=60)
    try:
        os.unlink(os.path.expanduser("~/.goal-loop-escape-probe"))
    except OSError:
        pass
    check("negative_test --json exits nonzero when a probe escapes",
          r.returncode != 0, f"exit {r.returncode} with output {r.stdout[:80]}")


def main():
    isolated = [test_descendant_ignoring_sigterm_is_killed,
                test_newer_same_index_attempt_is_recovered,
                test_late_resolution_is_qualified_by_index_and_attempt,
                test_resolution_without_attempt_stops_as_ledger_corruption,
                test_notify_delivered_must_be_boolean,
                test_append_validates_before_writing,
                test_reconcile_over_budget_is_not_adopted,
                test_terminal_reconcile_with_existing_stop_record,
                test_nan_budget_is_refused,
                test_any_unexpected_exception_fails_closed,
                test_string_budget_is_refused,
                test_unknown_event_type_is_corruption,
                test_terminal_entry_classifies_a_pending_artifact,
                test_stop_before_launch_runs_no_trial,
                test_trial_record_without_result_is_corruption,
                test_null_arm_spec_is_refused,
                test_string_seq_is_corruption,
                test_trial_timeout_is_actually_clipped,
                test_malformed_ledger_record_does_not_poison,
                test_notify_command_type_is_checked,
                test_waived_containment_is_refused_when_unattended,
                test_unit_must_match_the_one_that_was_tested,
                test_unblocked_probe_is_refused,
                test_probe_without_a_control_is_refused,
                test_run_probes_classifies_unrelated_errors_inconclusive,
                test_evidence_requires_the_canonical_probe_set,
                test_control_plane_inside_the_subject_is_refused,
                test_empty_negative_test_is_refused,
                test_start_without_an_attempt_is_refused,
                test_ledger_record_without_seq_is_handled,
                test_supervised_is_not_self_attested,
                 test_accept_over_budget_records_discarded_not_trial,
                 test_budget_read_is_the_acceptance_decision,
                 test_crash_at_trial_and_stop_write_fsync_does_not_reorder_or_merge,
                 test_first_ledger_creation_fsyncs_parent_directory,
                 test_idempotent_stop_retries_directory_durability,
                 test_append_does_not_acknowledge_short_successful_write,
                 test_malformed_ledger_event_is_refused,
                test_non_mapping_charter_is_refused]
    for test in isolated:
        tmp = tempfile.mkdtemp(prefix="goal-loop-bounds-")
        try:
            test(tmp)
        except Exception as exc:
            check(test.__name__, False, f"test itself raised: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    test_reboot_downtime_is_accounted(None)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
