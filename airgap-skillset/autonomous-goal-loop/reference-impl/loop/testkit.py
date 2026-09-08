"""Shared test helpers, written because green was not telling the truth.

Checks asserted that the run had stopped and never asked why. A stop is a stop
whatever caused it, so a test could inject a malformed artifact, watch the run
halt for an unrelated clock anomaly, and report success — with the artifact it
was supposedly testing never even read.

Two rules follow, and everything here exists to enforce them:

1. Assert the reason, not only the outcome. `stopped_because` fails when the run
   stopped for something other than what the test is about.
2. Build fixtures whose timestamps are internally consistent. Hardcoded historic
   `at` values with a small `mono` look exactly like a tampered clock, which is
   what silently hijacked the suites.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ledger

LAST_OUTPUT = ""
LAST_RETURN_CODE = 0


def boot():
    return open("/proc/sys/kernel/random/boot_id").read().strip()


def stamp(ago_s=0.0):
    """A timestamp whose wall and monotonic components agree.

    `ago_s` moves both together, so a fixture can look older without looking
    like the clock was moved.
    """
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() - ago_s)),
            "mono": round(time.monotonic() - ago_s, 3),
            "boot": boot()}


def fire(run_dir, times=1, timeout=300, supervised=True):
    """Run an iteration, using the explicit supervised test entry when asked."""
    env = dict(os.environ)
    env.pop("GOAL_LOOP_SUPERVISED", None)
    command = [os.path.join(HERE, "run-supervised.sh" if supervised else "run.sh")]
    command.append(run_dir)
    global LAST_OUTPUT, LAST_RETURN_CODE
    out = []
    for _ in range(times):
        out.append(subprocess.run(command,
                                  capture_output=True, text=True, timeout=timeout,
                                  env=env))
    LAST_OUTPUT = "\n".join((r.stdout or "") + (r.stderr or "") for r in out)
    LAST_RETURN_CODE = max((r.returncode for r in out), default=0)
    return out if times > 1 else out[0]


def stop_reason(run_dir):
    """The recorded reason, preferring the ledger over the sentinel text."""
    path = os.path.join(run_dir, "iterations.jsonl")
    try:
        records = ledger.read(path)
    except ledger.Corrupt:
        records = []
    for record in reversed(records):
        if record["event"] == "stop":
            return str(record.get("reason", ""))
    sentinel = os.path.join(run_dir, "STOP")
    if os.path.exists(sentinel):
        return open(sentinel).read().strip()
    return None


def stopped_because(run_dir, expected, output="", returncode=None):
    """(ok, detail). Stopping for the wrong reason is a failure, not a pass.

    A stop whose reason is the generic catch-all ("unexpected error, human
    required") never counts as stopping for a specific reason — that is the whole
    point of naming one, so a missing validator that falls through to the
    backstop cannot pass a test that names a precise cause."""
    reason = stop_reason(run_dir)
    if reason is None:
        return False, "the run did not stop at all"
    if "unexpected error" in reason.lower() and "unexpected error" not in expected.lower():
        return False, f"fell through to the catch-all, not the named check: {reason!r}"
    if expected.lower() not in reason.lower():
        return False, f"stopped for an unrelated reason: {reason!r}"
    return True, reason


def did_not_stop(run_dir):
    reason = stop_reason(run_dir)
    return reason is None, f"unexpectedly stopped: {reason!r}"


def events(run_dir):
    return [r["event"] for r in
            ledger.read(os.path.join(run_dir, "iterations.jsonl"))]


def charter(tmp, run_id, **overrides):
    """A charter whose fixture defaults do not themselves trip a checker."""
    base = {
        "run_id": run_id, "class": "evidence", "subject": "toy",
        "execution_mode": "supervised",
        "arms": {"a": {"argv": [sys.executable, "-c", "raise SystemExit(0)"], "cwd": tmp}},
        "notify": {"command": ["true"]},
        "containment": {"required": False}, "supervised": True,
        "stopping_rule": {"min_trials_per_arm": 20, "max_trials": 40, "ci_width": 0.01},
        "budgets": {"max_wallclock_s": 600, "max_infra_retries": 3,
                    "max_trial_duration_s": 30},
        "approved_by": "test",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    base.update(overrides)
    return base


def write_charter(run_dir, charter_dict):
    import preflight
    path = os.path.join(run_dir, "charter.json")
    json.dump(charter_dict, open(path, "w"))
    preflight.approve(path)
    return path
