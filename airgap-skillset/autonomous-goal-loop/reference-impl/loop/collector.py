"""Collector: runs one trial and emits a machine-readable artifact.

It is the only thing that produces an observation. It never returns prose for
something else to interpret, and it never decides whether the run should stop.

A trial is an argv plus a working directory, so any command that exits zero on
success is a valid subject. The artifact carries the exit code rather than an
interpretation of it, and enough identity to prove which trial it belongs to.
"""
import json
import os
import re
import signal
import subprocess
import tempfile
import sys
import time

# pytest prints one such line per failing test; capturing the ids turns "the
# suite failed" into "these tests failed", which is what flaky hunting needs.
PYTEST_FAILURE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)

# An unbounded capture lets a noisy command exhaust memory long before its
# timeout fires, so the trial dies in a way the loop cannot classify.
MAX_OUTPUT_BYTES = 1 << 20


def _kill_tree(proc, pgid):
    """Kill the whole process group without adding an uncharged grace period.

    The subject is started in its own session, so killing the direct child
    leaves its descendants running to race the next iteration.

    The group id is passed in rather than looked up: once `wait()` has reaped
    the leader its pid is gone, `os.getpgid` raises, and the cleanup used to
    return having killed nothing at all.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # SIGKILL immediately. Waiting five seconds after SIGTERM made the declared
    # trial deadline a suggestion, and the leader's exit says nothing about its
    # children. The unit's control-group kill is the backstop for descendants
    # that call setsid() and leave this process group.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # Reap the leader after the group is dead. This is intentionally after both
    # signals: a leader exit is not evidence that its descendants are gone.
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            proc.wait()
        except (ProcessLookupError, PermissionError):
            pass


def run_trial(argv, cwd, seed, timeout, deadline=None):
    started = time.monotonic()
    env = dict(os.environ, TRIAL_SEED=str(seed))
    # start_new_session puts the child in its own process group so a timeout
    # kills the whole tree; killing only the direct child leaves descendants
    # running and racing the next iteration.
    proc, out = None, ""
    # Output goes to a file rather than a pipe. Draining a pipe means reading to
    # EOF, and EOF only arrives when the process exits, so a silent hung subject
    # is never timed out: the read blocks past the deadline it was supposed to
    # enforce. A file lets wait() actually be the thing that times out, and caps
    # memory at what we choose to read back.
    with tempfile.TemporaryFile() as sink:
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd or None, env=env, start_new_session=True,
                stdout=sink, stderr=subprocess.STDOUT,
            )
            # start_new_session makes the child its own group leader, so its
            # pid is the group id — captured now, because wait() will reap it.
            wait_timeout = (max(0.0, deadline - time.monotonic())
                            if deadline is not None else max(0.0, timeout))
            exit_code = proc.wait(timeout=wait_timeout)
            status = "ok"
        except subprocess.TimeoutExpired:
            status, exit_code = "timeout", None
        except (FileNotFoundError, PermissionError, OSError) as exc:
            status, exit_code, out = "error", None, str(exc)
        finally:
            # Always, not only on timeout. A leader that exits normally says
            # nothing about what it spawned, and a trial that leaves background
            # processes behind has them race every later iteration.
            if proc:
                _kill_tree(proc, proc.pid)
        if status != "error":
            sink.seek(0)
            out = sink.read(MAX_OUTPUT_BYTES).decode(errors="replace")

    return {
        "status": status,
        "exit_code": exit_code,
        # Only an "ok" run says anything about the subject. A timeout or a
        # missing executable is evidence about the harness, and the evaluator
        # must not fold it into the subject's failure rate.
        "passed": status == "ok" and exit_code == 0,
        "duration_s": round(time.monotonic() - started, 3),
        "failures": sorted(set(PYTEST_FAILURE.findall(out)))[:20],
        "tail": out.strip().splitlines()[-3:] if not (status == "ok" and exit_code == 0) else [],
    }


def publish(artifact_path, result):
    """Write the artifact atomically.

    A direct write leaves a truncated file when the process dies mid-write, and
    a reader that trusts the filename then adopts a half-written observation or
    crashes on every retry. Write, flush, fsync, then rename.
    """
    tmp = artifact_path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(result, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, artifact_path)
    dirfd = os.open(os.path.dirname(artifact_path), os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def main():
    spec = json.loads(sys.argv[1])
    seed, timeout, artifact_path = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    # The optional absolute deadline lets the driver charge collector startup
    # before the trial wait begins. The four-argument form remains compatible
    # with direct collector callers and fixtures.
    deadline = float(sys.argv[5]) if len(sys.argv) == 6 else None
    result = run_trial(spec["argv"], spec.get("cwd"), seed, timeout, deadline)
    # Identity, so a reader can prove this artifact is the trial it expected
    # rather than a stale file that happens to share a name.
    result.update(run_id=spec["run_id"], trial_index=spec["trial_index"],
                  arm=spec.get("arm"), seed=seed, attempt=spec.get("attempt"))
    publish(artifact_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
