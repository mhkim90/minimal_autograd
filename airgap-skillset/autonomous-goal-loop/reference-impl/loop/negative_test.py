"""Containment negative test.

Every probe here MUST fail. Run it through the same invocation path the driver
uses; a unit file that reads correctly is not evidence that it is loaded and
effective. Exits nonzero if any probe succeeded, which means the run is not
contained and must not be declared unattended.
"""
import json
import os
import socket
import subprocess
import sys

HOME_PROBE = "write outside permitted paths (home)"
REPO_PROBE = "write to repo outside run dir"
PRIVILEGE_PROBE = "privilege acquisition (user namespace UID 0)"
FILESYSTEM_PROBES = {HOME_PROBE, REPO_PROBE}


def _probe_write(path):
    """Create-exclusive, then always clean up.

    Opening with "w" destroys whatever sits at the probe path to answer a
    question about permissions. Reporting the outcome from a failed *cleanup*
    is worse: the escape succeeded and the probe calls it blocked.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"escaped\n")
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass          # the write already succeeded; that is the finding


def write_outside(target=None):
    _probe_write(target)


def write_repo_outside_run(target=None):
    _probe_write(target)


def network(target=None):
    s = socket.create_connection(("1.1.1.1", 53), timeout=4)
    s.close()


class InconclusiveProbe(Exception):
    pass


def uid0_control():
    """Attempt real user-namespace root and classify only its direct result."""
    try:
        result = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--", "id", "-u"],
            capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "inconclusive"
    if result.returncode == 0 and result.stdout.strip() == "0":
        return "succeeded"
    stderr = result.stderr.lower()
    if (result.returncode != 0 and "unshare" in stderr and any(
            marker in stderr
            for marker in ("operation not permitted", "permission denied", "eacces"))):
        return "blocked"
    return "inconclusive"


def escalate(target=None):
    """Attempt actual namespace UID 0 acquisition, not setuid-bit mutation."""
    outcome = uid0_control()
    if outcome == "succeeded":
        return
    if outcome == "blocked":
        raise PermissionError("user namespace creation was denied")
    raise InconclusiveProbe("user namespace UID-0 control was inconclusive")


# The canonical set. check_containment_evidence requires exactly these names, so
# a charter cannot substitute one easy invented probe for the real four.
PROBES = {
    HOME_PROBE: write_outside,
    REPO_PROBE: write_repo_outside_run,
    "undeclared network connection": network,
    PRIVILEGE_PROBE: escalate,
}


# Errnos that mean a boundary refused the action. Anything else that raises is
# not evidence of containment — a stale probe file or an unrelated failure must
# not be able to impersonate a blocked probe.
import errno as _errno
_DENIED = {_errno.EACCES, _errno.EPERM, _errno.EROFS, _errno.ENETUNREACH,
           _errno.ENETDOWN, _errno.EHOSTUNREACH}


def run_probes(targets=None):
    """Run every probe here, now, and return {name: 'blocked'|'succeeded'|'inconclusive'}.

    Importable so preflight can run the contained side itself rather than trust a
    recorded outcome. A probe that returns escaped; one denied by the boundary is
    blocked; anything else is inconclusive, never counted as containment.
    """
    targets = targets or {}
    out = {}
    for name, fn in PROBES.items():
        try:
            fn(targets.get(name))
            out[name] = "succeeded"
        except PermissionError:
            out[name] = "blocked"
        except OSError as exc:
            out[name] = "blocked" if exc.errno in _DENIED else "inconclusive"
        except Exception:
            out[name] = "inconclusive"
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--charter", required=True)
    parser.add_argument("--mode", choices=("control", "contained"), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with open(args.charter) as fh:
        charter = json.load(fh)
    import preflight
    reason, targets = preflight.canonical_probe_targets(charter["containment"])
    if reason:
        parser.error(reason)
    outcomes = run_probes(targets)
    payload = {"mode": args.mode, "targets": targets, "probes": outcomes}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for name, outcome in outcomes.items():
            print(f"{outcome:>12}  {name:<48} target={targets.get(name)}")
    expected = "succeeded" if args.mode == "control" else "blocked"
    sys.exit(0 if all(value == expected for value in outcomes.values()) else 1)


if __name__ == "__main__":
    main()
