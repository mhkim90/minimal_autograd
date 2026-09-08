"""Append-only ledger and the identity-bound continuation lock.

Approval creates the ledger. Continuation opens that existing inode, locks the
same file descriptor, and checks it against the external anchor before doing
anything in the run directory. A ledger/anchor mismatch is unrecoverable: the
reader must never repair it by creating a replacement file.
"""
import fcntl
import json
import os
import sys
from contextlib import contextmanager


class Corrupt(Exception):
    """A complete line or history invariant is corrupt."""


class IdentityCorrupt(Corrupt):
    """The continuation path no longer names the approved ledger inode."""


class ContinuationUnavailable(IdentityCorrupt):
    """The approved ledger or its external anchor is absent."""


KNOWN_EVENTS = frozenset({
    "run_start", "iteration_start", "trial", "discarded", "explored",
    "iteration_end", "infra_retry", "notify", "stop",
})

COMMON_REQUIRED = ("at", "mono", "boot")
EVENT_REQUIRED = {
    "iteration_start": ("trial_index", "attempt"),
    "trial": ("trial_index", "attempt", "result"),
    "discarded": ("trial_index", "attempt", "result"),
    "stop": ("reason", "summary"),
    "infra_retry": ("trial_index", "attempt"),
    "notify": ("delivered",),
}


def validate_record(record, where=""):
    """Raise Corrupt if a record is not a well-formed, known event."""
    if not isinstance(record, dict) or not isinstance(record.get("event"), str):
        raise Corrupt(f"{where}: record has no event type")
    if record["event"] not in KNOWN_EVENTS:
        raise Corrupt(f"{where}: unknown event {record['event']!r}")
    missing = [f for f in COMMON_REQUIRED + EVENT_REQUIRED.get(record["event"], ())
               if f not in record]
    if missing:
        raise Corrupt(f"{where}: {record['event']} record is missing {missing}")
    if record["event"] in ("iteration_start", "trial", "discarded", "infra_retry"):
        attempt = record.get("attempt")
        if not isinstance(attempt, str) or not attempt:
            raise Corrupt(f"{where}: {record['event']}.attempt must be a non-empty string")
    if record["event"] == "notify" and not isinstance(record["delivered"], bool):
        raise Corrupt(f"{where}: notify.delivered must be a boolean")


def _validate_history(records, where):
    starts = set()
    resolutions = {}
    for record in records:
        event = record["event"]
        if event == "iteration_start":
            pair = (record["trial_index"], record["attempt"])
            if pair in starts:
                raise Corrupt(f"{where}: duplicate iteration_start pair {pair!r}")
            starts.add(pair)
        elif event in ("trial", "discarded"):
            pair = (record["trial_index"], record["attempt"])
            previous = resolutions.get(pair)
            if previous is not None:
                raise Corrupt(f"{where}: duplicate resolution pair {pair!r} "
                              f"({previous}/{event})")
            resolutions[pair] = event


def _parse(data, where):
    records = []
    for lineno, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.endswith("\n"):
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Corrupt(f"{where}:{lineno}: {exc}") from None
        validate_record(record, f"{where}:{lineno}")
        records.append(record)
    _validate_history(records, where)
    return records


def _read_fd_bytes(fd):
    position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(fd, position, os.SEEK_SET)
    return b"".join(chunks)


def read_fd(fd, where="<ledger-fd>"):
    """Read complete records from the already-open ledger descriptor."""
    return _parse(_read_fd_bytes(fd).decode(), where)


def read(path):
    """Return complete records; an unterminated final line is absent."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        return _parse(fh.read().decode(), path)


def _heal_fd(fd):
    data = _read_fd_bytes(fd)
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    os.ftruncate(fd, keep)
    os.fsync(fd)
    return len(data) - keep


def _write_all(fd, data):
    """Write every byte or raise without acknowledging the record."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("write returned no progress")
        remaining = remaining[written:]


def _sync_directory(path):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_fd(fd, path, record):
    """Append and fsync using the caller's locked, O_APPEND descriptor."""
    validate_record(record, path)
    dropped = _heal_fd(fd)
    if dropped:
        record = dict(record, healed_bytes=dropped)
    records = read_fd(fd, path)
    if records and not isinstance(records[-1].get("seq"), int):
        raise Corrupt(f"{path}: last record has a non-integer seq")
    record = dict(record, seq=(records[-1]["seq"] + 1 if records else 1))
    _write_all(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
    os.fsync(fd)
    _sync_directory(path)
    return record


def append(path, record):
    """Compatibility helper for approved test fixtures; never creates a ledger."""
    validate_record(record, path)
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    try:
        return append_fd(fd, path, record)
    finally:
        os.close(fd)


def ensure_durable(path):
    """Re-sync an existing ledger; never creates or repairs its identity."""
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    try:
        _heal_fd(fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync_directory(path)


def _anchor(anchor_path):
    try:
        with open(anchor_path) as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationUnavailable(
            f"anchor is missing or unreadable: {anchor_path}: {exc}") from None
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
        raise IdentityCorrupt(f"anchor is malformed: {anchor_path}")
    identity = value.get("ledger")
    if not isinstance(identity, dict) or not all(
            isinstance(identity.get(k), int) for k in ("st_dev", "st_ino")):
        raise IdentityCorrupt(f"anchor has no ledger identity: {anchor_path}")
    return value


def validate_identity(fd, path, anchor_path, run_id=None):
    """Check held FD, current path, and anchor all identify one inode."""
    anchor = _anchor(anchor_path)
    if run_id is not None and anchor["run_id"] != run_id:
        raise IdentityCorrupt(
            f"anchor run_id mismatch: {anchor['run_id']!r} != {run_id!r}")
    try:
        held = os.fstat(fd)
        current = os.stat(path)
    except OSError as exc:
        raise IdentityCorrupt(
            f"ledger path identity is unavailable: {path}: {exc}") from None
    held_id = (held.st_dev, held.st_ino)
    current_id = (current.st_dev, current.st_ino)
    anchor_id = (anchor["ledger"]["st_dev"], anchor["ledger"]["st_ino"])
    if held_id != anchor_id or current_id != anchor_id or held_id != current_id:
        raise IdentityCorrupt(
            f"ledger identity mismatch: held={held_id}, current={current_id}, "
            f"anchor={anchor_id}")
    return anchor


@contextmanager
def locked(path, anchor_path, run_id=None):
    """Open an existing ledger, lock it, validate, and validate before unlock."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_APPEND)
    except OSError as exc:
        raise ContinuationUnavailable(
            f"ledger is missing or cannot be opened without initialization: "
            f"{path}: {exc}") from None
    locked_fd = False
    pending = None
    cleanup_error = None
    try:
        # The open descriptor is the candidate ledger. Only its authoritative
        # post-lock comparison may approve continuation.
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked_fd = True
        validate_identity(fd, path, anchor_path, run_id)
        try:
            yield fd
        except BaseException:
            pending = sys.exc_info()
    except BaseException:
        if pending is None:
            pending = sys.exc_info()
    finally:
        if locked_fd:
            try:
                # This is the final pre-unlock check. It must not be replaced
                # by the post-unlock check below: both windows are meaningful.
                validate_identity(fd, path, anchor_path, run_id)
            except BaseException:
                cleanup_error = sys.exc_info()
        if locked_fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BaseException:
                if cleanup_error is None and pending is None:
                    cleanup_error = sys.exc_info()
            else:
                try:
                    # The unlock hook is intentionally followed by one last
                    # identity check, so a replacement in that window is
                    # corruption rather than a successful continuation.
                    validate_identity(fd, path, anchor_path, run_id)
                except BaseException:
                    if cleanup_error is None:
                        cleanup_error = sys.exc_info()
        try:
            os.close(fd)
        except BaseException:
            if cleanup_error is None and pending is None:
                cleanup_error = sys.exc_info()
    if (cleanup_error is not None
            and isinstance(cleanup_error[1], IdentityCorrupt)):
        raise cleanup_error[1].with_traceback(cleanup_error[2])
    if pending is not None:
        raise pending[1].with_traceback(pending[2])
    if cleanup_error is not None:
        raise cleanup_error[1].with_traceback(cleanup_error[2])
