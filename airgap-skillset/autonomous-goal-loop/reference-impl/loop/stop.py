"""Append the durable owner-stop record while holding the ledger FD lock."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger
import preflight


def boot_id():
    with open("/proc/sys/kernel/random/boot_id") as fh:
        return fh.read().strip()


def now():
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mono": round(time.monotonic(), 3), "boot": boot_id()}


def refuse_binding(run_dir, reason):
    """Durably refuse without appending an owner-stop record or notifying."""
    refusal = f"source binding refused: {reason}"
    path = os.path.join(run_dir, "STOP")
    with open(path, "w") as fh:
        fh.write(refusal + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    parent = os.path.dirname(os.path.abspath(path))
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"STOP: {refusal}", file=sys.stderr)
    raise SystemExit(2)


def main():
    run_dir = sys.argv[1]
    path = os.path.join(run_dir, "iterations.jsonl")
    try:
        charter = json.load(open(os.path.join(run_dir, "charter.json")))
        run_id = charter["run_id"]
        anchor = preflight.anchor_path(run_dir, run_id)
        with ledger.locked(path, anchor, run_id) as fd:
            source_refusal = preflight.check_source_binding(charter, run_dir)
            if source_refusal:
                refuse_binding(run_dir, source_refusal)
            records = ledger.read_fd(fd, path)
            if any(record["event"] == "stop" for record in records):
                os.fsync(fd)
                ledger._sync_directory(path)
                ledger.validate_identity(fd, path, anchor, run_id)
                return
            ledger.validate_identity(fd, path, anchor, run_id)
            ledger.append_fd(fd, path, {"event": "stop", "reason": "owner stop",
                                        "summary": {}, **now()})
            ledger.validate_identity(fd, path, anchor, run_id)
        print("owner stop recorded")
    except ledger.IdentityCorrupt as exc:
        print(f"STOP: ledger identity corruption, human required: {exc}",
              file=sys.stderr)
        raise SystemExit(2)
    except ledger.Corrupt as exc:
        print(f"STOP: ledger corrupt, human required: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (KeyError, json.JSONDecodeError, OSError) as exc:
        print(f"STOP: continuation unavailable, human required: {exc}",
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
