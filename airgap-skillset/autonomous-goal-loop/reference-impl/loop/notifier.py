"""Notifier: delivers the terminal record over the charter's channel.

An unattended run that stopped at three in the morning and told no one has
wasted the night it was given. Failure to notify is therefore not silent — it is
recorded and reported, and the run stays in a stopped-but-unresolved state
rather than exiting quietly.

The charter supplies `notify.command` as an argv list. The stop reason and
summary arrive on stdin as JSON and as the final argument, so both line-oriented
tools and JSON consumers work without a wrapper.
"""
import json
import subprocess

TIMEOUT_S = 30


def notify(charter, reason, summary):
    """Return a record describing the delivery attempt. Never raises."""
    channel = charter.get("notify", {})
    argv = channel.get("command")
    if not argv:
        return {"delivered": False, "channel": None,
                "error": "charter declares no notify.command"}

    payload = json.dumps({"run_id": charter.get("run_id"), "reason": reason,
                          "summary": summary}, sort_keys=True)
    try:
        proc = subprocess.run(
            list(argv) + [payload],
            input=payload.encode(),
            capture_output=True,
            timeout=TIMEOUT_S,
        )
    except Exception as exc:
        return {"delivered": False, "channel": argv[0],
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}

    if proc.returncode != 0:
        return {"delivered": False, "channel": argv[0],
                "error": f"exit {proc.returncode}: {proc.stderr.decode()[:120]}"}
    return {"delivered": True, "channel": argv[0]}
