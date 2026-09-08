"""Elapsed time that does not simply trust the wall clock.

Recording a boot id and a monotonic reading and then computing elapsed time
from wall timestamps anyway is worse than not recording them: the run claims a
reboot-safe budget it does not have. Monotonic readings are comparable only
within one boot, so elapsed time is the sum of per-boot monotonic spans plus
the wall gaps between boots.
"""
import time
from datetime import datetime


def parse(ts):
    """Parse an ISO timestamp, keeping its offset. Stripping the timezone makes
    every comparison silently wrong by the UTC offset."""
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")


def elapsed(records, tolerance_s=120):
    """Return (seconds, anomaly). An anomaly means the clock cannot bound the run."""
    usable = [r for r in records if "mono" in r and "boot" in r and "at" in r]
    if not usable:
        return 0.0, None
    if len(usable) == 1:
        # A single record still marks a start. Time since it counts, or a crash
        # right after run_start would let a run resume with a fresh budget: same
        # boot measured by the monotonic clock, another boot by the wall gap.
        one = usable[0]
        if _same_boot(one):
            return max(0.0, time.monotonic() - one["mono"]), None
        downtime = (datetime.now().astimezone() - parse(one["at"])).total_seconds()
        return (max(0.0, downtime), None) if downtime >= 0 else \
            (0.0, "wall clock is behind the last recorded time")

    segments, current = [], [usable[0]]
    for record in usable[1:]:
        if record["boot"] == current[-1]["boot"]:
            current.append(record)
        else:
            segments.append(current)
            current = [record]
    segments.append(current)

    total = 0.0
    for segment in segments:
        mono_span = segment[-1]["mono"] - segment[0]["mono"]
        if mono_span < 0:
            return 0.0, "monotonic clock went backwards within one boot"
        wall_span = (parse(segment[-1]["at"]) - parse(segment[0]["at"])).total_seconds()
        if abs(wall_span - mono_span) > tolerance_s:
            # Inside one boot these must agree. Disagreement means the wall
            # clock was adjusted, and repeated small backward adjustments would
            # otherwise let a run undercount its budget indefinitely.
            return 0.0, (f"wall clock disagrees with monotonic time by "
                         f"{abs(wall_span - mono_span):.0f}s within one boot")
        total += mono_span

    for previous, following in zip(segments, segments[1:]):
        gap = (parse(following[0]["at"]) - parse(previous[-1]["at"])).total_seconds()
        if gap < 0:
            return 0.0, "wall clock went backwards across a reboot"
        total += gap

    # Time since the last record still counts. Within this boot the monotonic
    # clock measures it; across a reboot only the wall clock survives, and
    # adding nothing there let a run come back from hours of downtime with its
    # budget apparently untouched.
    last = usable[-1]
    if _same_boot(last):
        total += max(0.0, time.monotonic() - last["mono"])
    else:
        downtime = (datetime.now().astimezone() - parse(last["at"])).total_seconds()
        if downtime < 0:
            return 0.0, "wall clock is behind the last recorded time"
        total += downtime
    return total, None


def _same_boot(record):
    with open("/proc/sys/kernel/random/boot_id") as fh:
        return record["boot"] == fh.read().strip()
