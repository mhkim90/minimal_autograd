"""Evaluator: the only thing that decides whether the run stops.

The stopping rule is fixed in the charter before the first trial and computed
here from recorded trials. It is never a judgment made after seeing a value.
"""
import hashlib
import math


def arm_for(trial_index, arms):
    return arms[trial_index % len(arms)] if arms else None


def seed_for(run_id, arm, arm_trial_index):
    """Give every arm its own seed stream.

    Deriving the seed from the global index while rotating arms by that same
    index confounds the two: with two arms one gets every even seed and the
    other every odd one, so a subject that behaves differently on odd inputs
    manufactures an arm difference with no adversary involved. Seeding from
    (run, arm, position within that arm) keeps the schedule fully determined by
    the index while making the streams independent.
    """
    key = f"{run_id}:{arm}:{arm_trial_index}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def is_observation(trial):
    """Only a completed run says anything about the subject.

    A timeout or a missing executable is evidence about the harness. Counting
    it as a subject failure inflates the very rate the run exists to measure.
    """
    return trial.get("status") == "ok"


def summarize(trials, z=1.96):
    """Wilson score interval over observations, with harness errors set aside."""
    observations = [t for t in trials if is_observation(t)]
    discarded = len(trials) - len(observations)
    n = len(observations)
    if n == 0:
        return {"n": 0, "failures": 0, "discarded": discarded, "rate": None,
                "low": 0.0, "high": 1.0, "width": 1.0}
    failures = sum(1 for t in observations if not t["passed"])
    p = failures / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low, high = max(0.0, centre - margin), min(1.0, centre + margin)
    return {
        "n": n,
        "failures": failures,
        "discarded": discarded,
        "rate": round(p, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        # Rounded for display; the unrounded value decides the threshold, or a
        # true width just above it would stop the run early.
        "width": round(high - low, 4),
        "_width_exact": high - low,
    }


def by_arm(trials):
    arms = {}
    for t in trials:
        arms.setdefault(t.get("arm") or "_", []).append(t)
    return {name: summarize(ts) for name, ts in sorted(arms.items())}


def flaky_tests(trials):
    """Test ids that failed at least once. In a suite expected to be green,
    any entry here is the finding the run exists to produce."""
    seen = {}
    for t in trials:
        if not is_observation(t):
            continue
        for f in t.get("failures", []):
            seen[f] = seen.get(f, 0) + 1
    return dict(sorted(seen.items(), key=lambda kv: -kv[1]))


def should_stop(trials, stopping_rule, arms=None):
    """Return (stop, reason, summary). Every declared arm must satisfy the rule."""
    overall = summarize(trials)
    per_arm = by_arm(trials)
    summary = {"overall": overall, "arms": per_arm, "flaky": flaky_tests(trials)}

    if len(trials) >= stopping_rule["max_trials"]:
        return True, "max_trials reached", summary

    if arms:
        unexpected = set(per_arm) - set(arms)
        if unexpected:
            # Comparing only the count would let a trial recorded under the
            # wrong arm satisfy the rule for an arm that was never observed.
            return True, f"unexpected arm in ledger: {sorted(unexpected)}", summary
        if not set(arms) <= set(per_arm):
            return False, None, summary

    ready = per_arm and all(
        s["n"] >= stopping_rule["min_trials_per_arm"]
        and s["_width_exact"] <= stopping_rule["ci_width"]
        for s in per_arm.values()
    )
    if ready:
        return True, "ci_width satisfied on every arm", summary
    return False, None, summary
