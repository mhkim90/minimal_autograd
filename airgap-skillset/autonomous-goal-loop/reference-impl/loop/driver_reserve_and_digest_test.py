"""Red gates for two Terra gate-(b) blockers the bundle tests did not reach.

Both defects are invisible to the existing suite for the same reason: every
fixture supplies the field or mutates the file that the check happens to look
at, so the check is never asked the question it exists to answer.

B-5. `driver_timeout_s` was optional and defaulted to `max_iteration_duration_s`.
A charter that omits it and sets the unit's `TimeoutStartSec` to the in-process
hard deadline passes every check while leaving systemd's `TimeoutStartFailureMode
=kill` firing at the same instant the in-process deadline path is supposed to
kill the collector and record the overrun.  The cgroup backstop still kills the
process tree -- `evidence/bundle2/cgroup_kill_receipt.txt` proves that much --
but the ordering the charter claims is not the ordering that runs.

B-4. `path_digest` walked with `followlinks=False` and hashed only `files`, so a
directory never contributed to the digest.  Creating or deleting an empty
directory, or adding or repointing a directory symlink, left `subject_frozen`
asserting a freeze the digest cannot see.
"""
import json
import os

import preflight
from bundle_b4_b6_test import _unattended


def _unattended_charter(tmp):
    """The approved unattended charter as a dict, from the run it created."""
    run_dir = _unattended(tmp)
    return json.load(open(os.path.join(run_dir, "charter.json")))


def _set_unit_start(charter, seconds):
    """Point the unit's start timeout at `seconds`, leaving the rest alone."""
    unit = charter["containment"]["unit"]
    lines = [line for line in open(unit).read().splitlines()
             if not line.startswith("TimeoutStartSec=")]
    lines.append(f"TimeoutStartSec={seconds}")
    open(unit, "w").write("\n".join(lines) + "\n")


def test_omitted_driver_timeout_is_refused_by_name(tmp_path):
    """The bypass itself: omit the reserve, align the unit, and pass today.

    This is the shape a mistaken charter actually takes.  It is not enough that
    some reason comes back -- an unrelated `TimeoutStartSec` mismatch would do
    that while leaving the reserve unenforced -- so the reason must name the
    missing field.
    """
    charter = _unattended_charter(str(tmp_path))
    del charter["budgets"]["driver_timeout_s"]
    _set_unit_start(charter, charter["budgets"]["max_iteration_duration_s"])

    reason = preflight._resource_contract(charter)

    assert reason, "a charter with no driver reserve was accepted"
    assert "driver_timeout_s" in reason, reason


def test_driver_timeout_mismatch_names_the_driver_field(tmp_path):
    """The old message blamed `max_iteration_duration_s` for a driver mismatch.

    A reason that names the wrong field sends the owner to edit a budget that is
    already correct, so the message is part of the contract, not cosmetics.
    """
    charter = _unattended_charter(str(tmp_path))
    _set_unit_start(charter, charter["budgets"]["driver_timeout_s"] + 5)

    reason = preflight._resource_contract(charter)

    assert reason, "a unit whose start timeout ignores the driver reserve passed"
    assert "driver_timeout_s" in reason, reason


def test_driver_timeout_must_exceed_the_in_process_deadline(tmp_path):
    """Equal deadlines are the race, so equality must be refused explicitly."""
    charter = _unattended_charter(str(tmp_path))
    charter["budgets"]["driver_timeout_s"] = \
        charter["budgets"]["max_iteration_duration_s"]
    _set_unit_start(charter, charter["budgets"]["driver_timeout_s"])

    reason = preflight._resource_contract(charter)

    assert reason, "a driver deadline equal to the collector deadline passed"
    assert "strictly greater" in reason, reason


def test_subject_digest_covers_an_empty_directory(tmp_path):
    """An empty directory has no files, so a file-only walk cannot see it."""
    subject = tmp_path / "subject"
    (subject / "keep").mkdir(parents=True)
    (subject / "keep" / "a.txt").write_text("a\n")
    before = preflight.path_digest(str(subject))

    (subject / "added").mkdir()

    assert preflight.path_digest(str(subject)) != before, \
        "adding an empty directory left the subject digest unchanged"


def test_subject_digest_covers_a_directory_symlink(tmp_path):
    """`followlinks=False` skips a directory symlink instead of recording it.

    The link is not followed, which is correct, but it was also not hashed, so
    swinging it at a different tree was invisible to the freeze check.
    """
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "a.txt").write_text("a\n")
    outside_one = tmp_path / "one"
    outside_one.mkdir()
    (outside_one / "x.txt").write_text("x\n")
    outside_two = tmp_path / "two"
    outside_two.mkdir()
    (outside_two / "y.txt").write_text("y\n")

    link = subject / "link"
    link.symlink_to(outside_one)
    before = preflight.path_digest(str(subject))

    link.unlink()
    link.symlink_to(outside_two)

    assert preflight.path_digest(str(subject)) != before, \
        "repointing a directory symlink left the subject digest unchanged"


def test_subject_digest_is_stable_across_identical_trees(tmp_path):
    """The freeze check is worthless if an unchanged subject drifts.

    Directory coverage must come from names, not from inodes or walk order.
    """
    def build(root):
        os.makedirs(os.path.join(root, "b", "c"))
        os.makedirs(os.path.join(root, "a"))
        open(os.path.join(root, "b", "c", "f.txt"), "w").write("f\n")
        os.symlink("b", os.path.join(root, "link"))

    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    build(str(one))
    build(str(two))

    assert preflight.path_digest(str(one)) == preflight.path_digest(str(two))
