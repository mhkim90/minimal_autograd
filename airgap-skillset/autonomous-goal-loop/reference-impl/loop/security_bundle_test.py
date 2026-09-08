"""Focused B-1..B-3 tests, written before the security bundle fixes."""
import json
import os
import subprocess
import sys

import negative_test
import preflight
import testkit
from bundle_b4_b6_test import _unattended


HERE = os.path.dirname(os.path.abspath(__file__))


def _run(tmp, **overrides):
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    overrides.setdefault("execution_mode", "supervised")
    charter = testkit.charter(tmp, "security", **overrides)
    testkit.write_charter(run_dir, charter)
    return run_dir, charter


def test_b1_hostile_inherited_env_cannot_authorize_default(tmp_path):
    # The default path is unattended; a supervised-approved charter must not be
    # authorized by the inherited environment variable.
    run_dir, _ = _run(str(tmp_path), execution_mode="supervised")
    env = dict(os.environ, GOAL_LOOP_SUPERVISED="1")
    result = subprocess.run([os.path.join(HERE, "run.sh"), run_dir],
                            capture_output=True, text=True, env=env)
    ok, detail = testkit.stopped_because(run_dir, "unattended",
                                         result.stdout + result.stderr,
                                         result.returncode)
    assert ok, detail
    assert testkit.stop_reason(run_dir) and "unattended" in testkit.stop_reason(run_dir)


def test_b1_systemd_rejects_explicit_supervised_mode(tmp_path):
    run_dir, _ = _run(str(tmp_path))
    env = dict(os.environ, INVOCATION_ID="systemd-test")
    result = subprocess.run([os.path.join(HERE, "run-supervised.sh"), run_dir],
                            capture_output=True, text=True, env=env)
    ok, detail = testkit.stopped_because(run_dir, "systemd", result.stdout + result.stderr,
                                         result.returncode)
    assert ok, detail


def test_b1_explicit_supervised_test_path_still_runs(tmp_path):
    run_dir, _ = _run(str(tmp_path))
    result = subprocess.run([os.path.join(HERE, "run-supervised.sh"), run_dir],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(r["event"] == "trial" for r in
               __import__("ledger").read(os.path.join(run_dir, "iterations.jsonl")))


def test_b1_supervised_entry_rejects_unattended_approved_charter(tmp_path):
    run_dir = _unattended(str(tmp_path))
    result = subprocess.run([os.path.join(HERE, "run-supervised.sh"), run_dir],
                            capture_output=True, text=True)
    ok, detail = testkit.stopped_because(run_dir, "execution mode", result.stdout + result.stderr,
                                         result.returncode)
    assert ok, detail
    assert testkit.stop_reason(run_dir) and "execution mode" in testkit.stop_reason(run_dir)


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(result=None, error=None):
    def run(*args, **kwargs):
        if error:
            raise error
        return result
    return run


def test_b2_uid0_success_is_a_succeeded_control(monkeypatch):
    monkeypatch.setattr(negative_test.subprocess, "run",
                        _fake_run(_Result(stdout="0\n")))
    assert negative_test.uid0_control() == "succeeded"


def test_b2_permission_denial_is_blocked(monkeypatch):
    monkeypatch.setattr(negative_test.subprocess, "run",
                        _fake_run(_Result(1, stderr="unshare: unshare failed: Operation not permitted")))
    assert negative_test.run_probes({})[negative_test.PRIVILEGE_PROBE] == "blocked"


def test_b2_ambiguous_nonzero_is_inconclusive(monkeypatch):
    monkeypatch.setattr(negative_test.subprocess, "run",
                        _fake_run(_Result(1, stderr="invalid argument")))
    assert negative_test.uid0_control() == "inconclusive"


def test_b2_real_uncontained_uid0_control_is_reported():
    assert negative_test.uid0_control() == "succeeded"


def _targets(tmp):
    return {
        negative_test.HOME_PROBE: os.path.realpath(os.path.join(tmp, "home-target")),
        negative_test.REPO_PROBE: os.path.realpath(os.path.join(tmp, "repo-target")),
    }


def test_b3_filesystem_probes_use_declared_targets_not_environment(tmp_path, monkeypatch):
    targets = _targets(str(tmp_path))
    seen = {}

    def capture(target=None):
        seen[len(seen)] = target

    monkeypatch.setattr(negative_test, "PROBES", {
        negative_test.HOME_PROBE: capture,
        negative_test.REPO_PROBE: capture,
    })
    monkeypatch.setenv("HOME", "/hostile-home")
    monkeypatch.setenv("GOAL_LOOP_REPO", "/hostile-repo")
    negative_test.run_probes(targets)
    assert set(seen.values()) == set(targets.values())


def test_b3_target_inside_writable_path_is_rejected(tmp_path):
    writable = os.path.realpath(os.path.join(str(tmp_path), "writable"))
    os.makedirs(writable)
    targets = _targets(str(tmp_path))
    targets[negative_test.HOME_PROBE] = os.path.join(writable, "escape")
    reason = preflight.validate_probe_targets({
        "writable_paths": [writable], "probe_targets": targets})
    assert reason and "writable" in reason


def test_b3_symlink_resolving_inside_writable_path_is_rejected(tmp_path):
    writable = os.path.realpath(os.path.join(str(tmp_path), "writable"))
    os.makedirs(writable)
    link = os.path.join(str(tmp_path), "link")
    os.symlink(writable, link)
    targets = _targets(str(tmp_path))
    targets[negative_test.HOME_PROBE] = os.path.join(link, "escape")
    reason = preflight.validate_probe_targets({
        "writable_paths": [writable], "probe_targets": targets})
    assert reason and "writable" in reason


def test_b3_evidence_must_record_exact_canonical_targets(tmp_path):
    writable = os.path.realpath(os.path.join(str(tmp_path), "writable"))
    os.makedirs(writable)
    targets = _targets(str(tmp_path))
    unit = os.path.join(str(tmp_path), "unit")
    open(unit, "w").write(
        "[Service]\nRestrictNamespaces=yes\n"
        f"ExecStart={os.path.join(HERE, 'run.sh')} {str(tmp_path)}\n"
        "Environment=GOAL_LOOP_SUPERVISED=\n"
        f"ReadWritePaths={writable}\n")
    source_digest = preflight.file_digest(os.path.join(HERE, "negative_test.py"))
    receipt_path = os.path.realpath(os.path.join(str(tmp_path), "receipt.json"))
    with open(receipt_path, "w") as fh:
        json.dump({
            "schema": preflight.CONTAINMENT_RECEIPT_SCHEMA,
            "at": "2026-08-30T00:00:00+0900",
            "probe_source_sha256": source_digest,
            "targets": {**targets, negative_test.HOME_PROBE: "/wrong"},
            "probes": {name: {"control": "succeeded", "contained": "blocked"}
                       for name in negative_test.PROBES},
        }, fh)
    containment = {
        "required": True, "writable_paths": [writable], "probe_targets": targets,
        "unit": unit, "unit_sha256": preflight.file_digest(unit),
        "probe_source": os.path.join(HERE, "negative_test.py"),
        "probe_source_sha256": source_digest,
        "verified_at": "2026-08-30T00:00:00+0900",
        "negative_test": {"receipt_path": receipt_path,
                          "receipt_sha256": preflight.file_digest(receipt_path)},
    }
    reason = preflight.check_containment_evidence(
        {"containment": containment, "arms": {}}, str(tmp_path))
    assert reason and "target" in reason


def test_b3_cli_json_receipt_records_explicit_targets(tmp_path):
    run_dir, charter = _run(str(tmp_path), containment={
        "required": True,
        "writable_paths": [os.path.realpath(os.path.join(str(tmp_path), "writable"))],
        "probe_targets": _targets(str(tmp_path)),
    })
    # The CLI consumes the charter itself; HOME and GOAL_LOOP_REPO are irrelevant.
    result = subprocess.run([
        sys.executable, os.path.join(HERE, "negative_test.py"),
        "--charter", os.path.join(run_dir, "charter.json"),
        "--mode", "control", "--json"], capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    assert receipt["targets"] == _targets(str(tmp_path))


def test_b2_current_unit_has_namespace_restriction_and_unattended_entry():
    # airgap-skillset port: reads the templated .example unit (placeholder
    # paths for an arbitrary deployment), not a concrete deployed unit;
    # the expected ReadWritePaths is the placeholder token the file itself
    # carries, not this dev machine's real state directory. See PORTING.md.
    path = os.path.join(HERE, "goal-loop-p4.service.example")
    unit = open(path).read()
    assert "RestrictNamespaces=yes" in unit
    assert "ExecStart=" in unit and "/loop/run.sh " in unit
    assert "--supervised" not in unit
    assert "Environment=GOAL_LOOP_SUPERVISED=" in unit
    assert preflight.check_unit_policy(
        path, ["<control-plane>/<run-id>"]) is None
