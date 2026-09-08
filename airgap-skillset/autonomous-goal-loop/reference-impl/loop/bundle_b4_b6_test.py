"""Red/green gates for the B-4..B-6 bundle.

The first check is deliberately about the durable stop record rather than a
matching process string.  It prevents a refusal test from passing because an
unrelated invocation happened to print the requested words.
"""
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time

import pytest
import preflight
import source_manifest
import testkit
import ledger
import iterate
import collector
import negative_test


HERE = os.path.dirname(os.path.abspath(__file__))


def _unattended(tmp, approve=True, **changes):
    sync_resources = changes.pop("unit_resources", False)
    start_failure_mode = changes.pop("unit_start_failure_mode", True)
    # Keep the control plane outside every arm cwd.  The real containment
    # contract refuses a control plane that would be destroyed with a subject.
    run_dir = os.path.join(tmp + "-control", "run")
    os.makedirs(os.path.join(run_dir, "results"))
    subject = os.path.join(tmp, "subject")
    writable = os.path.join(tmp, "state")
    os.makedirs(subject)
    os.makedirs(writable)
    unit = os.path.join(tmp, "loop.service")
    open(unit, "w").write(
        "[Service]\nTimeoutStartSec=32\n"
        + ("TimeoutStartFailureMode=kill\n" if start_failure_mode else "")
        + "TimeoutStopSec=1s\nKillMode=control-group\nSendSIGKILL=yes\n"
        + "RestrictNamespaces=yes\n"
        + f"ExecStart={os.path.join(HERE, 'run.sh')} {str(tmp)}\n"
        + "Environment=GOAL_LOOP_SUPERVISED=\n"
        + f"ReadWritePaths={writable}\n"
        "MemoryMax=512M\nCPUQuota=50%\nTasksMax=64\n")
    manifest, commit = source_manifest.default_paths(run_dir, "bundle")
    arm_path = os.path.join(tmp, "arm.py")
    if "arms" not in changes:
        open(arm_path, "w").write("raise SystemExit(0)\n")
        arms = {"a": {"argv": [sys.executable, arm_path], "cwd": tmp,
                       "source_paths": [os.path.realpath(arm_path)]}}
    else:
        arms = changes["arms"]
        for spec in arms.values():
            if isinstance(spec, dict) and "source_paths" not in spec:
                argv = spec.get("argv", [])
                if "-m" in argv:
                    module = argv[argv.index("-m") + 1]
                    spec["source_paths"] = [os.path.join(spec["cwd"], module + ".py")]
                elif argv and os.path.isfile(argv[-1]):
                    spec["source_paths"] = [argv[-1]]
    charter = {
        "run_id": "bundle", "class": "evidence", "subject": subject,
        "subject_frozen": True, "execution_mode": "unattended",
        "risk_level": "L1", "scope": {"allow": [writable], "deny": [subject]},
        "arms": arms,
        "notify": {"command": ["true"]},
        "containment": {"required": True, "unit": unit,
                        "writable_paths": [writable], "external_effects": []},
        "resources": {"MemoryMax": "512M", "CPUQuota": "50%", "TasksMax": 64},
        "components": {},
         "model": None, "route": {"mode": "headless", "primary": None,
                                    "degrade": "stop"},
         "baseline": {},
        "stopping_rule": {"min_trials_per_arm": 2, "max_trials": 4, "ci_width": 1.0},
        "budgets": {"max_wallclock_s": 600, "max_infra_retries": 3,
                     "max_iteration_duration_s": 30, "max_trial_duration_s": 10,
                     "driver_timeout_s": 32,
                     "termination_duration_s": 1},
        "approved_by": "test", "approved_at": "now",
        "unenforced_risks": [
            {"risk": risk, "reason": reason}
            for risk, reason in preflight.RESIDUAL_REASONS.items()
        ],
    }
    charter.update(changes)
    # A budgets override replaces the whole dict, so it drops the required driver
    # reserve.  Refill it here rather than in each test: without it every one of
    # these charters is refused for a missing reserve before it reaches the check
    # the test is actually named after, which is a refusal for the wrong reason.
    # The reserve is deliberately *not* the unit's start timeout, so a test that
    # means to exercise a TimeoutStartSec mismatch still gets one.
    if isinstance(charter.get("budgets"), dict):
        iteration = charter["budgets"].get("max_iteration_duration_s")
        if isinstance(iteration, (int, float)) and not isinstance(iteration, bool):
            charter["budgets"].setdefault("driver_timeout_s", iteration + 2)
    if ("budgets" in changes
            and changes["budgets"].get("termination_duration_s") not in (None, 1)):
        lines = [line for line in open(unit).read().splitlines()
                 if not line.startswith("TimeoutStartSec=")]
        lines.append(f"TimeoutStartSec={charter['budgets']['driver_timeout_s']}")
        open(unit, "w").write("\n".join(lines) + "\n")
    if isinstance(charter.get("budgets"), dict):
        charter["budgets"].setdefault("termination_duration_s", 1)
    if sync_resources and isinstance(charter.get("resources"), dict):
        values = charter["resources"]
        unit_lines = [line for line in open(unit).read().splitlines()
                      if not any(line.startswith(key + "=")
                                 for key in ("MemoryMax", "CPUQuota", "TasksMax"))]
        unit_lines.extend(f"{key}={values[key]}" for key in
                          ("MemoryMax", "CPUQuota", "TasksMax") if key in values)
        unit_lines = [line for line in unit_lines
                      if not line.startswith("TimeoutStopSec=")]
        unit_lines.append(f"TimeoutStopSec={charter['budgets'].get('termination_duration_s', 1)}s")
        open(unit, "w").write("\n".join(unit_lines) + "\n")
    if isinstance(charter.get("containment"), dict) \
            and charter["containment"].get("required") is True:
        containment = charter["containment"]
        unit_lines = [line for line in open(unit).read().splitlines()
                      if not line.startswith("ReadWritePaths=")]
        unit_lines.extend(f"ReadWritePaths={path}"
                          for path in containment.get("writable_paths", []))
        open(unit, "w").write("\n".join(unit_lines) + "\n")
        probe_source = os.path.realpath(os.path.join(HERE, "negative_test.py"))
        targets = {
            negative_test.HOME_PROBE: os.path.realpath(
                os.path.join(str(tmp), "home-target")),
            negative_test.REPO_PROBE: os.path.realpath(
                os.path.join(str(tmp), "repo-target")),
        }
        stamp = "2026-08-30T00:00:00+0900"
        probe_digest = preflight.file_digest(probe_source)
        containment.setdefault("unit_sha256", preflight.file_digest(unit))
        containment.setdefault("probe_source", probe_source)
        containment.setdefault("probe_source_sha256", probe_digest)
        containment.setdefault("probe_targets", targets)
        containment.setdefault("verified_at", stamp)
        receipt_path = os.path.realpath(os.path.join(str(tmp), "containment-receipt.json"))
        containment.setdefault("negative_test", {"receipt_path": receipt_path})
    if "notify" not in changes:
        charter["notify"]["source_paths"] = [os.path.realpath(shutil.which("true"))]
    elif isinstance(charter.get("notify"), dict) and "source_paths" not in charter["notify"]:
        command = charter["notify"].get("command", [])
        charter["notify"]["source_paths"] = [command[-1]] if command and os.path.isfile(command[-1]) else []
    if "components" not in changes:
        component_paths = {
            "containment": "preflight.py", "lease": "ledger.py",
            "collector": "collector.py", "attempt_id": "iterate.py",
            "evaluator": "evaluator.py", "ledger_writer": "ledger.py",
        }
        for name, filename in component_paths.items():
            path_component = os.path.join(HERE, filename)
            component_hash = preflight.file_digest(path_component)
            charter["components"][name] = {
                "path": path_component, "sha256": component_hash,
            }
    if "measurement" not in changes:
        measurement_path = os.path.realpath(os.path.join(tmp, "measure.py"))
        open(measurement_path, "w").write(
            "import json; print(json.dumps({'value': 0, 'count': 1}))\n")
        charter["measurement"] = {
            "argv": [sys.executable, measurement_path], "cwd": os.path.realpath(tmp),
            "timeout_s": 2, "parser": {"id": preflight.BASELINE_PARSER_ID,
                                          "field": "value", "count_field": "count"},
            "source_paths": [measurement_path],
        }
    charter["baseline"] = {"value": 0, "count": 1}
    if "baseline" in changes:
        charter["baseline"] = changes["baseline"]
    if "source_manifest" not in changes:
        expected, problem = source_manifest.expected_paths(charter, run_dir)
        if problem:
            expected = set(os.path.join(HERE, name)
                           for name in source_manifest.FIXED_NAMES)
            expected.update((os.path.join(run_dir, "charter.json"), unit))
            expected.update(os.path.realpath(spec["path"])
                            for spec in charter["components"].values())
        charter["source_manifest"] = {
            "manifest_path": manifest, "commit_path": commit,
            "files": sorted(expected),
        }
    path = os.path.join(run_dir, "charter.json")
    json.dump(charter, open(path, "w"))
    if charter["containment"].get("required") is True:
        # Limitation: this hermetic fixture executes only the control half and
        # supplies synthetic contained outcomes/provenance for unrelated gates.
        # It is never evidence; the real-receipt test below replaces this path.
        control = subprocess.run(
            [sys.executable, os.path.join(HERE, "negative_test.py"),
             "--charter", path, "--mode", "control", "--json"],
            capture_output=True, text=True, check=True)
        observed = json.loads(control.stdout)
        receipt = {
            "schema": preflight.CONTAINMENT_RECEIPT_SCHEMA, "at": stamp,
            "probe_source_sha256": charter["containment"]["probe_source_sha256"],
            "targets": observed["targets"],
             "probes": {name: {"control": outcome, "contained": "blocked"}
                       for name, outcome in observed["probes"].items()},
             "provenance": {
                 "control": {"cgroup": "fixture-control-cgroup",
                              "invocation_id": None},
                 # The cgroup must end in its own unit, the way the kernel
                 # writes it. A placeholder pair that disagrees would be refused
                 # by the provenance binding check, and every test relying on
                 # this fixture would fail for that reason instead of its own.
                 "contained": {"cgroup": "/fixture.slice/fixture-contained.service",
                                "invocation_id": "fixture-invocation",
                                "unit": "fixture-contained.service"},
             },
             "control_execution": {
                "argv": [sys.executable, os.path.join(HERE, "negative_test.py"),
                         "--charter", path, "--mode", "control", "--json"],
                "return_code": control.returncode,
                "stdout_sha256": __import__("hashlib").sha256(
                    control.stdout.encode()).hexdigest(),
            },
        }
        with open(charter["containment"]["negative_test"]["receipt_path"], "w") as fh:
            json.dump(receipt, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
        # Pin whatever bytes were actually written, not a correct receipt: a
        # malformed-receipt test must still reach the check it is named after
        # rather than being refused for an unrelated digest mismatch.
        declaration = charter["containment"]["negative_test"]
        declaration["receipt_sha256"] = preflight.file_digest(
            declaration["receipt_path"])
        with open(path, "w") as fh:
            json.dump(charter, fh)
    if approve:
        preflight.approve(path)
    return run_dir


def _refusal(tmp, expected, **changes):
    try:
        run_dir = _unattended(tmp, **changes)
    except RuntimeError as exc:
        assert expected.lower() in str(exc).lower(), str(exc)
        return
    result = testkit.fire(run_dir, supervised=False)
    ok, detail = testkit.stopped_because(run_dir, expected,
                                         result.stdout + result.stderr,
                                         result.returncode)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    assert ok, detail
    assert not any(record["event"] == "trial" for record in records)


def test_stopped_because_requires_the_durable_reason(tmp_path):
    # This was a false positive before testkit's output fallback was removed.
    (tmp_path / "iterations.jsonl").write_text(json.dumps({
        "event": "stop", "reason": "different durable reason", "summary": {},
        **testkit.stamp(), "seq": 1,
    }) + "\n")
    testkit.LAST_OUTPUT = "STOP: requested expected reason"
    testkit.LAST_RETURN_CODE = 2
    ok, _ = testkit.stopped_because(str(tmp_path), "expected reason")
    assert not ok


def test_b4_rejects_l3_risk(tmp_path):
    _refusal(str(tmp_path), "risk_level", risk_level="L3")


def test_b4_rejects_absent_scope(tmp_path):
    _refusal(str(tmp_path), "scope", scope=None)


def test_b4_rejects_unfrozen_subject(tmp_path):
    _refusal(str(tmp_path), "frozen", subject_frozen=False)


def test_b4_rejects_writable_subject(tmp_path):
    subject = os.path.join(str(tmp_path), "subject")
    _refusal(str(tmp_path), "writable path",
             containment={"required": True, "unit": os.path.join(str(tmp_path), "loop.service"),
                          "writable_paths": [subject], "external_effects": []})


def test_b4_rejects_unmeasured_baseline(tmp_path):
    _refusal(str(tmp_path), "baseline", baseline={"value": None})


def test_b4_forged_legacy_receipt_cannot_approve_failing_real_measurement(tmp_path):
    """Legacy success JSON must not stand in for an observed command run."""
    run_dir = _unattended(str(tmp_path), approve=False)
    charter_path = os.path.join(run_dir, "charter.json")
    charter = json.load(open(charter_path))
    legacy_path = os.path.realpath(os.path.join(run_dir, "baseline.json"))
    legacy = {"result": "success", "test_id": "legacy",
              "subject": os.path.realpath(charter["subject"]),
              "subject_sha256": preflight.path_digest(charter["subject"]),
              "commit": source_manifest.current_commit(HERE),
              "value": 0, "count": 1}
    json.dump(legacy, open(legacy_path, "w"), sort_keys=True)
    charter["baseline"] = {"value": 0, "count": 1,
                            "commit": legacy["commit"], "measured_at": "legacy",
                            "receipt": {"path": legacy_path,
                                        "sha256": preflight.file_digest(legacy_path)}}
    missing = os.path.join(str(tmp_path), "does-not-exist.py")
    charter["measurement"] = {
        "argv": [sys.executable, missing], "cwd": os.path.realpath(str(tmp_path)),
        "timeout_s": 1, "parser": {"id": "json-field-v1", "field": "value"},
        "source_paths": [os.path.realpath(missing)],
    }
    json.dump(charter, open(charter_path, "w"))

    approval_files = [
        os.path.join(run_dir, "charter.sha256"),
        os.path.join(run_dir, "iterations.jsonl"),
        preflight.anchor_path(run_dir, charter["run_id"]),
        source_manifest.default_paths(run_dir, charter["run_id"])[0],
        source_manifest.default_paths(run_dir, charter["run_id"])[1],
    ]
    with pytest.raises(RuntimeError, match="measurement"):
        preflight.approve(charter_path)
    assert all(not os.path.lexists(path) for path in approval_files)


def _measurement(tmp_path, source):
    path = os.path.realpath(os.path.join(str(tmp_path), "measurement.py"))
    open(path, "w").write(source)
    return {"argv": [sys.executable, path], "cwd": os.path.realpath(str(tmp_path)),
            "timeout_s": 1, "parser": {"id": preflight.BASELINE_PARSER_ID,
                                          "field": "value", "count_field": "count"},
            "source_paths": [path]}


def _approval_paths(run_dir):
    anchor = preflight.anchor_path(run_dir, "bundle")
    manifest, commit = source_manifest.default_paths(run_dir, "bundle")
    paths = [os.path.join(run_dir, "charter.sha256"),
             os.path.join(run_dir, "iterations.jsonl"), anchor, manifest, commit]
    parent = os.path.dirname(anchor)
    prefix = os.path.basename(anchor) + ".tmp-"
    paths.extend(os.path.join(parent, name) for name in os.listdir(parent)
                 if name.startswith(prefix))
    return paths


def _assert_no_approval_owned_durable_state(run_dir):
    assert not [path for path in _approval_paths(run_dir) if os.path.lexists(path)]


def _approval_contract_fixture(tmp_path):
    """Return an unapproved charter with a receipt from a real control run."""
    run_dir = _unattended(str(tmp_path), approve=False)
    charter_path = os.path.join(run_dir, "charter.json")
    return run_dir, charter_path


@pytest.mark.parametrize("required", [False, None, 1],
                         ids=["false", "missing", "truthy-non-bool"])
def test_approval_requires_boolean_true_containment_before_effects(
        tmp_path, monkeypatch, required):
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    if required is None:
        del charter["containment"]["required"]
    else:
        charter["containment"]["required"] = required
    json.dump(charter, open(charter_path, "w"))

    def reached_later_gate(*args, **kwargs):
        raise AssertionError("approval reached measurement before containment gate")

    monkeypatch.setattr(preflight, "_measurement_spec", reached_later_gate)
    monkeypatch.setattr(negative_test, "run_probes",
                        lambda *args: pytest.fail("approval ran a live probe"))
    monkeypatch.setattr(preflight, "read_runtime_binding",
                        lambda *args: pytest.fail("approval read runtime binding"))
    with pytest.raises(RuntimeError, match="containment.required.*True"):
        preflight.approve(charter_path)
    _assert_no_approval_owned_durable_state(run_dir)


@pytest.mark.parametrize("field, expected", [
    ("unit digest", "unit"),
    ("probe digest", "probe_source_sha256"),
    ("probe binding", "canonical"),
    ("verified timestamp", "verified_at"),
    ("probe set", "probe set"),
    ("control outcome", "uncontained"),
    ("contained outcome", "not blocked"),
    ("recorded targets", "targets"),
    ("recorded probe digest", "probe_source_sha256"),
], ids=["unit-digest", "probe-digest", "probe-binding", "verified-at",
        "probe-set", "control-outcome", "contained-outcome", "recorded-targets",
        "recorded-probe-digest"])
def test_approval_rejects_malformed_static_containment_evidence_before_effects(
        tmp_path, monkeypatch, field, expected):
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    receipt_path = charter["containment"]["negative_test"]["receipt_path"]
    receipt = json.load(open(receipt_path))
    if field == "verified timestamp":
        charter["containment"]["verified_at"] = "different"
    elif field == "unit digest":
        charter["containment"]["unit_sha256"] = "0" * 64
    elif field == "probe digest":
        charter["containment"]["probe_source_sha256"] = "0" * 64
    elif field == "probe binding":
        charter["containment"]["probe_source"] = os.path.join(
            str(tmp_path), "not-the-runtime-probe.py")
    elif field == "probe set":
        receipt["probes"] = {"invented probe": {"control": "succeeded",
                                                 "contained": "blocked"}}
    elif field == "control outcome":
        receipt["probes"][preflight.CANONICAL_HOME_PROBE]["control"] = "failed"
    elif field == "contained outcome":
        receipt["probes"][preflight.CANONICAL_HOME_PROBE]["contained"] = "succeeded"
    elif field == "recorded targets":
        receipt["targets"][preflight.CANONICAL_HOME_PROBE] = os.path.join(
            str(tmp_path), "other-target")
    else:
        receipt["probe_source_sha256"] = "0" * 64
    if field not in {"unit digest", "probe digest", "probe binding", "verified timestamp"}:
        with open(receipt_path, "w") as fh:
            json.dump(receipt, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
        # Re-pin the declaration to the bytes this test just wrote. Leaving the
        # old digest would refuse the charter as a swapped receipt, which is a
        # real check but not this test's -- and every case here would go green
        # on a reason none of them are named after.
        charter["containment"]["negative_test"]["receipt_sha256"] = \
            preflight.file_digest(receipt_path)
    json.dump(charter, open(charter_path, "w"))

    def reached_later_gate(*args, **kwargs):
        raise AssertionError("approval reached measurement before containment evidence gate")

    monkeypatch.setattr(preflight, "_measurement_spec", reached_later_gate)
    with pytest.raises(RuntimeError, match=expected):
        preflight.approve(charter_path)
    _assert_no_approval_owned_durable_state(run_dir)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_containment_evidence_requires_regular_unit_file(tmp_path, monkeypatch, kind):
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    unit = charter["containment"]["unit"]
    os.unlink(unit)
    if kind == "directory":
        os.mkdir(unit)
    else:
        os.mkfifo(unit)

    def digest_must_not_open_nonregular(path):
        raise AssertionError("file_digest reached a non-regular containment unit")

    monkeypatch.setattr(preflight, "file_digest", digest_must_not_open_nonregular)
    reason = preflight.check_containment_evidence(charter, run_dir)
    assert reason == "containment unit must be a canonical regular file"


def test_valid_static_contract_is_checked_without_live_probe_or_runtime_binding(
        tmp_path, monkeypatch):
    _, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    checked = []
    original = preflight.check_containment_evidence

    def observe(*args, **kwargs):
        checked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(preflight, "check_containment_evidence", observe)
    monkeypatch.setattr(negative_test, "run_probes",
                        lambda *args: pytest.fail("approval ran a live probe"))
    monkeypatch.setattr(preflight, "read_runtime_binding",
                        lambda *args: pytest.fail("approval read runtime binding"))
    assert preflight.check_approval_contract(charter, os.path.dirname(charter_path)) is None
    assert checked == [True]


def test_arbitrary_digest_matching_probe_file_cannot_bind_runtime_probe(tmp_path,
                                                                         monkeypatch):
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    canonical = charter["containment"]["probe_source"]
    substitute = os.path.join(str(tmp_path), "copied-negative-test.py")
    shutil.copyfile(canonical, substitute)
    charter["containment"]["probe_source"] = os.path.realpath(substitute)
    charter["containment"]["probe_source_sha256"] = preflight.file_digest(substitute)
    charter["containment"]["negative_test"]["probe_source_sha256"] = \
        preflight.file_digest(substitute)
    json.dump(charter, open(charter_path, "w"))

    def reached_later_gate(*args, **kwargs):
        raise AssertionError("approval reached measurement before probe binding gate")

    monkeypatch.setattr(preflight, "_measurement_spec", reached_later_gate)
    with pytest.raises(RuntimeError, match="canonical"):
        preflight.approve(charter_path)
    _assert_no_approval_owned_durable_state(run_dir)


@pytest.mark.parametrize("source, expected", [
    ("raise SystemExit(7)\n", "baseline measurement returned nonzero"),
    ("import time; time.sleep(2)\n", "baseline measurement timed out"),
    ("print('not json')\n", "baseline measurement output is malformed"),
    ("print('{\"value\": 9, \"count\": 2}')\n", "allowed baseline claim"),
])
def test_b4_measurement_failures_leave_no_approval_state(tmp_path, source, expected):
    run_dir = _unattended(str(tmp_path), approve=False,
                          measurement=_measurement(tmp_path, source))
    with pytest.raises(RuntimeError, match=expected):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_missing_registry_entry_leaves_no_approval_state(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path), approve=False)
    missing = preflight.REQUIRED_COMPONENTS[0]
    monkeypatch.delitem(preflight.COMPONENT_EXERCISE_REGISTRY, missing)
    with pytest.raises(RuntimeError, match="registry"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


@pytest.mark.parametrize("changes, expected", [
    ({"risk_level": "L3"}, "risk_level"),
    ({"scope": None}, "scope"),
    ({"subject_frozen": False}, "frozen"),
    ({"subject": "missing-subject"}, "existing frozen path"),
    ({"model": "configured-default"}, "model=null"),
    ({"route": {"primary": "agent=luna"}}, "closed route"),
    ({"resources": {"MemoryMax": "512M"}}, "CPUQuota"),
    ({"unenforced_risks": []}, "canonical residual IDs"),
    ({"source_manifest": None}, "source declaration"),
], ids=["risk", "scope", "unfrozen", "missing-subject", "model",
       "route", "resources", "residual", "source"])
def test_b4_contract_refusal_has_zero_approval_state(tmp_path, changes, expected):
    run_dir = _unattended(str(tmp_path), approve=False, **changes)
    with pytest.raises(RuntimeError, match=expected):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_measurement_source_drift_during_execution_refuses_without_approval_owned_state(
        tmp_path):
    source = "open(__file__, 'a').write('# drift\\n'); print('{\\\"value\\\": 0, \\\"count\\\": 1}')\n"
    run_dir = _unattended(str(tmp_path), approve=False,
                          measurement=_measurement(tmp_path, source))
    with pytest.raises(RuntimeError, match="source, component, or subject digest"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_absolute_subject_child_effect_remains_but_approval_owned_state_is_absent(
        tmp_path):
    subject = os.path.join(str(tmp_path), "subject")
    source_path = os.path.realpath(os.path.join(str(tmp_path), "measurement.py"))
    open(source_path, "w").write(
        "import pathlib,sys; pathlib.Path(sys.argv[1], 'drift').write_text('x'); "
        "print('{\"value\": 0, \"count\": 1}')\n")
    measurement = {"argv": [sys.executable, source_path, subject],
                   "cwd": os.path.realpath(str(tmp_path)), "timeout_s": 1,
                   "parser": {"id": preflight.BASELINE_PARSER_ID, "field": "value",
                              "count_field": "count"}, "source_paths": [source_path]}
    run_dir = _unattended(str(tmp_path), approve=False, measurement=measurement)
    with pytest.raises(RuntimeError, match="source, component, or subject digest"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    assert os.path.exists(os.path.join(subject, "drift"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_component_digest_drift_during_exercise_refuses_without_state(tmp_path,
                                                                          monkeypatch):
    run_dir = _unattended(str(tmp_path), approve=False)
    original = preflight._run_component_exercise
    calls = [0]
    drifted = [False]

    def drift(name, spec, charter):
        result = original(name, spec, charter)
        calls[0] += 1
        if name == preflight.REQUIRED_COMPONENTS[0]:
            drifted[0] = True
        return result

    original_snapshot = preflight._snapshot

    def snapshot(charter, measurement):
        value = original_snapshot(charter, measurement)
        if drifted[0]:
            value["components"][preflight.REQUIRED_COMPONENTS[0]]["sha256"] = "drift"
        return value

    monkeypatch.setattr(preflight, "_run_component_exercise", drift)
    monkeypatch.setattr(preflight, "_snapshot", snapshot)
    with pytest.raises(RuntimeError, match="source, component, or subject digest"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    assert calls == [len(preflight.REQUIRED_COMPONENTS)]
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_positive_anchor_has_exact_observed_execution_records(tmp_path, monkeypatch):
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    observed_baselines = []
    observed_components = []
    original_popen = preflight.subprocess.Popen
    original_run = preflight.subprocess.run

    class ObservedProcess:
        def __init__(self, process):
            self._process = process

        def communicate(self, *args, **kwargs):
            stdout, stderr = self._process.communicate(*args, **kwargs)
            parsed = json.loads(stdout.decode("utf-8").splitlines()[0])
            observed_baselines.append({
                "argv": list(charter["measurement"]["argv"]),
                "return_code": self._process.returncode,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "parsed_result": parsed,
            })
            return stdout, stderr

        def __getattr__(self, name):
            return getattr(self._process, name)

    def observe_popen(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        process = original_popen(*args, **kwargs)
        if list(argv) == charter["measurement"]["argv"]:
            return ObservedProcess(process)
        return process

    def observe_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        result = original_run(*args, **kwargs)
        if isinstance(argv, list) and "--exercise-component" in argv:
            parsed = json.loads(result.stdout.decode("utf-8").splitlines()[0])
            observed_components.append({
                "name": argv[argv.index("--exercise-component") + 1],
                "argv": list(argv), "return_code": result.returncode,
                "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
                "parsed_result": parsed,
            })
        return result

    # These wrappers delegate to the real subprocess implementations.  They
    # observe the invocations rather than fabricating execution evidence.
    monkeypatch.setattr(preflight.subprocess, "Popen", observe_popen)
    monkeypatch.setattr(preflight.subprocess, "run", observe_run)
    preflight.approve(charter_path)
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    anchor = json.load(open(preflight.anchor_path(run_dir, "bundle")))
    execution = anchor["execution"]
    assert execution["schema"] == "goal-loop-approval/v1"
    assert execution["before"] == execution["after"]
    baseline = execution["baseline"]
    assert baseline["argv"] == charter["measurement"]["argv"]
    assert baseline["cwd"] == charter["measurement"]["cwd"]
    assert baseline["timeout_s"] == charter["measurement"]["timeout_s"]
    assert baseline["source_paths"] == charter["measurement"]["source_paths"]
    assert baseline["return_code"] == 0
    assert baseline["test_id"] == preflight.BASELINE_PROFILE_ID
    assert baseline["value_count_binding"] == {"value": 0, "count": 1}
    assert len(observed_baselines) == 1
    assert {field: baseline[field] for field in (
        "argv", "return_code", "stdout_sha256", "stderr_sha256", "parsed_result"
    )} == observed_baselines[0]
    for field in ("executable", "source_digests", "stdout_sha256",
                  "stderr_sha256", "started_at", "finished_at",
                  "started_mono", "finished_mono", "parsed_result"):
        assert field in baseline
    rows = {row["name"]: row for row in execution["components"]}
    assert set(rows) == set(preflight.REQUIRED_COMPONENTS)
    assert len(observed_components) == len(preflight.REQUIRED_COMPONENTS)
    assert {item["name"] for item in observed_components} == set(
        preflight.REQUIRED_COMPONENTS)
    for name in preflight.REQUIRED_COMPONENTS:
        row = rows[name]
        matching = [item for item in observed_components if item["name"] == name]
        assert len(matching) == 1
        assert {field: row[field] for field in (
            "name", "argv", "return_code", "stdout_sha256", "stderr_sha256",
            "parsed_result"
        )} == matching[0]
        assert row["exercise_id"] == preflight.COMPONENT_EXERCISE_REGISTRY[name]["exercise_id"]
        assert row["profile_id"] == preflight.COMPONENT_EXERCISE_REGISTRY[name]["profile_id"]
        assert row["test_id"] == preflight.COMPONENT_EXERCISE_REGISTRY[name]["test_id"]
        assert row["return_code"] == 0 and row["result"] == "success"
        assert row["parsed_result"]["sha256"] == row["sha256"]
        for field in ("argv", "cwd", "timeout_s", "executable", "source_paths",
                      "source_digests", "fixed_source_digests",
                      "declared_source_digests", "started_at", "finished_at",
                      "started_mono", "finished_mono", "stdout_sha256",
                      "stderr_sha256", "component_digest_before",
                      "component_digest_after", "subject_digest_before",
                      "subject_digest_after"):
            assert field in row
        assert row["finished_at"] >= row["started_at"]
        assert row["finished_mono"] >= row["started_mono"]
        assert row["component_digest_before"] == row["component_digest_after"] == row["sha256"]
        assert row["subject_digest_before"] == row["subject_digest_after"]
    assert not os.path.exists(os.path.join(run_dir, "baseline.json"))
    assert not os.path.exists(os.path.join(run_dir, "component-evidence.json"))


def test_b4_missing_contained_execution_provenance_refuses_before_effects(tmp_path):
    """Contained values without an execution record cannot authorize approval."""
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    receipt_path = charter["containment"]["negative_test"]["receipt_path"]
    receipt = json.load(open(receipt_path))
    del receipt["provenance"]
    json.dump(receipt, open(receipt_path, "w"), sort_keys=True)
    # Re-pin, or this refuses as a swapped receipt and never reaches the
    # provenance gate this test is named for.
    charter["containment"]["negative_test"]["receipt_sha256"] = \
        preflight.file_digest(receipt_path)
    json.dump(charter, open(charter_path, "w"))
    with pytest.raises(RuntimeError, match="contained execution provenance"):
        preflight.approve(charter_path)
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_positive_anchor_binds_real_receipt_bytes_outcomes_and_provenance(tmp_path):
    """The anchor binds the host-observed bundle1 receipt, not fixture values."""
    run_dir, charter_path = _approval_contract_fixture(tmp_path)
    charter = json.load(open(charter_path))
    containment = charter["containment"]
    # Derive the receipt from the checked-in declaration rather than hardcoding
    # a path here. The declaration is what a charter points at, so a test that
    # reconstructs the path itself would keep passing after the declaration
    # stopped naming this file.
    declaration = json.load(open(os.path.realpath(os.path.join(
        os.path.dirname(HERE), "evidence", "bundle1",
        "receipt-declaration.json"))))
    receipt_path = declaration["receipt_path"]
    declared_sha = declaration["receipt_sha256"]
    with open(receipt_path, "rb") as fh:
        receipt_bytes = fh.read()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    evidence_dir = os.path.dirname(receipt_path)
    control = json.load(open(os.path.join(evidence_dir, "control.json")))
    contained = json.load(open(os.path.join(evidence_dir, "contained.json")))
    targets = json.load(open(os.path.join(evidence_dir, "targets.json")))
    assert receipt["targets"] == targets
    assert receipt["probes"] == {
        name: {"control": control["probes"][name],
               "contained": contained["probes"][name]}
        for name in control["probes"]}
    assert receipt["provenance"]["contained"] == {
        key: contained[key] for key in ("cgroup", "invocation_id")
    } | {"unit": "goal-loop-containment-probe.service"}
    containment["negative_test"] = {"receipt_path": receipt_path,
                                    "receipt_sha256": declared_sha}
    containment["probe_targets"] = receipt["targets"]
    containment["verified_at"] = receipt["at"]
    manifest = charter["source_manifest"]
    fixture_receipt = os.path.realpath(os.path.join(
        str(tmp_path), "containment-receipt.json"))
    manifest["files"] = [receipt_path if path == fixture_receipt else path
                          for path in manifest["files"]]
    json.dump(charter, open(charter_path, "w"))

    preflight.approve(charter_path)
    anchor = json.load(open(preflight.anchor_path(run_dir, charter["run_id"])))
    binding = anchor.get("containment_receipt")
    assert binding and binding["path"] == receipt_path
    assert binding["sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert binding["sha256"] == declared_sha, \
        "the declaration no longer pins the receipt it names"
    assert binding["probes"] == receipt["probes"]
    assert binding["provenance"] == receipt["provenance"]
    assert binding["provenance"]["contained"]["invocation_id"]


def test_b4_child_effect_residual_is_limited_to_snapshot_inputs():
    assert preflight.RESIDUAL_REASONS["approval_child_effects_not_rolled_back"] == (
        "approval detects drift in the tracked control-plane tree, declared "
        "source/component inputs, and subject (tracked-tree/source drift) and "
        "refuses without isolating or rolling them back; only approval-owned "
        "durable state is absent"
    )


def test_b4_relative_baseline_write_leaves_subject_and_control_tree_unchanged(tmp_path):
    """A failed approval command must not write relative to the run inputs."""
    marker = os.path.join(str(tmp_path), "subject", "relative-baseline-artifact")
    measurement = _measurement(
        tmp_path,
        "open('relative-baseline-artifact', 'w').write('mutated')\n"
        "print('{\"value\": 0, \"count\": 1}')\n")
    run_dir = _unattended(str(tmp_path), approve=False, measurement=measurement)
    measurement["cwd"] = os.path.realpath(os.path.join(str(tmp_path), "subject"))
    charter_path = os.path.join(run_dir, "charter.json")
    charter = json.load(open(charter_path))
    charter["measurement"] = measurement
    json.dump(charter, open(charter_path, "w"))
    preflight.approve(charter_path)
    assert not os.path.exists(marker)


def test_b4_timed_out_baseline_kills_and_reaps_descendants(tmp_path):
    """A timed-out baseline must not leave its child running."""
    child_pid = os.path.join(str(tmp_path), "baseline-child.pid")
    child = (f"import os,time; open({child_pid!r}, 'w').write(str(os.getpid())); "
             "time.sleep(30)")
    source = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
              "time.sleep(30)\n")
    run_dir = _unattended(str(tmp_path), approve=False,
                          measurement=_measurement(tmp_path, source))
    with pytest.raises(RuntimeError, match="timed out"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not os.path.exists(child_pid):
        time.sleep(0.02)
    assert os.path.exists(child_pid)
    pid = int(open(child_pid).read())
    assert not _pid_alive(pid), f"baseline descendant survived timeout: {pid}"


def test_iterate_validates_source_and_receipt_before_terminal_recovery_and_notify(
        tmp_path, monkeypatch):
    """A valid terminal STOP recovers only after real source/receipt checks."""
    run_dir = _unattended(str(tmp_path))
    open(os.path.join(run_dir, "STOP"), "w").write("owner requested\n")
    events = []
    original_source = preflight.check_source_binding
    original_receipt = preflight.check_containment_evidence
    original_notify = iterate.notifier.notify

    def source(*args, **kwargs):
        result = original_source(*args, **kwargs)
        events.append("source")
        return result

    def receipt(*args, **kwargs):
        result = original_receipt(*args, **kwargs)
        events.append("receipt")
        return result

    def notify(*args, **kwargs):
        events.append("notify")
        return original_notify(*args, **kwargs)

    # The host does not provide the approved user-unit binding; only that
    # unavailable prerequisite is patched. Source, receipt, and notifier
    # semantics remain real and are observed by wrappers.
    monkeypatch.setattr(preflight, "check_runtime_binding",
                        lambda *args, **kwargs: (None, {"test": True}))
    monkeypatch.setattr(preflight, "check_source_binding", source)
    monkeypatch.setattr(preflight, "check_containment_evidence", receipt)
    monkeypatch.setattr(iterate.notifier, "notify", notify)
    monkeypatch.setattr(preflight, "run_live",
                        lambda *args, **kwargs: pytest.fail("live containment ran"))

    iterate.iterate(run_dir, supervised=False)
    assert events == ["source", "receipt", "notify"]


def test_iterate_refuses_unbound_receipt_during_terminal_recovery_without_effects(
        tmp_path, monkeypatch):
    """Receipt refusal is terminal but does not recover, probe, or notify."""
    run_dir = _unattended(str(tmp_path))
    open(os.path.join(run_dir, "STOP"), "w").write("owner requested\n")
    anchor_path = preflight.anchor_path(run_dir, "bundle")
    anchor = json.load(open(anchor_path))
    del anchor["containment_receipt"]
    json.dump(anchor, open(anchor_path, "w"), sort_keys=True)
    results = os.path.join(run_dir, "results")
    os.rmdir(results)
    events = []
    original_source = preflight.check_source_binding
    original_receipt = preflight.check_containment_evidence
    original_notify = iterate.notifier.notify

    def source(*args, **kwargs):
        result = original_source(*args, **kwargs)
        events.append("source")
        return result

    def receipt(*args, **kwargs):
        result = original_receipt(*args, **kwargs)
        events.append("receipt")
        return result

    def notify(*args, **kwargs):
        events.append("notify")
        return original_notify(*args, **kwargs)

    monkeypatch.setattr(preflight, "check_runtime_binding",
                        lambda *args, **kwargs: (None, {"test": True}))
    monkeypatch.setattr(preflight, "check_source_binding", source)
    monkeypatch.setattr(preflight, "check_containment_evidence", receipt)
    monkeypatch.setattr(iterate.notifier, "notify", notify)
    monkeypatch.setattr(preflight, "run_live",
                        lambda *args, **kwargs: pytest.fail("live containment ran"))

    with pytest.raises(SystemExit):
        iterate.iterate(run_dir, supervised=False)
    assert events == ["source", "receipt"]
    assert not os.path.exists(results)
    ok, detail = testkit.stopped_because(
        run_dir, "containment receipt is not bound to the approval anchor", "", 0)
    assert ok, detail
    assert not any(record["event"] == "notify" for record in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b4_component_exercise_dispatch_has_behavioral_postcondition():
    name = "containment"
    path = os.path.join(HERE, "preflight.py")
    registry = preflight.COMPONENT_EXERCISE_REGISTRY[name]
    result = subprocess.run(
        preflight._component_command(
            name, {"path": path, "sha256": preflight.file_digest(path)}, registry),
        cwd=HERE, capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout)
    assert observed["postcondition"] == registry["postcondition"]
    assert observed["postcondition"]["invariant"] != "digest-only"


def test_b4_reapproval_verifies_records_without_rerunning_commands(tmp_path,
                                                                     monkeypatch):
    run_dir = _unattended(str(tmp_path))
    charter_path = os.path.join(run_dir, "charter.json")
    monkeypatch.setattr(preflight, "_run_baseline",
                        lambda *args: pytest.fail("baseline reran on reapproval"))
    monkeypatch.setattr(preflight, "_run_component_exercise",
                        lambda *args: pytest.fail("component exercise reran on reapproval"))
    preflight.approve(charter_path)


@pytest.mark.parametrize("field", ["measurement", "subject"])
def test_b4_post_approval_bound_input_drift_refuses(tmp_path, field):
    run_dir = _unattended(str(tmp_path))
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    if field == "measurement":
        source = charter[field]["source_paths"][0]
        with open(source, "a") as fh:
            fh.write("# changed after approval\n")
    else:
        open(os.path.join(charter["subject"], "changed"), "w").write("x")
    with pytest.raises(RuntimeError, match="drifted after approval"):
        preflight.approve(os.path.join(run_dir, "charter.json"))


@pytest.mark.parametrize("bad", ["result", "success", "receipt", "test_id"])
def test_b4_measurement_instruction_rejects_result_fields(tmp_path, bad):
    run_dir = _unattended(str(tmp_path), approve=False)
    charter_path = os.path.join(run_dir, "charter.json")
    charter = json.load(open(charter_path))
    charter["measurement"][bad] = "forged"
    json.dump(charter, open(charter_path, "w"))
    with pytest.raises(RuntimeError, match="result or receipt"):
        preflight.approve(charter_path)
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_rejects_missing_headless_binding(tmp_path):
    _refusal(str(tmp_path), "model=null", model="")
    # A separate exercise keeps the route/degrade branch discriminating too.
    other = tmp_path / "route"
    other.mkdir()
    _refusal(str(other), "degrade", route={"primary": "agent=luna"})


def test_b4_rejects_undeclared_effects(tmp_path):
    subject = os.path.join(str(tmp_path), "subject")
    state = os.path.join(str(tmp_path), "state")
    _refusal(str(tmp_path), "external_effects", containment={
        "required": True, "unit": os.path.join(str(tmp_path), "loop.service"),
        "writable_paths": [state]})


def test_b4_rejects_resource_mismatches_and_unbounded_trials(tmp_path):
    _refusal(str(tmp_path), "TimeoutStartSec", budgets={"max_wallclock_s": 600,
             "max_infra_retries": 3, "max_iteration_duration_s": 31,
             "max_trial_duration_s": 10})
    other = tmp_path / "trial"
    other.mkdir()
    _refusal(str(other), "exceeds the usable iteration interval", budgets={"max_wallclock_s": 600,
             "max_infra_retries": 3, "max_iteration_duration_s": 30,
             "max_trial_duration_s": 31})
    other = tmp_path / "memory"
    other.mkdir()
    _refusal(str(other), "MemoryMax", resources={"MemoryMax": "1G", "CPUQuota": "50%",
             "TasksMax": 64})
    other = tmp_path / "cpu"
    other.mkdir()
    _refusal(str(other), "CPUQuota", resources={"MemoryMax": "512M", "CPUQuota": "25%",
             "TasksMax": 64})
    other = tmp_path / "tasks"
    other.mkdir()
    _refusal(str(other), "TasksMax", resources={"MemoryMax": "512M", "CPUQuota": "50%",
             "TasksMax": 32})


@pytest.mark.parametrize("missing", ["MemoryMax", "CPUQuota", "TasksMax"],
                         ids=["missing-MemoryMax", "missing-CPUQuota", "missing-TasksMax"])
def test_b4_rejects_missing_resource_binding(tmp_path, missing):
    resources = {"MemoryMax": "512M", "CPUQuota": "50%", "TasksMax": 64}
    del resources[missing]
    _refusal(str(tmp_path), missing, resources=resources)


def test_b4_rejects_missing_or_unexercised_component(tmp_path):
    components = {}
    names = {"containment": "preflight.py", "lease": "ledger.py",
             "collector": "collector.py", "attempt_id": "iterate.py",
             "evaluator": "evaluator.py", "ledger_writer": "ledger.py"}
    for name, filename in names.items():
        path = os.path.join(HERE, filename)
        sha = preflight.file_digest(path)
        components[name] = {
            "path": path, "sha256": sha,
            "supervised_exercise_receipt": {
                "exercised": True, "status": "success",
                "identity": f"{name}:{sha}", "evidence": "fixture"}}
    components.pop("collector")
    _refusal(str(tmp_path), "collector", components=components)
    other = tmp_path / "component"
    other.mkdir()
    components["collector"] = {
        "path": os.path.join(HERE, "collector.py"),
        "sha256": preflight.file_digest(os.path.join(HERE, "collector.py"))}
    _refusal(str(other), "exercise receipt", components=components)
    other = tmp_path / "nonexec"
    other.mkdir()
    nonexec = os.path.join(str(other), "component")
    open(nonexec, "w").write("#!/bin/sh\nexit 0\n")
    components["collector"] = {"path": nonexec,
                                "supervised_exercise_receipt": {"ok": True}}
    _refusal(str(other), "cannot declare an exercise receipt", components=components)


def test_b4_rejects_undocumented_residual(tmp_path):
    _refusal(str(tmp_path), "canonical residual IDs", unenforced_risks=[{
        "risk": "made-up residual", "reason": "deliberately accepted for testing"}])


def test_b5_hung_trial_has_a_durable_deadline_reason(tmp_path):
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(os.path.join(run_dir, "results"))
    charter = testkit.charter(str(tmp_path), "hung", execution_mode="supervised",
                              arms={"a": {"argv": [sys.executable, "-c",
                                                    "import time; time.sleep(30)"],
                                           "cwd": str(tmp_path)}},
                              budgets={"max_wallclock_s": 30, "max_infra_retries": 3,
                                       "max_trial_duration_s": 1})
    testkit.write_charter(run_dir, charter)
    result = testkit.fire(run_dir, timeout=10)
    ok, detail = testkit.stopped_because(run_dir, "trial deadline exceeded",
                                         result.stdout + result.stderr,
                                         result.returncode)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b5_supervised_timeout_leaves_no_trial_processes(tmp_path):
    """The collector must get time to run its process-group cleanup.

    The outer subprocess clock starts before collector.py has initialized.  On
    the old equal-timeout path it killed collector.py first, leaving this
    session's trial leader and child alive even though the ledger said the
    iteration stopped.
    """
    leader_pid = os.path.join(str(tmp_path), "leader.pid")
    child_pid = os.path.join(str(tmp_path), "child.pid")
    child = ("import os,time; "
             f"open({child_pid!r}, 'w').write(str(os.getpid())); "
             "time.sleep(30)")
    leader = ("import os,subprocess,sys,time; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
              f"open({leader_pid!r}, 'w').write(str(os.getpid())); "
              "time.sleep(30)")
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(os.path.join(run_dir, "results"))
    charter = testkit.charter(
        str(tmp_path), "supervised-liveness", arms={
            "a": {"argv": [sys.executable, "-c", leader],
                  "cwd": str(tmp_path)}},
        budgets={"max_wallclock_s": 30, "max_infra_retries": 3,
                 "max_trial_duration_s": 0.5})
    testkit.write_charter(run_dir, charter)

    result = testkit.fire(run_dir, timeout=10)
    ok, detail = testkit.stopped_because(
        run_dir, "trial deadline exceeded",
        result.stdout + result.stderr, result.returncode)
    assert ok, detail
    assert not any(r["event"] in ("trial", "discarded") for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not (
            os.path.exists(leader_pid) and os.path.exists(child_pid)):
        time.sleep(0.02)
    pids = [int(open(path).read()) for path in (leader_pid, child_pid)]
    try:
        assert all(not _pid_alive(pid) for pid in pids), \
            f"trial processes survived collector timeout: {pids}"
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().split()[2]
        if state == "Z":
            return False
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def test_b5_setsid_descendants_require_the_unit_cgroup_backstop():
    # airgap-skillset port: reads the templated .example unit (placeholder
    # paths for an arbitrary deployment), not a concrete deployed unit.
    # Assertions are the same security-relevant fields as upstream; none of
    # them are path-specific. See PORTING.md.
    unit = open(os.path.join(HERE, "goal-loop-p4.service.example")).read()
    assert "KillMode=control-group" in unit
    assert "TimeoutStartSec=600" in unit
    assert "TimeoutStopSec=1s" in unit


def test_b5_process_group_only_cleanup_does_not_claim_setsid_containment(tmp_path):
    """A real escaped child is a red witness for process-group-only cleanup.

    Without a live user systemd unit this fixture intentionally proves survival,
    not death.  The static unit assertions above are the required cgroup
    backstop; claiming the child died here would overstate rootless evidence.
    """
    marker = os.path.join(str(tmp_path), "child.pid")
    child = ("import os,signal,time; os.setsid(); "
             "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             f"open({marker!r},'w').write(str(os.getpid())); time.sleep(30)")
    leader = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)")
    spec = json.dumps({"argv": [sys.executable, "-c", leader], "cwd": str(tmp_path),
                       "run_id": "setsid", "trial_index": 0, "arm": "a",
                       "attempt": "setsid"})
    artifact = os.path.join(str(tmp_path), "trial.json")
    subprocess.run([sys.executable, os.path.join(HERE, "collector.py"), spec,
                    "1", "0.2", artifact], capture_output=True, timeout=10)
    time.sleep(0.2)
    pid = int(open(marker).read())
    try:
        os.kill(pid, 0)
        survived = True
    except OSError:
        survived = False
    if survived:
        os.kill(pid, signal.SIGKILL)
    assert survived


def test_b6_changed_listed_source_stops_before_trial(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    unit = os.path.join(str(tmp_path), "loop.service")
    with open(unit, "a") as fh:
        fh.write("# changed after approval\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed", "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b6_missing_listed_source_stops_before_trial(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    os.unlink(os.path.join(str(tmp_path), "loop.service"))
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "declared source is missing", "", 0)
    assert ok, detail


def test_b6_changed_manifest_and_commit_bindings_stop_durably(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    manifest, commit = source_manifest.default_paths(run_dir, "bundle")
    with open(manifest, "a") as fh:
        fh.write("\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "source manifest changed", "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))

    other = tmp_path / "commit"
    other.mkdir()
    run_dir = _unattended(str(other))
    _, commit = source_manifest.default_paths(run_dir, "bundle")
    with open(commit, "a") as fh:
        fh.write("tampered\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "commit.txt changed", "", 0)
    assert ok, detail


def test_b6_extra_manifest_member_is_not_accepted(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    manifest_path, _ = source_manifest.default_paths(run_dir, "bundle")
    extra = os.path.join(str(tmp_path), "unlisted-loop-source.py")
    open(extra, "w").write("# not part of the runtime closure\n")
    manifest = json.load(open(manifest_path))
    manifest["files"].append({"path": extra,
                               "sha256": preflight.file_digest(extra)})
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    # The binding is deliberately updated too: this isolates the declared-set
    # refusal from the separate changed-manifest digest refusal.
    anchor_path = preflight.anchor_path(run_dir, "bundle")
    anchor = json.load(open(anchor_path))
    anchor["source_binding"]["manifest_sha256"] = preflight.file_digest(manifest_path)
    with open(anchor_path, "w") as fh:
        json.dump(anchor, fh, sort_keys=True)
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "source set mismatch", "", 0)
    assert ok, detail


def test_b6_anchor_binding_is_checked(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    anchor_path = preflight.anchor_path(run_dir, "bundle")
    anchor = json.load(open(anchor_path))
    anchor["source_binding"]["commit_path"] = os.path.join(str(tmp_path), "other.commit")
    with open(anchor_path, "w") as fh:
        json.dump(anchor, fh, sort_keys=True)
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "anchor paths", "", 0)
    assert ok, detail


def _write_notifier(tmp_path, marker):
    path = os.path.join(str(tmp_path), "notify.py")
    open(path, "w").write(
        "import pathlib; pathlib.Path(%r).write_text('notified')\n" % marker)
    return path


def _iterate_without_live_containment(run_dir, monkeypatch):
    """Keep B-6 red witnesses independent of this host's unavailable sandbox."""
    monkeypatch.setattr(preflight, "run_live", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight, "check_runtime_binding",
                        lambda *args, **kwargs: (None, {"test": True}))
    try:
        iterate.iterate(run_dir, supervised=False)
    except SystemExit:
        pass


def test_b6_notifier_script_is_declared_and_changed_source_stops(tmp_path, monkeypatch):
    marker = os.path.join(str(tmp_path), "notified")
    script = _write_notifier(tmp_path, marker)
    run_dir = _unattended(str(tmp_path),
                          notify={"command": [sys.executable, script]})
    with open(script, "a") as fh:
        fh.write("# changed\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed",
                                         "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b6_python_module_is_declared_and_changed_source_stops(tmp_path, monkeypatch):
    module = os.path.join(str(tmp_path), "bound_module.py")
    open(module, "w").write("raise SystemExit(0)\n")
    run_dir = _unattended(str(tmp_path), arms={
        "a": {"argv": [sys.executable, "-m", "bound_module"],
              "cwd": str(tmp_path)}})
    with open(module, "a") as fh:
        fh.write("# changed\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed",
                                         "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b6_shell_sourced_dependency_is_declared_and_changed_source_stops(tmp_path, monkeypatch):
    dependency = os.path.join(str(tmp_path), "shell-dependency.sh")
    script = os.path.join(str(tmp_path), "shell-entry.sh")
    open(dependency, "w").write("true\n")
    open(script, "w").write("#!/bin/sh\n. %s\n" % dependency)
    os.chmod(script, 0o755)
    run_dir = _unattended(str(tmp_path), arms={
        "a": {"argv": [script], "cwd": str(tmp_path),
              "source_paths": [script, dependency]}})
    with open(dependency, "a") as fh:
        fh.write("# changed\n")
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed",
                                         "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b6_opaque_shell_command_is_rejected_durably(tmp_path, monkeypatch):
    _refusal(str(tmp_path), "opaque command", arms={
        "a": {"argv": ["/bin/sh", "-c", "true"], "cwd": str(tmp_path)}})


def test_b6_missing_declared_source_is_rejected_even_with_rebound_anchor(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path), approve=False, source_manifest={
        "manifest_path": source_manifest.default_paths(
            os.path.join(str(tmp_path), "run"), "bundle")[0],
        "commit_path": source_manifest.default_paths(
            os.path.join(str(tmp_path), "run"), "bundle")[1],
        "files": []})
    with pytest.raises(RuntimeError, match="source declaration"):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b6_manifest_missing_expected_entry_with_rebound_anchor_stops(tmp_path,
                                                                       monkeypatch):
    run_dir = _unattended(str(tmp_path))
    manifest_path, _ = source_manifest.default_paths(run_dir, "bundle")
    manifest = json.load(open(manifest_path))
    removed = manifest["files"].pop()
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    anchor_path = preflight.anchor_path(run_dir, "bundle")
    anchor = json.load(open(anchor_path))
    anchor["source_binding"]["manifest_sha256"] = preflight.file_digest(manifest_path)
    with open(anchor_path, "w") as fh:
        json.dump(anchor, fh, sort_keys=True)
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "source set mismatch", "", 0)
    assert ok, detail
    assert removed["path"]
    assert not any(r["event"] == "trial" for r in
                    ledger.read(os.path.join(run_dir, "iterations.jsonl")))


@pytest.mark.parametrize("kind", ["python_script", "python_module", "shell_script",
                                  "direct_executable"])
def test_b6_unrelated_source_cannot_bind_the_real_command_target(tmp_path, kind,
                                                                  monkeypatch):
    unrelated = os.path.join(str(tmp_path), "unrelated.py")
    open(unrelated, "w").write("# unrelated\n")
    if kind == "python_script":
        target = os.path.join(str(tmp_path), "target.py")
        open(target, "w").write("raise SystemExit(0)\n")
        arm = {"argv": [sys.executable, target], "cwd": str(tmp_path),
               "source_paths": [unrelated]}
        expected = "command target is not declared"
    elif kind == "python_module":
        target = os.path.join(str(tmp_path), "bound_target.py")
        open(target, "w").write("raise SystemExit(0)\n")
        arm = {"argv": [sys.executable, "-m", "bound_target"],
               "cwd": str(tmp_path), "source_paths": [unrelated]}
        expected = "command target is not declared"
    elif kind == "shell_script":
        target = os.path.join(str(tmp_path), "target.sh")
        open(target, "w").write("#!/bin/sh\nexit 0\n")
        os.chmod(target, 0o755)
        arm = {"argv": ["/bin/sh", target], "cwd": str(tmp_path),
               "source_paths": [unrelated]}
        expected = "command target is not declared"
    else:
        target = os.path.join(str(tmp_path), "target-exec")
        open(target, "w").write("#!/bin/sh\nexit 0\n")
        os.chmod(target, 0o755)
        arm = {"argv": [target], "cwd": str(tmp_path),
               "source_paths": [unrelated]}
        expected = "command target is not declared"
    _refusal(str(tmp_path), expected, arms={"a": arm})


def test_b6_unrelated_source_cannot_bind_a_direct_notifier(tmp_path, monkeypatch):
    marker = os.path.join(str(tmp_path), "notified")
    target = _write_notifier(tmp_path, marker)
    unrelated = os.path.join(str(tmp_path), "unrelated.py")
    open(unrelated, "w").write("# unrelated\n")
    _refusal(str(tmp_path), "command target is not declared", notify={
        "command": [target], "source_paths": [unrelated]})
    assert not os.path.exists(marker)


def test_b6_fixed_source_set_includes_all_entries_and_stop_sources():
    assert {"run.sh", "run-supervised.sh", "stop.sh", "stop.py",
            "source_manifest.py"}.issubset(set(source_manifest.FIXED_NAMES))


def test_b6_stop_entry_refuses_before_ledger_stop_on_binding_mismatch(tmp_path):
    marker = os.path.join(str(tmp_path), "notified")
    script = _write_notifier(tmp_path, marker)
    run_dir = _unattended(str(tmp_path), notify={"command": [sys.executable, script]})
    with open(os.path.join(str(tmp_path), "loop.service"), "a") as fh:
        fh.write("# changed after approval\n")
    result = subprocess.run([os.path.join(HERE, "stop.sh"), run_dir],
                            capture_output=True, text=True)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    assert result.returncode != 0
    assert not any(record["event"] == "stop" for record in records)
    assert "source binding" in open(os.path.join(run_dir, "STOP")).read()
    assert "listed source changed" in open(os.path.join(run_dir, "STOP")).read()
    assert not os.path.exists(marker)


def test_b6_binding_refusal_does_not_notify_or_create_results(tmp_path, monkeypatch):
    marker = os.path.join(str(tmp_path), "notified")
    script = _write_notifier(tmp_path, marker)
    run_dir = _unattended(str(tmp_path), notify={"command": [sys.executable, script]})
    with open(os.path.join(str(tmp_path), "loop.service"), "a") as fh:
        fh.write("# changed after approval\n")
    os.rmdir(os.path.join(run_dir, "results"))
    _iterate_without_live_containment(run_dir, monkeypatch)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    ok, detail = testkit.stopped_because(run_dir, "listed source changed", "", 0)
    assert ok, detail
    assert not os.path.exists(marker)
    assert not os.path.exists(os.path.join(run_dir, "results"))
    assert any(record["event"] == "stop" and
               "listed source changed" in record["reason"] for record in records)
    assert not any(record["event"] == "notify" for record in records)


def test_b6_binding_refusal_preserves_specific_reason_over_existing_stop(tmp_path,
                                                                          monkeypatch):
    marker = os.path.join(str(tmp_path), "notified")
    script = _write_notifier(tmp_path, marker)
    run_dir = _unattended(str(tmp_path), notify={"command": [sys.executable, script]})
    ledger.append(os.path.join(run_dir, "iterations.jsonl"), {
        "event": "stop", "reason": "owner stop", "summary": {}, **testkit.stamp()})
    with open(os.path.join(str(tmp_path), "loop.service"), "a") as fh:
        fh.write("# changed after approval\n")
    os.rmdir(os.path.join(run_dir, "results"))
    _iterate_without_live_containment(run_dir, monkeypatch)
    records = ledger.read(os.path.join(run_dir, "iterations.jsonl"))
    ok, detail = testkit.stopped_because(run_dir, "listed source changed", "", 0)
    assert ok, detail
    assert [record["reason"] for record in records if record["event"] == "stop"] == [
        "owner stop", "preflight refused: listed source changed after approval: "
        + os.path.join(str(tmp_path), "loop.service")]
    assert not os.path.exists(marker)
    assert not any(record["event"] == "notify" for record in records)


def test_b5_trial_and_work_deadline_reserve_termination_budget(tmp_path):
    _refusal(str(tmp_path), "usable iteration interval", budgets={
        "max_wallclock_s": 600, "max_infra_retries": 3,
        "max_iteration_duration_s": 30, "max_trial_duration_s": 30,
        "termination_duration_s": 1})


def test_b5_start_timeout_kills_without_uncharged_stop_grace(tmp_path):
    _refusal(str(tmp_path), "TimeoutStartFailureMode", unit_start_failure_mode=False)


def test_b5_remaining_budget_is_inside_iteration_reserve():
    charter = {"budgets": {"max_wallclock_s": 600,
                            "max_trial_duration_s": 100,
                            "max_iteration_duration_s": 30,
                            "termination_duration_s": 1}}
    assert iterate.remaining_budget(charter, 0, iteration_elapsed=0) == 29
    assert iterate.remaining_budget(charter, 0, iteration_elapsed=2) == 27


def test_b5_deadlines_charge_initialization_before_trial_and_cleanup():
    started = time.monotonic() - 2
    charter = {"budgets": {"max_wallclock_s": 600,
                            "max_trial_duration_s": 100,
                            "max_iteration_duration_s": 10,
                            "termination_duration_s": 1}}
    trial, collector = iterate.trial_deadlines(charter, 0, started)
    assert trial <= started + 9
    assert collector <= started + 10
    assert round(collector - trial, 6) == 1


def test_b5_supervised_missing_termination_duration_uses_bounded_default():
    charter = testkit.charter(".", "supervised-default")
    assert iterate.termination_duration(charter, supervised=True) == 1.0


def test_b4_residual_registry_rejects_readme_extra_and_reason_mismatch(tmp_path):
    risks = [{"risk": risk, "reason": reason}
             for risk, reason in zip(preflight.ACCEPTED_RESIDUAL_IDS,
                                     ["reason" for _ in preflight.ACCEPTED_RESIDUAL_IDS])]
    lines = ["## Not defended"]
    for risk in preflight.ACCEPTED_RESIDUAL_IDS:
        lines.append(f"- **{risk}** — reason")
    lines.append("- **extra_residual** — reason")
    readme = tmp_path / "README.md"
    readme.write_text("\n".join(lines) + "\n")
    _refusal(str(tmp_path), "residual registry", readme=str(readme),
             unenforced_risks=risks)


def test_b5_resource_policy_rejects_unit_timeout_drift(tmp_path):
    run_dir = _unattended(str(tmp_path), approve=False)
    unit = os.path.join(str(tmp_path), "loop.service")
    with open(unit, "a") as fh:
        fh.write("TimeoutStartSec=31\n")
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    reason = preflight._resource_contract(charter)
    assert reason and "driver_timeout_s" in reason


def test_b5_driver_deadline_is_strictly_before_collector_deadline():
    charter = {"budgets": {"max_wallclock_s": 600,
                            "max_iteration_duration_s": 30,
                            "max_trial_duration_s": 10,
                            "termination_duration_s": 2}}
    trial, collector = iterate.trial_deadlines(charter, 0, time.monotonic())
    assert collector - trial == pytest.approx(2)


def test_b6_source_binding_rejects_changed_manifest_before_trial(tmp_path):
    run_dir = _unattended(str(tmp_path))
    manifest_path, _ = source_manifest.default_paths(run_dir, "bundle")
    with open(manifest_path, "a") as fh:
        fh.write("\n")
    assert preflight.check_source_binding(
        json.load(open(os.path.join(run_dir, "charter.json"))), run_dir)


# airgap-skillset port: upstream had a `test_skill_mirrors_are_byte_identical`
# here, asserting byte-equality between a `.claude/skills/autonomous-goal-loop`
# and a `.codex/skills/autonomous-goal-loop` tree in the *source* repo. That
# invariant is specific to a repo laid out with dual harness-mirror trees;
# airgap-skillset is a single flat package meant to be dropped wherever a
# closed-network repo keeps its skills, with no dual-tree structure of its
# own to assert equality between. Removed rather than adapted — there is no
# analogous pair of trees on this side to compare. See PORTING.md.


def test_b6_source_verification_precedes_normal_entry_side_effects(tmp_path, monkeypatch):
    run_dir = _unattended(str(tmp_path))
    unit = os.path.join(str(tmp_path), "loop.service")
    with open(unit, "a") as fh:
        fh.write("# changed\n")
    results = os.path.join(run_dir, "results")
    os.rmdir(results)
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed",
                                         "", 0)
    assert ok, detail
    assert not os.path.exists(results)
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b6_source_verification_precedes_stop_recovery_and_notification(tmp_path, monkeypatch):
    marker = os.path.join(str(tmp_path), "notified")
    script = os.path.join(str(tmp_path), "notify.py")
    open(script, "w").write("import pathlib; pathlib.Path(%r).write_text('notified')\n" % marker)
    run_dir = _unattended(str(tmp_path),
                          notify={"command": [sys.executable, script]})
    open(os.path.join(run_dir, "STOP"), "w").write("owner requested\n")
    with open(os.path.join(str(tmp_path), "loop.service"), "a") as fh:
        fh.write("# changed\n")
    results = os.path.join(run_dir, "results")
    os.rmdir(results)
    _iterate_without_live_containment(run_dir, monkeypatch)
    ok, detail = testkit.stopped_because(run_dir, "listed source changed",
                                         "", 0)
    assert ok, detail
    assert not os.path.exists(results)
    assert not any(r["event"] == "trial" for r in
                   ledger.read(os.path.join(run_dir, "iterations.jsonl")))


def test_b5_termination_duration_is_declared_and_exactly_bound(tmp_path):
    _refusal(str(tmp_path), "TimeoutStopSec", budgets={
        "max_wallclock_s": 600, "max_infra_retries": 3,
        "max_iteration_duration_s": 30, "max_trial_duration_s": 10,
        "termination_duration_s": 2})


@pytest.mark.parametrize("field,value", [
    ("MemoryMax", "infinity"), ("CPUQuota", "0%"), ("TasksMax", "0"),
    ("MemoryMax", "malformed"), ("CPUQuota", "not-a-percent"),
    ("TasksMax", "-1"),
])
def test_b5_rejects_nonpositive_infinite_or_malformed_resources(tmp_path, field, value):
    resources = {"MemoryMax": "512M", "CPUQuota": "50%", "TasksMax": 64}
    resources[field] = value
    _refusal(str(tmp_path), "finite positive", unit_resources=True, resources=resources)


def _bad_component_components():
    names = {"containment": "preflight.py", "lease": "ledger.py",
             "collector": "collector.py", "attempt_id": "iterate.py",
             "evaluator": "evaluator.py", "ledger_writer": "ledger.py"}
    result = {}
    for name, filename in names.items():
        path = os.path.join(HERE, filename)
        sha = preflight.file_digest(path)
        result[name] = {
            "path": path, "sha256": sha,
            "supervised_exercise_receipt": {
                "exercised": True, "status": "success",
                "identity": f"{name}:{sha}", "evidence": "fixture"}}
    return result


def test_b4_requires_the_complete_canonical_residual_registry(tmp_path):
    risks = [{"risk": risk, "reason": preflight.RESIDUAL_REASONS[risk]}
             for risk in preflight.ACCEPTED_RESIDUAL_IDS[:-1]]
    _refusal(str(tmp_path), "canonical residual IDs", unenforced_risks=risks)


def test_b4_approval_binds_a_deterministic_subject_digest(tmp_path):
    run_dir = _unattended(str(tmp_path))
    anchor = json.load(open(preflight.anchor_path(run_dir, "bundle")))
    binding = anchor.get("subject_binding")
    assert binding and binding["path"] == os.path.realpath(
        os.path.join(str(tmp_path), "subject"))
    assert binding["sha256"] == preflight.path_digest(binding["path"])


def test_b4_subject_mutation_after_preflight_stops_before_acceptance(tmp_path,
                                                                      monkeypatch):
    subject = os.path.join(str(tmp_path), "subject")
    arm = os.path.join(str(tmp_path), "mutate.py")
    open(arm, "w").write(
        "import pathlib; pathlib.Path(%r, 'changed').write_text('changed')\n"
        "raise SystemExit(0)\n" % subject)
    run_dir = _unattended(str(tmp_path), arms={
        "a": {"argv": [sys.executable, arm], "cwd": str(tmp_path),
              "source_paths": [os.path.realpath(arm)]}})
    monkeypatch.setattr(preflight, "run_live", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight, "check_runtime_binding",
                        lambda *args, **kwargs: (None, {"test": True}))
    try:
        iterate.iterate(run_dir, supervised=False)
    except SystemExit:
        pass
    ok, detail = testkit.stopped_because(run_dir, "subject changed", "", 0)
    assert ok, detail
    assert not any(r["event"] == "trial" for r in ledger.read(
        os.path.join(run_dir, "iterations.jsonl")))


def test_b4_truthy_baseline_without_measurement_receipt_is_rejected(tmp_path):
    run_dir = _unattended(str(tmp_path), approve=False, baseline={
        "value": 0, "commit": "HEAD", "measured_at": "now"})
    charter_path = os.path.join(run_dir, "charter.json")
    with pytest.raises(RuntimeError, match="baseline"):
        preflight.approve(charter_path)
    assert not os.path.exists(os.path.join(run_dir, "charter.sha256"))


@pytest.mark.parametrize("field,value,expected", [
    ("model", "arbitrary-default", "model"),
    ("route", {"primary": "arbitrary", "degrade": "continue"}, "route"),
])
def test_b4_rejects_arbitrary_model_or_route(tmp_path, field, value, expected):
    run_dir = _unattended(str(tmp_path), approve=False, **{field: value})
    with pytest.raises(RuntimeError, match=expected):
        preflight.approve(os.path.join(run_dir, "charter.json"))
    _assert_no_approval_owned_durable_state(run_dir)


def test_b4_charter_only_component_receipts_are_not_approval_evidence(tmp_path):
    run_dir = _unattended(str(tmp_path))
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    charter["components"]["collector"]["supervised_exercise_receipt"] = {
        "exercised": True, "status": "success", "identity": "minted",
        "evidence": "charter-only fiction"}
    reason = preflight.check_unattended_entry(charter, run_dir)
    assert reason and "cannot declare an exercise receipt" in reason


def test_b5_policy_requires_start_failure_kill_control_group_and_send_sigkill(tmp_path):
    run_dir = _unattended(str(tmp_path))
    unit = os.path.join(str(tmp_path), "loop.service")
    lines = [line for line in open(unit).read().splitlines()
             if not line.startswith("SendSIGKILL=")]
    open(unit, "w").write("\n".join(lines) + "\n")
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    reason = preflight._resource_contract(charter)
    assert reason and "SendSIGKILL" in reason


def test_b5_kill_tree_records_term_kill_and_reap_order(monkeypatch):
    events = []

    class FakeProc:
        def wait(self, **kwargs):
            events.append("reap")

    def killpg(pgid, sig):
        events.append("term" if sig == signal.SIGTERM else "kill")

    monkeypatch.setattr(os, "killpg", killpg)
    collector._kill_tree(FakeProc(), 123)
    assert events == ["term", "kill", "reap"]


def test_b5_driver_deadline_leaves_collector_cleanup_reserve():
    charter = {"budgets": {"max_wallclock_s": 600,
                            "max_iteration_duration_s": 30,
                            "max_trial_duration_s": 10,
                            "termination_duration_s": 2}}
    trial, collector = iterate.trial_deadlines(charter, 0, time.monotonic())
    assert collector > trial
    assert collector - trial == pytest.approx(2)
    assert collector - time.monotonic() < 30


def test_b5_unattended_requires_active_approved_systemd_binding(tmp_path,
                                                                monkeypatch):
    run_dir = _unattended(str(tmp_path))
    charter = json.load(open(os.path.join(run_dir, "charter.json")))
    observed = {"cgroup": "/user.slice/user-1000.slice/app.slice/loop.service",
                "unit": "loop.service", "invocation_id": "invocation",
                "exec_pid": str(os.getpid())}
    reason, record = preflight.check_runtime_binding(charter, observed=observed)
    assert reason is None
    assert record["unit"] == "loop.service"
    monkeypatch.setattr(preflight, "read_runtime_binding", lambda: {
        **observed, "cgroup": "/user.slice/user-1000.slice/app.slice/other.service"})
    reason, _ = preflight.check_runtime_binding(charter)
    assert reason and "approved unit" in reason


def test_b6_git_resolution_does_not_use_inherited_path(tmp_path, monkeypatch):
    fake = tmp_path / "git"
    fake.write_text("#!/bin/sh\nprintf fake\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert source_manifest.current_commit(HERE) != "fake"


def test_b6_undeclared_python_import_is_narrowed_by_mandatory_residual(tmp_path):
    module = tmp_path / "runtime_dependency.py"
    module.write_text("VALUE = 1\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import runtime_dependency\n")
    charter = {"arms": {"a": {"argv": [sys.executable, str(entry)],
                               "cwd": str(tmp_path),
                               "source_paths": [str(entry)]}},
               "notify": {"command": ["/usr/bin/true"],
                          "source_paths": ["/usr/bin/true"]}}
    sources, reason = source_manifest._command_declarations(charter)
    assert reason is None and os.path.realpath(str(module)) not in sources
    assert "undeclared_runtime_imports" in preflight.ACCEPTED_RESIDUAL_IDS
    assert "undeclared_runtime_imports" in open(
        os.path.join(os.path.dirname(HERE), "README.md")).read()
