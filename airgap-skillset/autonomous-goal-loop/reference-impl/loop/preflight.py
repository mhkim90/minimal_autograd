"""Preflight entry conditions, implemented as checks rather than paragraphs.

Supervised entry preserves the original lightweight fixture contract. Unattended
entry additionally checks the B-4 charter, resource and component contracts,
then B-6 verifies the approval-bound runtime source before any trial action.

Every function here returns a reason string on refusal and None on success.
"""
import errno
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL_PROBE_SOURCE = os.path.realpath(os.path.join(HERE, "negative_test.py"))
CANONICAL_HOME_PROBE = "write outside permitted paths (home)"
CANONICAL_REPO_PROBE = "write to repo outside run dir"
CANONICAL_PROBE_NAMES = frozenset({
    CANONICAL_HOME_PROBE,
    CANONICAL_REPO_PROBE,
    "undeclared network connection",
    "privilege acquisition (user namespace UID 0)",
})

REQUIRED = ("run_id", "class", "subject", "execution_mode", "arms", "stopping_rule", "budgets",
            "notify", "containment", "approved_by", "approved_at")

SUPPORTED_CLASSES = ("evidence",)
EXECUTION_MODES = ("unattended", "supervised")
COMPONENT_SOURCES = {
    "containment": "preflight.py", "lease": "ledger.py",
    "collector": "collector.py", "attempt_id": "iterate.py",
    "evaluator": "evaluator.py", "ledger_writer": "ledger.py",
}
REQUIRED_COMPONENTS = tuple(COMPONENT_SOURCES)
RESIDUAL_REASONS = {
    "privileged_cgroup_escape": "privileged actors may escape the unit cgroup",
    "unavailable_cgroup_controllers": "resource controllers may be unavailable",
    "pre_existing_external_daemons": "pre-existing external daemons are outside this run",
    "post_verification_source_toctou": "post-verification source mutation is outside this fallible-actor contract",
    "notification_exactly_once": "notification is at-least-once rather than exactly once",
    "reboot_wallclock_accounting": "reboot gaps rely on wall-clock accounting",
    "probe_target_toctou": "probe target validation and use are not one no-follow kernel operation",
    "operator_supplied_containment_control": "the operator-supplied receipt records host-observed four-probe outcomes and contained execution provenance; approval binds and parses its bytes but does not independently execute or attest them",
    "source_or_receipt_binding_refusal_unnotified": "source or containment-receipt binding refusal intentionally does not invoke the charter notifier because that notifier is governed by the failed source contract; iteration refusal writes a locked durable stop and fsynced STOP, while stop.sh refusal writes a fsynced STOP only, and neither path recovers, reconciles, probes, or notifies",
    "approval_child_effects_not_rolled_back": "approval detects drift in the tracked control-plane tree, declared source/component inputs, and subject (tracked-tree/source drift) and refuses without isolating or rolling them back; only approval-owned durable state is absent",
    "post_verification_tampering": "an actor may alter the verifier or anchor after verification",
    "bare_stop_during_iteration": "a bare STOP during a trial is only converted on the next entry",
    "post_decision_budget_crossing": "a deadline may pass during trial append or fsync after the decision",
    "clock_anomaly_granularity": "backward wall-clock and wall-monotonic disagreement share one anomaly check",
    "artifact_extra_fields": "unknown extra artifact fields are carried through unexamined",
    "ledger_event_value_domains": "recognised ledger event value domains are not fully checked",
    "inherited_environment": "the trial environment is inherited wholesale",
    "undeclared_runtime_imports": "Python runtime imports beyond declared command sources are not generically closed",
    "spawned_executables": "executables spawned by a declared command are not generically closed",
    "shell_interpreter_loader": "shell, interpreter, and dynamic-loader bindings are not a complete runtime closure",
    "commit_marker_limitations": "commit.txt proves the recorded repository marker only, not every runtime byte",
    "timer_behavior": "the timer is not disabled on stop",
    "local_approval_not_external_attestation": "approval is local execution evidence, not an independent external attestation",
    "measurement_command_semantics": "the measured command's inherited runtime semantics are not independently attested",
    "component_exercise_coverage": "registered component exercises cover the fixed profiles, not arbitrary implementations",
}
ACCEPTED_RESIDUAL_IDS = tuple(RESIDUAL_REASONS)

BASELINE_PARSER_ID = "json-value-count-v1"
BASELINE_PROFILE_ID = "baseline-command-v1"
CONTAINMENT_RECEIPT_SCHEMA = "goal-loop-containment-receipt/v1"
COMPONENT_EXERCISE_REGISTRY = {
    "containment": {"exercise_id": "goal-loop-component-containment-v1",
                    "profile_id": "containment-parser-refusal-v1",
                    "implementation": "exercise_containment_parser_v1",
                    "test_id": "containment-parser-refusal",
                    "postcondition": {"invariant": "resource-parser-refuses-invalid"}},
    "lease": {"exercise_id": "goal-loop-component-lease-v1",
              "profile_id": "lease-lock-integrity-v1",
              "implementation": "exercise_lease_lock_v1",
              "test_id": "lease-lock-append-read",
              "postcondition": {"invariant": "locked-ledger-append-read", "sequence": 1}},
    "collector": {"exercise_id": "goal-loop-component-collector-v1",
                   "profile_id": "collector-process-result-v1",
                   "implementation": "exercise_collector_process_v1",
                   "test_id": "collector-process-result",
                   "postcondition": {"invariant": "ok-process-result", "status": "ok"}},
    "attempt_id": {"exercise_id": "goal-loop-component-attempt-id-v1",
                    "profile_id": "attempt-deadline-stability-v1",
                    "implementation": "exercise_attempt_deadline_v1",
                    "test_id": "attempt-deadline-stability",
                    "postcondition": {"invariant": "deadline-reserves-termination", "reserve_s": 1}},
    "evaluator": {"exercise_id": "goal-loop-component-evaluator-v1",
                  "profile_id": "evaluator-stopping-interval-v1",
                  "implementation": "exercise_evaluator_stopping_v1",
                  "test_id": "evaluator-stopping-interval",
                  "postcondition": {"invariant": "stopping-rule-interval", "stopped": True}},
    "ledger_writer": {"exercise_id": "goal-loop-component-ledger-writer-v1",
                       "profile_id": "ledger-writer-fsync-integrity-v1",
                       "implementation": "exercise_ledger_writer_fsync_v1",
                       "test_id": "ledger-writer-fsync-read",
                       "postcondition": {"invariant": "fsync-append-read", "sequence": 1}},
}


def charter_digest(charter):
    return hashlib.sha256(
        json.dumps(charter, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check_schema(charter):
    if not isinstance(charter, dict):
        return f"charter is a {type(charter).__name__}, not a mapping"
    for field in ("stopping_rule", "budgets", "containment"):
        if field in charter and not isinstance(charter[field], dict):
            return f"charter.{field} is not a mapping"
    missing = [f for f in REQUIRED if f not in charter]
    if missing:
        return f"charter is missing required fields: {missing}"
    if charter["class"] not in SUPPORTED_CLASSES:
        # The ratchet and search classes are specified but unbuilt. Refusing
        # here is what keeps "specified" from silently becoming "running".
        return (f"class {charter['class']!r} is not implemented; "
                f"supported: {list(SUPPORTED_CLASSES)}")
    if charter["execution_mode"] not in EXECUTION_MODES:
        return (f"execution_mode {charter['execution_mode']!r} is not supported; "
                f"supported: {list(EXECUTION_MODES)}")

    # Shape, not just presence. Field-presence checks let an empty arms map and
    # a non-mapping notify through, and both crash later in code that has no way
    # to stop cleanly.
    if not isinstance(charter["arms"], dict) or not charter["arms"]:
        return "charter declares no arms"
    notify = charter.get("notify")
    if not isinstance(notify, dict) or not isinstance(notify.get("command"), list) \
            or not notify["command"] \
            or not all(isinstance(part, str) for part in notify["command"]):
        return "charter must declare notify.command as a non-empty list of strings"
    for field in ("min_trials_per_arm", "max_trials", "ci_width"):
        if field not in charter["stopping_rule"]:
            return f"stopping_rule is missing {field}"
        v = charter["stopping_rule"][field]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return f"stopping_rule.{field} must be a finite number"
    for field in ("max_wallclock_s", "max_infra_retries", "max_trial_duration_s"):
        if field not in charter["budgets"]:
            return f"budgets is missing {field}"
        v = charter["budgets"][field]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return f"budgets.{field} must be a finite number"
    if "termination_duration_s" in charter["budgets"]:
        v = charter["budgets"]["termination_duration_s"]
        if (not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v <= 0):
            return "budgets.termination_duration_s must be a finite positive number"

    for name, spec in charter["arms"].items():
        if not isinstance(spec, dict):
            return f"arm {name!r} is not a mapping"
        if "argv" not in spec or "cwd" not in spec:
            return f"arm {name!r} must declare both argv and cwd"
        if not os.path.isdir(spec["cwd"]):
            # A command that is green from a repository root can fail to even
            # collect one directory up, and those failures would be recorded as
            # observations about the subject.
            return f"arm {name!r} cwd does not exist: {spec['cwd']}"
    return None


def check_digest(charter, run_dir):
    """The charter is immutable after approval; prove it has not moved."""
    path = os.path.join(run_dir, "charter.sha256")
    current = charter_digest(charter)
    if not os.path.exists(path):
        return "charter.sha256 is missing; the charter was never approved"
    recorded = open(path).read().strip()
    if recorded != current:
        return f"charter changed after approval ({recorded[:12]} != {current[:12]})"
    return None


# Only these say a boundary refused the write. Anything else — a full disk, a
# quota, a name too long, a missing directory — is an unrelated failure that
# happens to raise OSError, and reading it as containment would pass an entirely
# uncontained run.
DENIED = {errno.EACCES, errno.EPERM, errno.EROFS}


def check_execution_mode(charter, supervised=False):
    expected = "supervised" if supervised else "unattended"
    if charter.get("execution_mode") != expected:
        return (f"execution mode mismatch: approved {charter.get('execution_mode')!r}, "
                f"entered through {expected!r} entry")
    if supervised and os.environ.get("INVOCATION_ID"):
        return "explicit supervised entry is refused under systemd"
    return None


def _overlap(path, other):
    return path == other or path.startswith(other + os.sep) or other.startswith(path + os.sep)


def _finite_positive(value, label):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        return f"{label} must be a finite positive number"
    return None


def _unit_value(lines, key):
    values = [line.partition("=")[2].strip() for line in lines
              if line.startswith(key + "=")]
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


def _seconds(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(s|sec|secs|seconds|min|m)?", value)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or "s"
    result = number * (60 if unit in ("min", "m") else 1)
    return result if math.isfinite(result) and result > 0 else None


def _memory_bytes(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) and value > 0 else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(B|K|M|G|T|P|E)?", value)
    if not match:
        return None
    multiplier = {None: 1, "B": 1, "K": 1024, "M": 1024 ** 2,
                  "G": 1024 ** 3, "T": 1024 ** 4,
                  "P": 1024 ** 5, "E": 1024 ** 6}[match.group(2)]
    result = float(match.group(1)) * multiplier
    return result if math.isfinite(result) and result > 0 else None


def _cpu_percent(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)%", value)
    if not match:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) and result > 0 else None


def _tasks(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        return None
    result = int(value)
    return result if result > 0 else None


def _resource_contract(charter):
    budgets = charter["budgets"]
    for key in ("max_iteration_duration_s", "max_trial_duration_s"):
        problem = _finite_positive(budgets.get(key), f"budgets.{key}")
        if problem:
            return problem
    containment = charter["containment"]
    unit = containment.get("unit")
    try:
        lines = open(unit).read().splitlines()
    except OSError as exc:
        return f"resource unit is unreadable: {exc}"
    # The driver reserve is required, not defaulted. Defaulting it to the
    # in-process hard deadline puts systemd's TimeoutStartFailureMode=kill at the
    # same instant the in-process path is supposed to kill the collector and
    # record the overrun, so the cgroup kill can pre-empt the record. Omitting a
    # field must not be a way to select the racing configuration.
    driver_timeout = budgets.get("driver_timeout_s")
    problem = _finite_positive(driver_timeout, "budgets.driver_timeout_s")
    if problem:
        return problem
    if driver_timeout <= budgets["max_iteration_duration_s"]:
        return ("budgets.driver_timeout_s must be strictly greater than "
                "budgets.max_iteration_duration_s, so the driver deadline "
                "leaves an explicit reserve over the collector outer deadline")
    termination = _finite_positive(budgets.get("termination_duration_s"),
                                   "budgets.termination_duration_s")
    if termination:
        return termination
    usable = (budgets["max_iteration_duration_s"]
              - budgets["termination_duration_s"])
    if usable <= 0:
        return ("budgets.max_iteration_duration_s must leave a positive usable "
                "iteration interval after termination reserve")
    if budgets["max_trial_duration_s"] > usable:
        return ("budgets.max_trial_duration_s exceeds the usable iteration interval "
                "after termination reserve")
    timeout = _unit_value(lines, "TimeoutStartSec")
    timeout_seconds = _seconds(timeout)
    if timeout_seconds is None or timeout_seconds != driver_timeout:
        return ("budgets.driver_timeout_s must equal unit TimeoutStartSec "
                f"({driver_timeout!r} != {timeout!r})")
    if _unit_value(lines, "TimeoutStartFailureMode") != "kill":
        return "containment unit must set TimeoutStartFailureMode=kill"
    if _unit_value(lines, "KillMode") != "control-group":
        return "containment unit must set KillMode=control-group"
    if _unit_value(lines, "SendSIGKILL") != "yes":
        return "containment unit must set SendSIGKILL=yes"
    stop = _unit_value(lines, "TimeoutStopSec")
    stop_seconds = _seconds(stop)
    if stop_seconds is None or stop_seconds != budgets["termination_duration_s"]:
        return ("budgets.termination_duration_s must equal unit TimeoutStopSec "
                f"({budgets['termination_duration_s']!r} != {stop!r})")
    resources = charter.get("resources") or charter.get("resource_limits")
    if not isinstance(resources, dict):
        resources = containment.get("resources")
    if not isinstance(resources, dict):
        return "charter must declare resources MemoryMax, CPUQuota, and TasksMax"
    for key in ("MemoryMax", "CPUQuota", "TasksMax"):
        expected = resources.get(key)
        if expected is None:
            return f"charter resources is missing {key}"
        actual = _unit_value(lines, key)
        if actual is None:
            return f"containment unit is missing {key}"
        parser = {"MemoryMax": _memory_bytes, "CPUQuota": _cpu_percent,
                  "TasksMax": _tasks}[key]
        expected_value, actual_value = parser(expected), parser(actual)
        if expected_value is None or actual_value is None:
            return f"{key} must be a finite positive value"
        if expected_value != actual_value:
            return f"{key} mismatch: charter {expected!r} != unit {actual!r}"
    return None


def _check_components(charter):
    components = charter.get("components")
    if not isinstance(components, dict):
        return "charter.components must declare every selected-class component"
    if set(components) != set(REQUIRED_COMPONENTS):
        missing = sorted(set(REQUIRED_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(REQUIRED_COMPONENTS))
        return f"component set mismatch: missing={missing}, extra={extra}"
    if set(COMPONENT_EXERCISE_REGISTRY) != set(REQUIRED_COMPONENTS):
        return "component exercise registry keys do not match REQUIRED_COMPONENTS"
    for name in REQUIRED_COMPONENTS:
        if name not in components:
            return f"component {name!r} is missing"
        spec = components[name]
        if not isinstance(spec, dict):
            return f"component {name!r} must declare path and sha256"
        if set(spec) != {"path", "sha256"}:
            return (f"component {name!r} cannot declare an exercise receipt or "
                    "test implementation")
        path = spec.get("path")
        canonical = os.path.realpath(path) if isinstance(path, str) else None
        expected = os.path.realpath(os.path.join(HERE, COMPONENT_SOURCES[name]))
        if canonical != path:
            return f"component {name!r} path must be canonical: {path!r}"
        if canonical != expected:
            return (f"component {name!r} path is not the selected class source: "
                    f"{path!r} != {expected!r}")
        if not os.path.isfile(path):
            return f"component {name!r} is missing: {path!r}"
        claimed = spec.get("sha256")
        actual = file_digest(path)
        if claimed != actual:
            return f"component {name!r} digest does not match source"
        if name not in COMPONENT_EXERCISE_REGISTRY:
            return f"component exercise registry is missing {name!r}"
    return None


def _check_unenforced_risks(charter):
    risks = charter.get("unenforced_risks")
    if not isinstance(risks, list):
        return "unenforced_risks must declare the complete residual registry"
    ids = []
    for risk in risks:
        if not isinstance(risk, dict) or set(risk) != {"risk", "reason"} \
                or not isinstance(risk.get("risk"), str) or not risk["risk"].strip() \
                or not isinstance(risk.get("reason"), str) or not risk["reason"].strip():
            return "each unenforced_risks entry needs one nonempty risk and reason"
        ids.append(risk["risk"])
    if ids != list(ACCEPTED_RESIDUAL_IDS):
        return ("unenforced_risks must exactly declare canonical residual IDs: "
                f"missing={sorted(set(ACCEPTED_RESIDUAL_IDS) - set(ids))}, "
                f"extra={sorted(set(ids) - set(ACCEPTED_RESIDUAL_IDS))}")
    readme = charter.get("readme") or os.path.join(
        os.path.dirname(HERE), "README.md")
    try:
        text = open(readme).read()
    except OSError as exc:
        return f"README Not defended is unreadable: {exc}"
    start = text.find("## Not defended")
    if start < 0:
        return "README must document unenforced risks under Not defended"
    section = text[start:]
    next_heading = section.find("\n## ", len("## Not defended"))
    if next_heading >= 0:
        section = section[:next_heading]
    found = re.findall(r"^- \*\*([A-Za-z0-9_]+)\*\* — ([^\n]+)$",
                       section, flags=re.MULTILINE)
    expected = [(risk, RESIDUAL_REASONS[risk]) for risk in ACCEPTED_RESIDUAL_IDS]
    if found != expected:
        return "README Not defended residual registry does not exactly match charter"
    for risk in risks:
        if risk["reason"] != RESIDUAL_REASONS[risk["risk"]]:
            return (f"unenforced risk {risk['risk']!r} reason does not match the "
                    "canonical residual registry")
    return None


def check_unattended_entry(charter, run_dir):
    """Validate the static contract and the approval-owned execution record."""
    problem = check_approval_contract(charter, run_dir)
    if problem:
        return problem
    return _baseline_check(charter, run_dir)


def _check_subject_contract(charter):
    if charter.get("risk_level") not in ("L1", "L2"):
        return "safety risk_level must be exactly L1 or L2"
    scope = charter.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("allow"), list) \
            or not isinstance(scope.get("deny"), list):
        return "scope must be present with allow and deny lists"
    subject = charter.get("subject")
    if not isinstance(subject, str) or not subject or not os.path.exists(subject):
        return "subject must name an existing frozen path"
    if not os.path.isabs(subject) or os.path.realpath(subject) != subject:
        return "subject must be an existing canonical path"
    if charter.get("subject_frozen") is not True:
        return "subject must be explicitly frozen"
    if any(not isinstance(p, str) or not os.path.isabs(p)
           or os.path.realpath(p) != p for p in scope["allow"] + scope["deny"]):
        return "scope paths must be canonical absolute paths"
    if subject not in set(scope["deny"]):
        return "frozen subject must be in scope.deny"
    if any(_overlap(subject, p) for p in scope["allow"]):
        return "frozen subject overlaps scope.allow"
    writable = charter["containment"].get("writable_paths")
    if not isinstance(writable, list):
        return "containment.writable_paths must be declared"
    if any(not isinstance(p, str) or not os.path.isabs(p)
           or os.path.realpath(p) != p for p in writable):
        return "containment.writable_paths must be canonical absolute paths"
    if any(_overlap(subject, p) for p in writable):
        return "frozen subject overlaps a writable path"
    return None


def check_approval_contract(charter, run_dir):
    """Complete unattended approval contract, independent of the anchor.

    This function reads only the proposed charter and static source/unit files.
    It intentionally does not inspect an approval record or a live systemd
    process, so callers can run it before any approval-side durable write or
    approval-owned subprocess.
    """
    problem = check_schema(charter)
    if problem:
        return problem
    if charter.get("execution_mode") != "unattended":
        return "approval contract requires execution_mode=unattended"
    if charter["containment"].get("required") is not True:
        return "approval contract requires containment.required to be exactly True"
    problem = check_containment_evidence(charter, run_dir)
    if problem:
        return problem
    problem = _check_subject_contract(charter)
    if problem:
        return problem
    # The approval record does not exist yet.  Validate its instruction and
    # optional claim, not an observed result.
    measurement, baseline_problem = _measurement_spec(charter)
    if baseline_problem:
        return baseline_problem
    _, baseline_problem = _baseline_claim(charter)
    if baseline_problem:
        return baseline_problem
    if charter.get("model") is not None:
        return "evidence headless mode requires model=null, not a default or arbitrary model"
    route = charter.get("route")
    if route != {"mode": "headless", "primary": None, "degrade": "stop"}:
        return ("evidence headless mode requires the closed route "
                "{mode: headless, primary: null, degrade: stop}")
    effects = charter["containment"].get("external_effects")
    if not isinstance(effects, list) or not all(isinstance(effect, str) for effect in effects):
        return "external_effects must be declared"
    resource_problem = _resource_contract(charter)
    if resource_problem:
        return resource_problem
    component_problem = _check_components(charter)
    if component_problem:
        return component_problem
    risk_problem = _check_unenforced_risks(charter)
    if risk_problem:
        return risk_problem
    import source_manifest
    _, source_problem = source_manifest.validate_declarations(charter, run_dir)
    if source_problem:
        return f"source declaration refused: {source_problem}"
    return None


def path_digest(path):
    """Digest a file or directory deterministically, including relative names."""
    path = os.path.realpath(path)
    if os.path.isfile(path):
        return file_digest(path)
    if not os.path.isdir(path):
        raise OSError(f"subject is not a file or directory: {path}")
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        files.sort()
        rel_root = os.path.relpath(root, path)
        # Directories have to contribute by name, or an empty one is invisible:
        # it holds no files, so a file-only walk cannot tell it was created or
        # removed. A directory symlink is worse -- followlinks=False correctly
        # declines to recurse, but that also dropped it from the digest, so the
        # link could be swung at another tree without disturbing the freeze
        # check. Record the link target rather than what it points at.
        for name in dirs:
            full = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name))
            if os.path.islink(full):
                h.update(f"dirlink\0{rel}\0".encode())
                h.update(os.readlink(full).encode())
                h.update(b"\0")
            else:
                h.update(f"dir\0{rel}\0".encode())
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name))
            h.update(f"file\0{rel}\0".encode())
            h.update(file_digest(full).encode())
    return h.hexdigest()


def _measurement_spec(charter):
    spec = charter.get("measurement")
    if not isinstance(spec, dict):
        return None, "unattended baseline measurement instruction is missing"
    forbidden = {"result", "success", "receipt", "receipt_path", "sha256", "test_id",
                 "value", "count", "commit", "measured_at"}
    if forbidden.intersection(spec):
        return None, "baseline measurement instruction must not contain a result or receipt"
    required = {"argv", "cwd", "timeout_s", "parser", "source_paths"}
    if not required.issubset(spec):
        return None, "baseline measurement instruction is incomplete"
    argv = spec["argv"]
    if not isinstance(argv, list) or not argv or not all(
            isinstance(part, str) and part for part in argv):
        return None, "baseline measurement argv must be a non-empty list of strings"
    cwd = spec["cwd"]
    if not isinstance(cwd, str) or not os.path.isabs(cwd) \
            or os.path.realpath(cwd) != cwd or not os.path.isdir(cwd):
        return None, "baseline measurement cwd must be an existing canonical directory"
    timeout_problem = _finite_positive(spec["timeout_s"],
                                       "baseline measurement timeout_s")
    if timeout_problem:
        return None, timeout_problem
    driver_timeout = (charter.get("budgets") or {}).get("driver_timeout_s")
    if isinstance(driver_timeout, (int, float)) and spec["timeout_s"] > driver_timeout:
        return None, "baseline measurement timeout_s exceeds driver_timeout_s"
    parser = spec["parser"]
    if parser != {"id": BASELINE_PARSER_ID, "field": "value",
                  "count_field": "count"}:
        return None, "baseline measurement parser/field is not the fixed profile"
    sources = spec["source_paths"]
    if not isinstance(sources, list) or not sources:
        return None, "baseline measurement source_paths must be non-empty"
    for source in sources:
        if not isinstance(source, str) or not os.path.isabs(source) \
                or os.path.realpath(source) != source or not os.path.isfile(source):
            return None, "baseline measurement source_paths must be existing canonical files"
    if len(sources) != len(set(sources)):
        return None, "baseline measurement source_paths must not contain duplicates"
    import source_manifest
    target, problem = source_manifest._command_target(
        "baseline measurement", argv, cwd)
    if problem:
        return None, problem
    executable = shutil.which(argv[0]) if not os.path.isabs(argv[0]) else argv[0]
    executable = os.path.realpath(executable) if executable else None
    if not executable or not os.path.isfile(executable):
        return None, "baseline measurement executable is missing"
    if target not in sources:
        return None, "baseline measurement command target is not declared"
    return {**spec, "cwd": cwd, "executable": executable, "target": target}, None


def _baseline_claim(charter):
    claim = charter.get("baseline")
    if claim is None:
        return None, None
    if not isinstance(claim, dict):
        return None, "baseline claim must be a mapping"
    if any(key in claim for key in ("receipt", "result", "success", "test_id")):
        return None, "baseline measurement claim cannot contain a receipt or result"
    if set(claim) != {"value", "count"}:
        return None, "baseline measurement claim must contain only value and count"
    if not isinstance(claim["count"], int) or isinstance(claim["count"], bool) \
            or claim["count"] <= 0:
        return None, "baseline claim count must be a positive integer"
    return claim, None


def _approval_record(charter, run_dir):
    try:
        anchor = json.load(open(anchor_path(run_dir, charter["run_id"])))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"approval anchor is unreadable: {exc}"
    record = anchor.get("execution")
    if not isinstance(record, dict):
        return None, "authoritative approval execution record is missing from anchor"
    return record, None


def _valid_execution_time(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _execution_time_problem(record, label):
    for field in ("started_at", "finished_at", "started_mono", "finished_mono"):
        if not _valid_execution_time(record.get(field)):
            return f"{label} {field} is missing or not finite"
    if record["finished_at"] < record["started_at"] \
            or record["finished_mono"] < record["started_mono"]:
        return f"{label} execution timing is not ordered"
    return None


def _digest_map(paths):
    return {path: file_digest(path) for path in paths}


def _validate_execution_record(charter, record):
    if record.get("schema") != "goal-loop-approval/v1":
        return "authoritative approval execution record schema is unsupported"
    measurement, problem = _measurement_spec(charter)
    if problem:
        return problem
    baseline = record.get("baseline")
    if not isinstance(baseline, dict):
        return "authoritative baseline execution record is missing"
    if (baseline.get("profile_id") != BASELINE_PROFILE_ID
            or baseline.get("test_id") != BASELINE_PROFILE_ID
            or baseline.get("argv") != measurement["argv"]
            or baseline.get("cwd") != measurement["cwd"]
            or baseline.get("timeout_s") != measurement["timeout_s"]
            or baseline.get("source_paths") != measurement["source_paths"]
            or baseline.get("parser") != measurement["parser"]
            or baseline.get("return_code") != 0):
        return "authoritative baseline execution record is not bound"
    timing_problem = _execution_time_problem(baseline, "baseline")
    if timing_problem:
        return timing_problem
    executable = baseline.get("executable")
    if not isinstance(executable, dict) or executable.get("path") != measurement["executable"] \
            or executable.get("sha256") != file_digest(measurement["executable"]):
        return "authoritative baseline executable binding is not bound"
    if baseline.get("source_digests") != _digest_map(measurement["source_paths"]):
        return "authoritative baseline source binding is not bound"
    if any(not isinstance(baseline.get(key), str) or
           not re.fullmatch(r"[0-9a-f]{64}", baseline[key])
           for key in ("stdout_sha256", "stderr_sha256")):
        return "authoritative baseline output digests are missing"
    parsed = baseline.get("parsed_result")
    binding = baseline.get("value_count_binding")
    if not isinstance(parsed, dict) or not isinstance(binding, dict) \
            or {"value": parsed.get("value"), "count": parsed.get("count")} != binding:
        return "authoritative baseline parsed result is not bound"
    claim, problem = _baseline_claim(charter)
    if problem:
        return problem
    if claim is not None and binding != claim:
        return "observed baseline does not match the allowed baseline claim"
    rows = record.get("components")
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_COMPONENTS):
        return "authoritative component execution record must cover every component"
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(REQUIRED_COMPONENTS):
        return "authoritative component execution record component set mismatch"
    for name in REQUIRED_COMPONENTS:
        row = by_name[name]
        registered = COMPONENT_EXERCISE_REGISTRY[name]
        spec = charter["components"][name]
        expected = {"name": name, "exercise_id": registered["exercise_id"],
                    "profile_id": registered["profile_id"],
                    "implementation": registered["implementation"],
                    "test_id": registered["test_id"], "path": spec["path"],
                    "sha256": spec["sha256"], "return_code": 0,
                    "result": "success", "cwd": HERE,
                    "timeout_s": charter["budgets"]["max_trial_duration_s"],
                    "argv": _component_command(name, spec, registered)}
        if any(row.get(key) != value for key, value in expected.items()):
            return f"component {name!r} authoritative exercise record is not bound"
        timing_problem = _execution_time_problem(row, f"component {name!r}")
        if timing_problem:
            return timing_problem
        executable = row.get("executable")
        if executable != {"path": os.path.realpath(sys.executable),
                          "sha256": file_digest(sys.executable)}:
            return f"component {name!r} executable binding is not bound"
        source_paths = [os.path.realpath(__file__), spec["path"]]
        if row.get("source_paths") != source_paths:
            return f"component {name!r} source paths are not bound"
        if row.get("source_digests") != _digest_map(source_paths):
            return f"component {name!r} source digests are not bound"
        if row.get("fixed_source_digests") != {
                os.path.realpath(__file__): file_digest(__file__)}:
            return f"component {name!r} fixed source digests are not bound"
        if row.get("declared_source_digests") != {spec["path"]: spec["sha256"]}:
            return f"component {name!r} declared source digests are not bound"
        expected_output = {"name": name, "exercise_id": registered["exercise_id"],
                           "profile_id": registered["profile_id"],
                            "path": spec["path"], "sha256": spec["sha256"],
                            "result": "success",
                            "postcondition": registered["postcondition"]}
        if row.get("parsed_result") != expected_output:
            return f"component {name!r} authoritative exercise result is malformed"
        if any(not isinstance(row.get(key), str) or
               not re.fullmatch(r"[0-9a-f]{64}", row[key])
               for key in ("stdout_sha256", "stderr_sha256")):
            return f"component {name!r} authoritative output digests are missing"
        if row.get("component_digest_before") != spec["sha256"] \
                or row.get("component_digest_after") != spec["sha256"]:
            return f"component {name!r} component digest binding is not bound"
        subject_sha = record.get("before", {}).get("subject", {}).get("sha256")
        if not isinstance(subject_sha, str) \
                or row.get("subject_digest_before") != subject_sha \
                or row.get("subject_digest_after") != subject_sha:
            return f"component {name!r} subject digest binding is not bound"
    return None


def _baseline_check(charter, run_dir):
    spec, problem = _measurement_spec(charter)
    if problem:
        return problem
    _, problem = _baseline_claim(charter)
    if problem:
        return problem
    record, problem = _approval_record(charter, run_dir)
    if problem:
        return problem
    problem = _validate_execution_record(charter, record)
    if problem:
        return problem
    baseline = record.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("profile_id") != BASELINE_PROFILE_ID:
        return "authoritative baseline execution record is missing"
    if (baseline.get("argv") != spec["argv"]
            or baseline.get("cwd") != spec["cwd"]
            or baseline.get("timeout_s") != spec["timeout_s"]
            or baseline.get("source_paths") != spec["source_paths"]):
        return "stored baseline instruction binding does not match charter"
    if baseline.get("parser") != spec["parser"]:
        return "stored baseline parser binding does not match charter"
    if baseline.get("test_id") != BASELINE_PROFILE_ID:
        return "stored baseline test ID is not the fixed profile"
    executable = baseline.get("executable")
    if not isinstance(executable, dict) or executable.get("path") != spec["executable"] \
            or executable.get("sha256") != file_digest(spec["executable"]):
        return "stored baseline executable binding does not match"
    if baseline.get("source_digests") != _digest_map(spec["source_paths"]):
        return "stored baseline source binding does not match"
    claim, _ = _baseline_claim(charter)
    if claim is not None and baseline.get("value_count_binding") != claim:
        return "observed baseline does not match the allowed baseline claim"
    if baseline.get("return_code") != 0 or not isinstance(baseline.get("parsed_result"), dict):
        return "baseline execution did not succeed"
    return None


def _approval_sidecar_bindings(charter, run_dir):
    """Check that execution evidence is owned by the external anchor."""
    record, problem = _approval_record(charter, run_dir)
    if problem:
        return problem
    rows = record.get("components")
    if not isinstance(rows, list) or {row.get("name") for row in rows
                                      if isinstance(row, dict)} != set(REQUIRED_COMPONENTS):
        return "authoritative component execution record must cover every component"
    return None


def read_runtime_binding():
    """Return only kernel cgroup and systemd invocation identity observations."""
    with open("/proc/self/cgroup") as fh:
        cgroup = next((line.split("::", 1)[1].strip()
                       for line in fh if "::" in line), "")
    return {"cgroup": cgroup, "unit": os.environ.get("SYSTEMD_UNIT"),
            "invocation_id": os.environ.get("INVOCATION_ID"),
            "exec_pid": os.environ.get("SYSTEMD_EXEC_PID")}


def check_runtime_binding(charter, observed=None):
    """Require this process to be in the approved running systemd unit cgroup."""
    if charter.get("execution_mode") != "unattended":
        return None, {}
    unit = (charter.get("containment") or {}).get("unit")
    if not isinstance(unit, str) or os.path.realpath(unit) != unit:
        return "approved unit path is not canonical", None
    expected_unit = os.path.basename(unit)
    observed = read_runtime_binding() if observed is None else observed
    cgroup = observed.get("cgroup") if isinstance(observed, dict) else None
    if not isinstance(cgroup, str) or not cgroup.endswith("/" + expected_unit):
        return "current process is not inside the approved unit cgroup", observed
    if observed.get("unit") != expected_unit:
        return "systemd invocation identity does not name the approved unit", observed
    if not isinstance(observed.get("invocation_id"), str) or not observed["invocation_id"]:
        return "systemd invocation identity is missing", observed
    if observed.get("exec_pid") != str(os.getpid()):
        return "systemd invocation identity does not identify this process", observed
    return None, {"cgroup": cgroup, "unit": expected_unit,
                  "invocation_id": observed["invocation_id"],
                  "exec_pid": observed["exec_pid"]}


def check_containment(charter, supervised=False):
    """Probe the boundary instead of trusting the unit file.

    A unit that reads correctly is not evidence that it is loaded. This is one
    write outside the permitted set: cheap, and the only claim worth making
    here. It is deliberately not the full negative test, which additionally
    needs an uncontained control run to mean anything.
    """
    containment = charter["containment"]

    if not containment.get("required", True):
        # The execution mode check above is bound to the distinct entry path.
        if not supervised:
            return ("containment.required is false, but this process was not "
                    "started in supervised mode; an unattended run may not "
                    "waive containment")
        return None

    target_reason, targets = canonical_probe_targets(containment)
    if target_reason:
        return target_reason
    import negative_test
    live = negative_test.run_probes(targets)
    escaped = [name for name, r in live.items() if r != "blocked"]
    if escaped:
        return (f"containment is not active: probes not blocked here and now: "
                f"{escaped}")
    return None


def _inside(path, parent):
    """Component-safe containment test for canonical absolute paths."""
    return path == parent or path.startswith(parent + os.sep)


def canonical_probe_targets(containment):
    """Validate and canonicalize the two filesystem probe targets."""
    writable = containment.get("writable_paths")
    if not isinstance(writable, list) or not writable:
        return "containment.writable_paths must be a non-empty list", None
    if not all(isinstance(path, str) and os.path.isabs(path) for path in writable):
        return "containment.writable_paths must contain absolute string paths", None
    writable_canonical = [os.path.realpath(path) for path in writable]

    targets = containment.get("probe_targets")
    expected = {CANONICAL_HOME_PROBE, CANONICAL_REPO_PROBE}
    if not isinstance(targets, dict) or set(targets) != expected:
        return (f"containment.probe_targets must have exactly the keys "
                f"{sorted(expected)}"), None
    if not all(isinstance(path, str) and os.path.isabs(path)
               for path in targets.values()):
        return "containment.probe_targets must contain absolute string paths", None

    canonical = {}
    for name, path in targets.items():
        resolved = os.path.realpath(path)
        for writable_path in writable_canonical:
            if _inside(resolved, writable_path):
                return (f"probe target {name!r} resolves inside writable path "
                        f"{writable_path}"), None
        if path != resolved:
            return (f"probe target {name!r} must be canonical: {path!r} "
                    f"!= {resolved!r}"), None
        canonical[name] = resolved
    return None, canonical


def validate_probe_targets(containment):
    return canonical_probe_targets(containment)[0]


def file_digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_source_binding(charter, run_dir):
    """Verify the approved declared command-source set and every listed digest."""
    if charter.get("execution_mode") != "unattended":
        return None
    import source_manifest
    manifest_path, commit_path, problem = source_manifest.configured_paths(charter, run_dir)
    if problem:
        return problem
    anchor = anchor_path(run_dir, charter["run_id"])
    try:
        anchor_value = json.load(open(anchor))
    except (OSError, json.JSONDecodeError) as exc:
        return f"source binding anchor is unreadable: {exc}"
    binding = anchor_value.get("source_binding")
    if not isinstance(binding, dict):
        return "source binding is missing from the approval anchor"
    if binding.get("manifest_path") != manifest_path or binding.get("commit_path") != commit_path:
        return "source binding anchor paths do not match the charter"
    try:
        actual_manifest_digest = file_digest(manifest_path)
        actual_commit_digest = file_digest(commit_path)
    except OSError as exc:
        return f"approved source binding file is missing: {exc}"
    if binding.get("manifest_sha256") != actual_manifest_digest:
        return "source manifest changed after approval"
    if binding.get("commit_sha256") != actual_commit_digest:
        return "commit.txt changed after approval"
    try:
        manifest_bytes = open(manifest_path, "rb").read()
        manifest = json.loads(manifest_bytes.decode())
        commit = open(commit_path).read()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"source binding is unreadable: {exc}"
    if manifest_bytes != source_manifest.canonical_json(manifest):
        return "source manifest is not canonical"
    if manifest.get("format") != source_manifest.FORMAT:
        return "source manifest format is not supported"
    if manifest.get("run_id") != charter["run_id"] \
            or manifest.get("manifest_path") != manifest_path \
            or manifest.get("commit_path") != commit_path:
        return "source manifest identity does not match the approved charter"
    if commit != manifest.get("commit", "") + "\n":
        return "commit.txt does not match the source manifest commit"
    expected, declaration_problem = source_manifest.validate_declarations(charter, run_dir)
    if declaration_problem:
        return f"source declaration refused: {declaration_problem}"
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return "source manifest must contain a non-empty finite file set"
    seen = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) \
                or not isinstance(entry.get("sha256"), str):
            return "source manifest entries require path and sha256"
        path = entry["path"]
        if not os.path.isabs(path) or os.path.realpath(path) != path:
            return "source manifest paths must be canonical absolute paths"
        seen.append(path)
    if len(seen) != len(set(seen)):
        return "source manifest contains duplicate paths"
    for entry in entries:
        try:
            actual = file_digest(entry["path"])
        except OSError as exc:
            return f"listed source is missing: {entry['path']}: {exc}"
        if actual != entry["sha256"]:
            return f"listed source changed after approval: {entry['path']}"
    if set(seen) != expected:
        missing, extra = sorted(expected - set(seen)), sorted(set(seen) - expected)
        return f"source manifest source set mismatch: missing={missing}, extra={extra}"
    if manifest.get("git_path") != source_manifest.GIT_PATH \
            or manifest.get("git_sha256") != source_manifest.GIT_SHA256:
        return "source manifest git binding is not approved"
    if manifest.get("commit") != source_manifest.current_commit(HERE):
        return "source manifest commit no longer matches HEAD"
    return check_subject_binding(charter, run_dir)


def check_subject_binding(charter, run_dir):
    """Reverify the approved subject before any result can be accepted."""
    if charter.get("execution_mode") != "unattended":
        return None
    path = os.path.realpath(charter.get("subject", ""))
    anchor = anchor_path(run_dir, charter["run_id"])
    try:
        binding = json.load(open(anchor)).get("subject_binding")
    except (OSError, json.JSONDecodeError) as exc:
        return f"subject binding is unreadable: {exc}"
    if not isinstance(binding, dict) or binding.get("path") != path:
        return "subject binding is missing or does not match the charter"
    try:
        actual = path_digest(path)
    except OSError as exc:
        return f"subject is unreadable after approval: {exc}"
    if binding.get("sha256") != actual:
        return "subject changed after approval"
    return None


def check_unit_policy(unit, writable_paths):
    """Check only the unit properties needed by this containment evidence."""
    try:
        lines = open(unit).read().splitlines()
    except OSError as exc:
        return f"containment unit is unreadable: {exc}"
    if "RestrictNamespaces=yes" not in lines:
        return "containment unit must set RestrictNamespaces=yes"
    starts = [line for line in lines if line.startswith("ExecStart=")]
    if len(starts) != 1 or "/loop/run.sh " not in starts[0]:
        return "containment unit must use the default unattended loop/run.sh entry"
    if "--supervised" in starts[0]:
        return "containment unit must not invoke the supervised entry"
    if "Environment=GOAL_LOOP_SUPERVISED=" not in lines:
        return "containment unit must clear GOAL_LOOP_SUPERVISED"
    recorded = []
    for line in lines:
        if line.startswith("ReadWritePaths="):
            recorded.extend(line.partition("=")[2].split())
    actual = {os.path.realpath(path) for path in recorded}
    expected = {os.path.realpath(path) for path in writable_paths}
    if actual != expected:
        return (f"containment unit ReadWritePaths {sorted(actual)} does not match "
                f"charter writable_paths {sorted(expected)}")
    return None


def anchor_path(run_dir, run_id):
    """The anchor is outside the run directory so directory replacement cannot
    replace both the ledger and the identity it is bound to."""
    filename = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()
    return os.path.join(os.path.dirname(os.path.abspath(run_dir)),
                        f"{filename}.anchor")


def _sync_directory(path):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, data, mode=0o644):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("write returned no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync_directory(path)


def _existing_anchor_matches(anchor, run_id, ledger_path):
    try:
        value = json.load(open(anchor))
        stat = os.stat(ledger_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (value.get("run_id") == run_id
            and value.get("ledger") == {"st_dev": stat.st_dev,
                                         "st_ino": stat.st_ino})


def _approval_temp_paths(anchor):
    parent = os.path.dirname(anchor)
    prefix = os.path.basename(anchor) + ".tmp-"
    try:
        return [os.path.join(parent, name) for name in os.listdir(parent)
                if name.startswith(prefix)]
    except OSError:
        return []


def _relevant_tree_snapshot(roots, ignored_control_roots=()):
    """Capture every entry in the subject and control trees, not fixed paths."""
    entries = {}
    for root in sorted(set(os.path.realpath(path) for path in roots if path)):
        if not os.path.lexists(root):
            entries[root] = {"kind": "missing"}
            continue
        if os.path.islink(root):
            entries[root] = {"kind": "symlink", "target": os.readlink(root)}
            continue
        if os.path.isfile(root):
            entries[root] = {"kind": "file", "sha256": file_digest(root)}
            continue
        if not os.path.isdir(root):
            entries[root] = {"kind": "other"}
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            files.sort()
            if current == root and root in ignored_control_roots:
                result_dir = os.path.join(root, "results")
                if os.path.isdir(result_dir):
                    entries[result_dir] = {"kind": "dir"}
                dirs[:] = [name for name in dirs if name != "results"]
                files = [name for name in files
                         if name not in ("charter.sha256", "iterations.jsonl")]
            for name in dirs:
                path = os.path.join(current, name)
                if os.path.islink(path):
                    entries[path] = {"kind": "symlink", "target": os.readlink(path)}
                else:
                    entries[path] = {"kind": "dir"}
            for name in files:
                path = os.path.join(current, name)
                entries[path] = {"kind": "file", "sha256": file_digest(path)}
    return entries


def _snapshot(charter, measurement):
    import source_manifest
    sources = [measurement["executable"], *measurement["source_paths"]]
    declared, problem = source_manifest._command_declarations(charter)
    if problem:
        raise RuntimeError(f"source declaration refused: {problem}")
    sources.extend(declared)
    return {
        "commit": source_manifest.current_commit(HERE),
        "subject": {"path": os.path.realpath(charter["subject"]),
                    "sha256": path_digest(charter["subject"])},
        "components": {
            name: {"path": spec["path"], "sha256": file_digest(spec["path"])}
            for name, spec in charter["components"].items()
        },
        "sources": {path: file_digest(path) for path in sorted(set(sources))},
        "relevant_tree": _relevant_tree_snapshot(
            [charter["subject"], measurement.get("_control_plane")],
            (os.path.realpath(measurement["_control_plane"]),)
            if measurement.get("_ignore_control_artifacts") else ()),
    }


def _kill_process_group(proc):
    """Kill and reap a timed-out baseline's complete session."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_baseline(measurement, claim):
    started_at, started_mono = time.time(), time.monotonic()
    scratch = tempfile.TemporaryDirectory(prefix="goal-loop-baseline-")
    proc = None
    try:
        proc = subprocess.Popen(measurement["argv"], cwd=scratch.name,
                                shell=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=measurement["timeout_s"])
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        raise RuntimeError("baseline measurement timed out") from exc
    except OSError as exc:
        if proc is not None:
            _kill_process_group(proc)
        raise RuntimeError(f"baseline measurement could not execute: {exc}") from exc
    finally:
        scratch.cleanup()
    finished_at, finished_mono = time.time(), time.monotonic()
    stdout, stderr = stdout or b"", stderr or b""
    if proc.returncode != 0:
        raise RuntimeError(f"baseline measurement returned nonzero: {proc.returncode}")
    try:
        lines = stdout.decode("utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("expected one JSON line")
        parsed = json.loads(lines[0])
        value, count = parsed["value"], parsed["count"]
    except (UnicodeDecodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"baseline measurement output is malformed: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(count, int) \
            or isinstance(count, bool) or count <= 0:
        raise RuntimeError("baseline measurement parsed count is invalid")
    observed = {"value": value, "count": count}
    if claim is not None and observed != claim:
        raise RuntimeError("observed baseline does not match the allowed baseline claim")
    return {
        "profile_id": BASELINE_PROFILE_ID, "parser": measurement["parser"],
        "argv": measurement["argv"], "cwd": measurement["cwd"],
        "timeout_s": measurement["timeout_s"],
        "source_paths": measurement["source_paths"],
        "executable": {"path": measurement["executable"],
                        "sha256": file_digest(measurement["executable"])},
        "source_digests": _digest_map(measurement["source_paths"]),
        "started_at": started_at, "finished_at": finished_at,
        "started_mono": started_mono, "finished_mono": finished_mono,
        "return_code": proc.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "parsed_result": parsed, "value_count_binding": observed,
        "test_id": BASELINE_PROFILE_ID,
    }


def _fixed_component_checks(name):
    """Return a code-owned behavioral check for one selected component."""
    if name == "containment":
        import preflight as component
        assert component._memory_bytes("512M") == 512 * 1024 * 1024
        assert component._cpu_percent("50%") == 50
        assert component._tasks("64") == 64
        assert component._cpu_percent("0%") is None
        return {"invariant": "resource-parser-refuses-invalid"}
    if name == "lease":
        import ledger as component
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger")
            anchor = os.path.join(directory, "anchor")
            open(path, "wb").close()
            stat = os.stat(path)
            with open(anchor, "w") as fh:
                json.dump({"run_id": "exercise", "ledger": {
                    "st_dev": stat.st_dev, "st_ino": stat.st_ino}}, fh)
            record = {"event": "run_start", "at": "exercise", "mono": 1,
                      "boot": "exercise"}
            with component.locked(path, anchor, "exercise") as fd:
                written = component.append_fd(fd, path, record)
            assert written["seq"] == 1 and component.read(path)[0]["seq"] == 1
        return {"invariant": "locked-ledger-append-read", "sequence": 1}
    if name == "collector":
        import collector as component
        with tempfile.TemporaryDirectory() as directory:
            result = component.run_trial(
                [sys.executable, "-c", "print('component-ok')"], directory,
                7, 2)
        assert result["status"] == "ok" and result["passed"] is True
        return {"invariant": "ok-process-result", "status": "ok"}
    if name == "attempt_id":
        import iterate as component
        charter = {"budgets": {"max_wallclock_s": 60,
                                "max_iteration_duration_s": 10,
                                "max_trial_duration_s": 5,
                                "termination_duration_s": 1}}
        trial, outer = component.trial_deadlines(charter, 0, time.monotonic())
        assert outer > trial and outer - trial == 1
        return {"invariant": "deadline-reserves-termination", "reserve_s": 1}
    if name == "evaluator":
        import evaluator as component
        rule = {"min_trials_per_arm": 1, "max_trials": 4, "ci_width": 1.0}
        trials = [{"arm": "a", "status": "ok", "passed": True},
                  {"arm": "b", "status": "ok", "passed": True}]
        stopped, _, _ = component.should_stop(trials, rule, ["a", "b"])
        assert component.arm_for(2, ["a", "b"]) == "a" and stopped
        return {"invariant": "stopping-rule-interval", "stopped": True}
    if name == "ledger_writer":
        import ledger as component
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger")
            open(path, "wb").close()
            record = {"event": "run_start", "at": "exercise", "mono": 1,
                      "boot": "exercise"}
            written = component.append(path, record)
            assert written["seq"] == 1 and component.read(path)[0]["event"] == "run_start"
        return {"invariant": "fsync-append-read", "sequence": 1}
    raise RuntimeError(f"no fixed component exercise for {name!r}")


def _component_command(name, spec, registry):
    return [sys.executable, os.path.realpath(__file__), "--exercise-component",
            name, spec["path"], spec["sha256"], registry["exercise_id"],
            registry["profile_id"]]


def _run_component_exercise(name, spec, charter):
    registry = COMPONENT_EXERCISE_REGISTRY.get(name)
    if not isinstance(registry, dict):
        raise RuntimeError(f"component exercise registry is missing {name!r}")
    command = _component_command(name, spec, registry)
    component_before = file_digest(spec["path"])
    subject_before = path_digest(charter["subject"])
    started_at, started_mono = time.time(), time.monotonic()
    try:
        result = subprocess.run(command, cwd=HERE, shell=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=charter["budgets"]["max_trial_duration_s"])
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"component {name!r} exercise timed out") from exc
    except OSError as exc:
        raise RuntimeError(f"component {name!r} exercise could not execute: {exc}") from exc
    finished_at, finished_mono = time.time(), time.monotonic()
    stdout, stderr = result.stdout or b"", result.stderr or b""
    if result.returncode != 0:
        raise RuntimeError(f"component {name!r} exercise returned nonzero: {result.returncode}")
    try:
        lines = stdout.decode("utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("expected one JSON line")
        parsed = json.loads(lines[0])
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"component {name!r} exercise output is malformed: {exc}") from exc
    expected = {"name": name, "exercise_id": registry["exercise_id"],
                "profile_id": registry["profile_id"], "path": spec["path"],
                "sha256": spec["sha256"], "result": "success",
                "postcondition": registry["postcondition"]}
    if parsed != expected:
        raise RuntimeError(f"component {name!r} exercise postcondition failed")
    component_after = file_digest(spec["path"])
    subject_after = path_digest(charter["subject"])
    if component_before != component_after or subject_before != subject_after:
        raise RuntimeError("approval source, component, or subject digest drifted during execution")
    source_paths = [os.path.realpath(__file__), spec["path"]]
    return {"name": name, "exercise_id": registry["exercise_id"],
            "profile_id": registry["profile_id"],
            "test_id": registry["test_id"],
            "implementation": registry["implementation"], "path": spec["path"],
            "sha256": spec["sha256"], "argv": command,
            "cwd": HERE, "timeout_s": charter["budgets"]["max_trial_duration_s"],
            "executable": {"path": os.path.realpath(sys.executable),
                           "sha256": file_digest(sys.executable)},
            "source_paths": source_paths,
            "source_digests": _digest_map(source_paths),
            "fixed_source_digests": {os.path.realpath(__file__): file_digest(__file__)},
            "declared_source_digests": {spec["path"]: spec["sha256"]},
            "return_code": result.returncode,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "parsed_result": parsed, "result": "success",
            "started_at": started_at, "finished_at": finished_at,
            "started_mono": started_mono, "finished_mono": finished_mono,
            "component_digest_before": component_before,
            "component_digest_after": component_after,
            "subject_digest_before": subject_before,
            "subject_digest_after": subject_after}


def _verify_snapshot(before, after):
    if before != after:
        raise RuntimeError("approval source, component, or subject digest drifted during execution")


def _verify_stored_approval(charter, run_dir, digest, anchor, ledger_path,
                            digest_path, source_paths):
    if not _existing_anchor_matches(anchor, charter["run_id"], ledger_path):
        raise RuntimeError("approval bootstrap identity is corrupt; refusing repair")
    try:
        if open(digest_path).read().strip() != digest:
            raise RuntimeError("charter changed after approval; refusing re-approval")
        value = json.load(open(anchor))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"approval record is unreadable: {exc}") from exc
    record = value.get("execution")
    if not isinstance(record, dict):
        raise RuntimeError("authoritative approval execution record is missing")
    measurement, problem = _measurement_spec(charter)
    if problem:
        raise RuntimeError(problem)
    measurement["_control_plane"] = run_dir
    measurement["_ignore_control_artifacts"] = True
    current = _snapshot(charter, measurement)
    if record.get("before") != record.get("after") or record.get("after") != current:
        raise RuntimeError("approval source, component, or subject digest drifted after approval")
    problem = _validate_execution_record(charter, record)
    if problem:
        raise RuntimeError(problem)
    binding = value.get("source_binding")
    if not isinstance(binding, dict) or binding.get("manifest_path") != source_paths[0] \
            or binding.get("commit_path") != source_paths[1]:
        raise RuntimeError("source approval binding is corrupt; refusing repair")
    try:
        if binding["manifest_sha256"] != file_digest(source_paths[0]) \
                or binding["commit_sha256"] != file_digest(source_paths[1]):
            raise RuntimeError("approved source binding changed after approval")
    except (KeyError, OSError) as exc:
        raise RuntimeError("approved source binding is unreadable") from exc
    return digest


def check_containment_evidence(charter, run_dir):
    """Tie the containment claim to the thing that was actually tested.

    A negative test says something about the unit that was loaded when it ran,
    and only if the same probes succeeded without containment first. Recording
    the result without binding it to the unit, or without the control, leaves a
    number that reads like evidence and is not.
    """
    containment = charter["containment"]
    if not containment.get("required", True):
        return None

    target_reason, targets = canonical_probe_targets(containment)
    if target_reason:
        return target_reason

    # The control plane has to outlive the subject being restored or removed,
    # so it cannot live inside it.
    plane = os.path.realpath(run_dir)
    for name, spec in (charter.get("arms") or {}).items():
        subject = os.path.realpath(spec.get("cwd", ""))
        if subject and (plane == subject or plane.startswith(subject + os.sep)):
            return (f"the control plane {plane} lies inside arm {name!r}'s subject "
                    f"{subject}; restoring the subject would destroy the record")

    unit = containment.get("unit")
    recorded = containment.get("unit_sha256")
    if not isinstance(unit, str) or not os.path.isabs(unit) \
            or os.path.realpath(unit) != unit:
        return "containment unit path must be canonical"
    if not isinstance(recorded, str) or not recorded:
        return "containment.required is set but no unit and digest are recorded"
    if not os.path.exists(unit):
        return f"containment unit is missing: {unit}"
    if not os.path.isfile(unit):
        return "containment unit must be a canonical regular file"
    if file_digest(unit) != recorded:
        return ("the containment unit changed after its negative test; "
                "the evidence describes a unit that is no longer in force")
    policy = check_unit_policy(unit, containment["writable_paths"])
    if policy:
        return policy

    receipt, receipt_problem = _read_containment_receipt(charter, targets)
    if receipt_problem:
        return receipt_problem
    test = containment["negative_test"]
    verified_at = containment.get("verified_at")
    if verified_at != receipt["at"]:
        return ("containment.verified_at and receipt.at must be the same "
                "non-empty string")
    # Inline outcomes are compatibility input only. They must agree with the
    # bytes read above; they can never replace the receipt or mint an outcome.
    inline = {key: test[key] for key in ("at", "targets", "probes") if key in test}
    expected_inline = {"at": receipt["at"], "targets": receipt["targets"],
                       "probes": receipt["probes"]}
    if inline and inline != {key: expected_inline[key] for key in inline}:
        return "inline containment outcomes do not match the bound receipt"
    # Bind to the actual probe file, not two supplied strings that only have to
    # equal each other. Hash the implementation that would run, and require both
    # the charter and receipt to name that same digest.
    source = containment.get("probe_source")
    if not isinstance(source, str) or os.path.realpath(source) != source:
        return "containment.probe_source must be a canonical path"
    if source != CANONICAL_PROBE_SOURCE:
        return ("containment.probe_source must bind the canonical negative_test.py "
                "runtime implementation")
    if not os.path.isfile(source):
        return "containment.probe_source must name the probe implementation on disk"
    actual = file_digest(source)
    for where, claimed in (("charter", containment.get("probe_source_sha256")),
                           ("receipt", receipt.get("probe_source_sha256"))):
        if claimed != actual:
            return (f"{where} probe_source_sha256 does not match the file on disk; "
                    "the recorded evidence describes a different probe")
    anchor = anchor_path(run_dir, charter["run_id"])
    if os.path.lexists(anchor):
        try:
            anchor_value = json.load(open(anchor))
        except (OSError, json.JSONDecodeError) as exc:
            return f"containment receipt anchor is unreadable: {exc}"
        if anchor_value.get("containment_receipt") != receipt:
            return "containment receipt is not bound to the approval anchor"
    return None


def _read_containment_receipt(charter, targets):
    """Read the operator-supplied receipt and return its exact-byte binding."""
    test = (charter.get("containment") or {}).get("negative_test")
    if not isinstance(test, dict):
        return None, "negative_test receipt declaration must be a mapping"
    path = test.get("receipt_path")
    if not isinstance(path, str) or not os.path.isabs(path) \
            or os.path.realpath(path) != path:
        return None, "negative_test.receipt_path must be a canonical path"
    if not os.path.isfile(path):
        return None, f"containment receipt is missing: {path}"
    try:
        raw = open(path, "rb").read()
    except OSError as exc:
        return None, f"containment receipt is unreadable: {exc}"
    # Binding the bytes into the anchor stops the receipt changing after
    # approval. It does not stop a different receipt being in place *at*
    # approval: whatever sits at receipt_path is what gets bound, and without
    # this the charter never said which file it meant. Check the declaration
    # before parsing, so a swapped receipt is refused as a swap rather than as
    # whatever its contents happen to fail on.
    declared = test.get("receipt_sha256")
    if not isinstance(declared, str) or not declared.strip():
        return None, ("negative_test.receipt_sha256 must declare the expected "
                      "receipt digest")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != declared:
        return None, ("containment receipt digest does not match the charter "
                      f"declaration ({actual} != {declared})")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"containment receipt is unreadable: {exc}"
    if not isinstance(value, dict) or value.get("schema") != CONTAINMENT_RECEIPT_SCHEMA:
        return None, "containment receipt schema is unsupported"
    at = value.get("at")
    if not isinstance(at, str) or not at.strip():
        return None, "containment receipt at must be non-empty"
    if value.get("targets") != targets:
        return None, "containment receipt targets do not match the charter"
    probes = value.get("probes")
    if not isinstance(probes, dict) or set(probes) != CANONICAL_PROBE_NAMES:
        return None, ("containment receipt probe set does not match the canonical "
                      "four probes")
    for name, outcome in probes.items():
        if not isinstance(outcome, dict):
            return None, f"containment receipt probe {name!r} is not a mapping"
        if outcome.get("control") != "succeeded":
            return None, f"receipt probe {name!r} did not succeed uncontained"
        if outcome.get("contained") != "blocked":
            return None, f"receipt probe {name!r} was not blocked under containment"
    source_digest = value.get("probe_source_sha256")
    if not isinstance(source_digest, str):
        return None, "containment receipt probe source digest is missing"
    provenance = value.get("provenance")
    contained_provenance = (provenance.get("contained")
                            if isinstance(provenance, dict) else None)
    if (not isinstance(contained_provenance, dict)
            or not all(isinstance(contained_provenance.get(field), str)
                       and contained_provenance[field].strip()
                       for field in ("cgroup", "invocation_id", "unit"))):
        return None, ("containment receipt contained execution provenance is "
                      "missing or incomplete")
    # Three non-empty strings are not a binding. The kernel mints the cgroup
    # path and it ends in the unit that owns the process, so a cgroup and a unit
    # that disagree describe two different things and neither can be trusted as
    # the environment the contained probes ran in.
    if not contained_provenance["cgroup"].endswith(
            "/" + contained_provenance["unit"]):
        return None, ("containment receipt contained cgroup "
                      f"{contained_provenance['cgroup']!r} does not name its "
                      f"unit {contained_provenance['unit']!r}")
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
            "schema": value["schema"], "at": at, "targets": targets,
            "probes": probes, "provenance": provenance,
            "probe_source_sha256": source_digest}, None


def validate_charter(charter, run_dir):
    """Checks that read files and compute. Safe on a stopped run.

    Separated from the boundary probe because a stopped run still needs its
    charter validated — `finish()` executes `notify.command` — while it must not
    act on the world, and because a containment failure must never be able to
    trap a run that owes a notification.
    """
    return check_schema(charter) or check_digest(charter, run_dir)


def run_static(charter, run_dir, supervised=False):
    """Run every entry check that reads files but does not probe live state."""
    return (validate_charter(charter, run_dir)
            or check_execution_mode(charter, supervised=supervised)
            or (check_unattended_entry(charter, run_dir)
                if not supervised else None)
            or (check_containment_evidence(charter, run_dir)
                if supervised else None))


def run_live(charter, supervised=False):
    """Run the live boundary probe after static checks and identity binding."""
    return check_containment(charter, supervised=supervised)


def run(charter, run_dir, supervised=False):
    """Full entry gate, with runtime identity checked before live probing."""
    refusal = run_static(charter, run_dir, supervised=supervised)
    if refusal:
        return refusal
    binding_refusal, _ = check_runtime_binding(charter)
    if binding_refusal:
        return binding_refusal
    return run_live(charter, supervised=supervised)


def _supervised_approve(charter_path):
    """Perform the one durable bootstrap, in ledger-then-anchor order.

    A partial bootstrap is deliberately not repaired. Cross-directory atomicity
    is unavailable, so ledger-without-anchor and anchor-without-ledger are both
    unrecoverable and require a human decision.
    """
    charter = json.load(open(charter_path))
    digest = charter_digest(charter)
    run_dir = os.path.dirname(os.path.abspath(charter_path))
    ledger_path = os.path.join(run_dir, "iterations.jsonl")
    anchor = anchor_path(run_dir, charter["run_id"])
    digest_path = os.path.join(run_dir, "charter.sha256")
    source = None
    subject_binding = None
    if charter.get("execution_mode") == "unattended":
        subject = charter.get("subject")
        if charter.get("subject_frozen") is True and (
                not isinstance(subject, str) or not os.path.exists(subject)):
            raise RuntimeError("cannot bind missing unattended subject")
        if isinstance(subject, str) and os.path.exists(subject):
            subject = os.path.realpath(subject)
            subject_binding = {"path": subject, "sha256": path_digest(subject)}
        import source_manifest
        manifest_path, commit_path, problem = source_manifest.configured_paths(
            charter, run_dir)
        if problem:
            raise RuntimeError(f"source approval refused: {problem}")
        source = (manifest_path, commit_path,
                  source_manifest.build(charter, run_dir, manifest_path, commit_path))
    ledger_exists = os.path.lexists(ledger_path)
    anchor_exists = os.path.lexists(anchor)
    source_exists = (source and (os.path.lexists(source[0]) or os.path.lexists(source[1])))
    if ledger_exists or anchor_exists:
        if not (ledger_exists and anchor_exists):
            raise RuntimeError(
                "approval bootstrap is partial and unrecoverable; refusing repair")
        if source_exists and not (os.path.lexists(source[0]) and os.path.lexists(source[1])):
            raise RuntimeError(
                "source approval bootstrap is partial and unrecoverable; refusing repair")
        if not _existing_anchor_matches(anchor, charter["run_id"], ledger_path):
            raise RuntimeError(
                "approval bootstrap identity is corrupt; refusing repair")
        try:
            if open(digest_path).read().strip() != digest:
                raise RuntimeError("charter changed after approval; refusing re-approval")
        except FileNotFoundError:
            raise RuntimeError("approval digest is missing; refusing repair") from None
        if source:
            binding = json.load(open(anchor)).get("source_binding")
            if not isinstance(binding, dict) \
                    or binding.get("manifest_path") != source[0] \
                    or binding.get("commit_path") != source[1] \
                    or binding.get("manifest_sha256") != file_digest(source[0]) \
                    or binding.get("commit_sha256") != file_digest(source[1]):
                raise RuntimeError("source approval binding is corrupt; refusing repair")
        if subject_binding and json.load(open(anchor)).get("subject_binding") != subject_binding:
            raise RuntimeError("subject approval binding is corrupt; refusing repair")
        return digest

    _write_new(digest_path, (digest + "\n").encode())
    # Safe crash interpretation: this file is durable before the anchor is
    # published. If publication fails, continuation must fail closed.
    _write_new(ledger_path, b"")
    stat = os.stat(ledger_path)
    payload_value = {"run_id": charter["run_id"],
                     "ledger": {"st_dev": stat.st_dev, "st_ino": stat.st_ino}}
    if subject_binding:
        payload_value["subject_binding"] = subject_binding
    payload = json.dumps(payload_value,
                         sort_keys=True).encode() + b"\n"
    temporary = f"{anchor}.tmp-{os.getpid()}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("write returned no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, anchor)
    _sync_directory(anchor)
    if source:
        manifest_path, commit_path, manifest = source
        _write_new(commit_path, (manifest["commit"] + "\n").encode())
        _write_new(manifest_path, json.dumps(manifest, sort_keys=True,
                                             separators=(",", ":")).encode() + b"\n")
        # The anchor is the identity authority, so publish the two source
        # digests in it before continuation can pass the B-6 gate. Replacing
        # the anchor inode is safe here: no continuation lock exists during
        # approval and the ledger identity remains the same.
        value = json.load(open(anchor))
        value["source_binding"] = {
            "manifest_path": manifest_path,
            "manifest_sha256": file_digest(manifest_path),
            "commit_path": commit_path,
            "commit_sha256": file_digest(commit_path),
        }
        baseline_receipt = (charter.get("baseline") or {}).get("receipt")
        if isinstance(baseline_receipt, dict):
            value["baseline_receipt_binding"] = {
                "path": baseline_receipt.get("path"),
                "sha256": baseline_receipt.get("sha256"),
            }
        component_evidence = charter.get("component_exercises")
        if isinstance(component_evidence, dict):
            value["component_exercise_binding"] = {
                "path": component_evidence.get("receipt_path"),
                "sha256": component_evidence.get("sha256"),
            }
        temporary = f"{anchor}.tmp-{os.getpid()}-bound"
        _write_new(temporary, json.dumps(value, sort_keys=True).encode() + b"\n")
        os.replace(temporary, anchor)
        _sync_directory(anchor)
    return digest


def approve(charter_path):
    """Approve supervised charters unchanged; execute unattended gates first."""
    charter = json.load(open(charter_path))
    if charter.get("execution_mode") != "unattended":
        return _supervised_approve(charter_path)
    run_dir = os.path.dirname(os.path.abspath(charter_path))
    # This is deliberately before digest/anchor/ledger/manifest existence
    # checks, baseline execution, component exercises, and every _write_new.
    # It cannot consult an anchor or a live systemd bus.
    problem = check_approval_contract(charter, run_dir)
    if problem:
        raise RuntimeError(problem)
    digest = charter_digest(charter)
    ledger_path = os.path.join(run_dir, "iterations.jsonl")
    anchor = anchor_path(run_dir, charter["run_id"])
    digest_path = os.path.join(run_dir, "charter.sha256")
    import source_manifest
    manifest_path, commit_path, problem = source_manifest.configured_paths(charter, run_dir)
    if problem:
        raise RuntimeError(f"source approval refused: {problem}")
    approval_paths = [digest_path, ledger_path, anchor, manifest_path, commit_path]
    existing = [path for path in approval_paths if os.path.lexists(path)]
    existing.extend(_approval_temp_paths(anchor))
    if existing:
        if (not all(os.path.lexists(path) for path in approval_paths)
                or _approval_temp_paths(anchor)):
            raise RuntimeError("approval bootstrap is partial and unrecoverable; refusing repair")
        return _verify_stored_approval(charter, run_dir, digest, anchor,
                                       ledger_path, digest_path,
                                       (manifest_path, commit_path))

    measurement, problem = _measurement_spec(charter)
    if problem:
        raise RuntimeError(problem)
    measurement["_control_plane"] = run_dir
    receipt, receipt_problem = _read_containment_receipt(
        charter, canonical_probe_targets(charter["containment"])[1])
    if receipt_problem:
        raise RuntimeError(receipt_problem)
    claim, problem = _baseline_claim(charter)
    if problem:
        raise RuntimeError(problem)
    if charter.get("subject_frozen") is not True:
        raise RuntimeError("subject must be explicitly frozen")
    if not isinstance(charter.get("subject"), str) or not os.path.exists(charter["subject"]):
        raise RuntimeError("cannot bind missing unattended subject")

    before = _snapshot(charter, measurement)
    if not before["commit"]:
        raise RuntimeError("cannot bind source manifest without a git commit")
    baseline = _run_baseline(measurement, claim)
    components = [_run_component_exercise(name, charter["components"][name], charter)
                  for name in REQUIRED_COMPONENTS]
    after = _snapshot(charter, measurement)
    _verify_snapshot(before, after)
    execution = {"schema": "goal-loop-approval/v1", "before": before,
                 "after": after, "baseline": baseline, "components": components}
    manifest = source_manifest.build(charter, run_dir, manifest_path, commit_path)
    stable = _snapshot(charter, measurement)
    _verify_snapshot(after, stable)
    manifest_bytes = source_manifest.canonical_json(manifest)
    commit_bytes = (manifest["commit"] + "\n").encode()
    subject_binding = before["subject"]
    payload_value = {"run_id": charter["run_id"],
                     "ledger": None, "subject_binding": subject_binding,
                     "execution": execution,
                     "containment_receipt": receipt,
                     "source_binding": {
                         "manifest_path": manifest_path,
                         "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                         "commit_path": commit_path,
                         "commit_sha256": hashlib.sha256(commit_bytes).hexdigest()}}

    _write_new(digest_path, (digest + "\n").encode())
    _write_new(ledger_path, b"")
    stat = os.stat(ledger_path)
    payload_value["ledger"] = {"st_dev": stat.st_dev, "st_ino": stat.st_ino}
    temporary = f"{anchor}.tmp-{os.getpid()}"
    _write_new(temporary, json.dumps(payload_value, sort_keys=True).encode() + b"\n")
    os.replace(temporary, anchor)
    _sync_directory(anchor)
    _write_new(commit_path, commit_bytes)
    _write_new(manifest_path, manifest_bytes)
    return digest


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--exercise-component":
        _, _, name, path, claimed, exercise_id, profile_id = sys.argv
        registry = COMPONENT_EXERCISE_REGISTRY.get(name)
        if (not isinstance(registry, dict) or registry["exercise_id"] != exercise_id
                or registry["profile_id"] != profile_id
                or file_digest(path) != claimed):
            raise SystemExit(1)
        postcondition = _fixed_component_checks(name)
        if postcondition != registry["postcondition"]:
            raise SystemExit(1)
        print(json.dumps({"name": name, "exercise_id": exercise_id,
                          "profile_id": profile_id, "path": path,
                          "sha256": claimed, "result": "success",
                          "postcondition": postcondition},
                         sort_keys=True, separators=(",", ":")))
    else:
        print(approve(sys.argv[1]))
