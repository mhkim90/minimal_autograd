"""Approval-bound declared command-source set for unattended runs.

There is deliberately no import or shell inspection here.  An approval binds
the fixed loop sources plus the paths explicitly declared for every command;
anything that cannot be described that way is refused at entry.  This is a
declared command-source contract, not a claim that Python's complete runtime
closure has been discovered.
"""
import hashlib
import importlib.machinery
import json
import os
import re
import shutil
import subprocess


HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = "goal-loop-source-manifest/v1"
FIXED_NAMES = (
    "run.sh", "iterate.py", "collector.py", "evaluator.py", "notifier.py",
    "ledger.py", "preflight.py", "clock.py", "negative_test.py",
    "source_manifest.py", "run-supervised.sh", "stop.sh", "stop.py",
)
SHELLS = {"sh", "bash", "dash", "zsh", "ksh"}
PYTHON_RE = re.compile(r"(?:python(?:3(?:\.\d+)?)?|pypy(?:3(?:\.\d+)?)?)$")


def _resolve_git():
    for candidate in ("/usr/bin/git", "/bin/git"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.realpath(candidate)
    return None


def _early_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


GIT_PATH = _resolve_git()
GIT_SHA256 = _early_digest(GIT_PATH) if GIT_PATH else None


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(path, label):
    if not isinstance(path, str) or not os.path.isabs(path):
        return None, f"{label} must contain absolute paths"
    resolved = os.path.realpath(path)
    if resolved != path:
        return None, f"{label} path must be canonical: {path!r}"
    return resolved, None


def _fixed_paths(charter, run_dir):
    unit = (charter.get("containment") or {}).get("unit")
    paths = [os.path.join(HERE, name) for name in FIXED_NAMES]
    paths.append(os.path.join(run_dir, "charter.json"))
    if isinstance(unit, str):
        paths.append(unit)
    receipt = ((charter.get("containment") or {}).get("negative_test") or {}).get(
        "receipt_path")
    if isinstance(receipt, str):
        paths.append(receipt)
    return paths


def _source_file(path, label):
    canonical, problem = _canonical(path, label)
    if problem:
        return None, problem
    if not os.path.isfile(canonical):
        return None, f"{label} target is missing: {canonical}"
    return canonical, None


def _resolve_module(module, cwd, label):
    """Resolve `python -m` without importing package code.

    Namespace, zip and other opaque loaders have no stable source file to bind,
    so unattended approval refuses them rather than guessing what Python will
    execute.
    """
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        return None, f"{label} module resolution requires an absolute cwd"
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module or ""):
        return None, f"{label} python module target is not a static module name"
    search = [cwd]
    spec = None
    for part in module.split("."):
        spec = importlib.machinery.PathFinder.find_spec(part, search)
        if spec is None or spec.origin is None:
            return None, f"{label} python module target is namespace or unresolved: {module!r}"
        if spec.submodule_search_locations is not None:
            search = list(spec.submodule_search_locations)
    if spec.submodule_search_locations is not None:
        candidates = [os.path.join(path, "__main__.py") for path in search]
        target = next((path for path in candidates if os.path.isfile(path)), None)
        if target is None:
            return None, f"{label} python module target has no canonical __main__.py"
    else:
        target = spec.origin
    canonical, problem = _source_file(target, f"{label} python module target")
    if problem:
        return None, problem
    if not canonical.endswith(".py"):
        return None, f"{label} python module target uses an opaque loader: {canonical}"
    return canonical, None


def _command_target(label, command, cwd):
    executable = os.path.basename(command[0])
    args = command[1:]
    if "-c" in args or executable in {"eval", "exec"}:
        return None, f"opaque command is not allowed for unattended {label}"
    if PYTHON_RE.fullmatch(executable):
        if not args:
            return None, f"{label} python command has no script or module target"
        if args[0] == "-m":
            if len(args) < 2 or args[1].startswith("-"):
                return None, f"{label} python module target is missing"
            return _resolve_module(args[1], cwd, label)
        if args[0].startswith("-"):
            return None, f"{label} python command form is not bindable"
        target, problem = _source_file(args[0], f"{label} python script target")
        if problem:
            return None, problem
        return target, None
    if executable in SHELLS:
        if not args or args[0].startswith("-"):
            return None, f"{label} shell script target is missing or opaque"
        return _source_file(args[0], f"{label} shell script target")

    executable_path = command[0]
    if not os.path.isabs(executable_path):
        executable_path = shutil.which(executable_path)
        if executable_path is None:
            return None, f"{label} executable target cannot be resolved"
    return _source_file(executable_path, f"{label} executable target")


def _command_sources(label, command, declaration):
    if not isinstance(command, list) or not command \
            or not all(isinstance(part, str) and part for part in command):
        return set(), f"{label} command must be a non-empty structured argv"
    raw_sources = declaration.get("source_paths") if isinstance(declaration, dict) else None
    if raw_sources is None:
        raw_sources = []
    if not isinstance(raw_sources, list):
        return set(), f"{label}.source_paths must be a list"
    sources = set()
    for source in raw_sources:
        canonical, problem = _canonical(source, f"{label}.source_paths")
        if problem:
            return set(), problem
        sources.add(canonical)

    target, problem = _command_target(label, command, declaration.get("cwd"))
    if problem:
        return set(), problem
    if target not in sources:
        return set(), f"{label} command target is not declared: {target}"
    return sources, None


def _command_declarations(charter):
    sources = set()
    notify = charter.get("notify") or {}
    found, problem = _command_sources("notify", notify.get("command"), notify)
    if problem:
        return set(), problem
    sources.update(found)
    if "measurement" in charter:
        measurement = charter.get("measurement") or {}
        found, problem = _command_sources("baseline measurement",
                                          measurement.get("argv"), measurement)
        if problem:
            return set(), problem
        sources.update(found)
    arms = charter.get("arms")
    if not isinstance(arms, dict):
        return set(), "arms must be a mapping for source closure"
    for name, spec in arms.items():
        if not isinstance(spec, dict):
            return set(), f"arm {name!r} is not a mapping"
        found, problem = _command_sources(f"arm {name!r}", spec.get("argv"), spec)
        if problem:
            return set(), problem
        sources.update(found)
    return sources, None


def expected_paths(charter, run_dir):
    """Return the mechanically expected set, independent of manifest contents."""
    paths = set(_fixed_paths(charter, run_dir))
    measurement = charter.get("measurement")
    if isinstance(measurement, dict):
        found, problem = _command_sources("baseline measurement",
                                          measurement.get("argv"), measurement)
        if problem:
            return paths, problem
        paths.update(found)
    command_sources, problem = _command_declarations(charter)
    if problem:
        return paths, problem
    paths.update(command_sources)
    for name, spec in (charter.get("components") or {}).items():
        value = spec.get("path", spec.get("source")) if isinstance(spec, dict) else spec
        if isinstance(value, str) and os.path.isabs(value):
            # Component validation reports non-canonical paths specifically;
            # source closure must not mask that durable component refusal.
            paths.add(os.path.realpath(value))
        else:
            return paths, f"component {name!r} path must be an absolute path"
    return paths, None


def _declared_paths(charter, run_dir):
    spec = charter.get("source_manifest")
    declared = spec.get("files") if isinstance(spec, dict) else None
    if not isinstance(declared, list):
        expected, _ = expected_paths(charter, run_dir)
        return expected
    paths = set()
    for path in declared:
        canonical, problem = _canonical(path, "source_manifest.files")
        if problem:
            return set()
        paths.add(canonical)
    return paths


def validate_declarations(charter, run_dir):
    """Validate the charter's exact declaration and return its expected paths."""
    expected, problem = expected_paths(charter, run_dir)
    if problem:
        return expected, problem
    spec = charter.get("source_manifest")
    if not isinstance(spec, dict):
        return expected, "unattended source_manifest declaration is missing"
    declared = spec.get("files")
    if not isinstance(declared, list) or not declared:
        return expected, "source declaration must contain a non-empty files list"
    canonical = []
    for path in declared:
        value, problem = _canonical(path, "source_manifest.files")
        if problem:
            return expected, problem
        canonical.append(value)
    if len(canonical) != len(set(canonical)):
        return expected, "source declaration contains duplicate paths"
    if set(canonical) != expected:
        missing, extra = sorted(expected - set(canonical)), sorted(set(canonical) - expected)
        return expected, f"source declaration set mismatch: missing={missing}, extra={extra}"
    for path in expected:
        if not os.path.isfile(path):
            return expected, f"declared source is missing: {path}"
    return expected, None


def paths_for(charter, run_dir):
    """Return only the explicit, canonical declared command-source set."""
    expected, problem = validate_declarations(charter, run_dir)
    if problem:
        raise ValueError(problem)
    return tuple(sorted(expected))


def default_paths(run_dir, run_id):
    import preflight
    anchor = preflight.anchor_path(run_dir, run_id)
    parent = os.path.dirname(anchor)
    stem = os.path.basename(anchor)[:-len(".anchor")]
    return (os.path.join(parent, stem + ".source-manifest.json"),
            os.path.join(parent, stem + ".commit.txt"))


def configured_paths(charter, run_dir):
    spec = charter.get("source_manifest")
    manifest, commit = default_paths(run_dir, charter["run_id"])
    if spec is None:
        # Runtime entry still rejects this missing declaration.  Approval keeps
        # its historical bootstrap behavior so the refusal is durable and
        # cannot be confused with an uninitialized ledger.
        return manifest, commit, None
    if not isinstance(spec, dict):
        return manifest, commit, "source_manifest must be a mapping"
    supplied_manifest = spec.get("manifest_path", manifest)
    supplied_commit = spec.get("commit_path", commit)
    if not isinstance(supplied_manifest, str) or not isinstance(supplied_commit, str):
        return None, None, "source_manifest paths must be strings"
    manifest, commit = os.path.realpath(supplied_manifest), os.path.realpath(supplied_commit)
    if manifest != supplied_manifest or commit != supplied_commit:
        return None, None, "source_manifest paths must be canonical"
    anchor_parent = os.path.dirname(default_paths(run_dir, charter["run_id"])[0])
    if (os.path.dirname(manifest), os.path.dirname(commit)) != (anchor_parent, anchor_parent):
        return None, None, "source_manifest files must be beside the external anchor"
    return manifest, commit, None


def current_commit(path):
    directory = os.path.dirname(path)
    if not GIT_PATH or not GIT_SHA256 or digest(GIT_PATH) != GIT_SHA256:
        return None
    try:
        env = {"PATH": os.path.dirname(GIT_PATH), "LC_ALL": "C", "LANG": "C"}
        return subprocess.run([GIT_PATH, "-C", directory, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True,
                              timeout=5, env=env).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build(charter, run_dir, manifest_path, commit_path):
    commit = current_commit(HERE)
    if not commit:
        raise RuntimeError("cannot bind source manifest without a git commit")
    files = [{"path": path, "sha256": digest(path)}
             for path in sorted(_declared_paths(charter, run_dir))]
    return {"format": FORMAT, "run_id": charter["run_id"],
            "manifest_path": manifest_path, "commit_path": commit_path,
            "git_path": GIT_PATH, "git_sha256": GIT_SHA256,
            "commit": commit, "files": files}


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
