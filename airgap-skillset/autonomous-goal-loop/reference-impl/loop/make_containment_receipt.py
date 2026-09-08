"""Emit one half of a containment receipt from probes actually run here, now.

Run twice: once uncontained for the control half, once inside the approved unit
for the contained half. The controller merges the two into a
`goal-loop-containment-receipt/v1` receipt.

Nothing in this file decides an outcome. It runs `negative_test.run_probes` in
whatever environment it was started in and records what happened, which is the
whole point: a receipt whose contained half is written by anything other than a
contained process is a receipt about nothing.
"""
import json
import os
import sys

import negative_test
import preflight


def main():
    mode, targets_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    targets = json.load(open(targets_path))
    probes = negative_test.run_probes(targets)
    record = {
        "mode": mode,
        "probes": probes,
        "targets": targets,
        "probe_source_sha256": preflight.file_digest(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "negative_test.py")),
        "cgroup": open("/proc/self/cgroup").read().strip(),
        "invocation_id": os.environ.get("INVOCATION_ID"),
    }
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
