"""Print the live runtime-binding observation and the verdict it produces.

Run under the approved unit only. Everything here is read from the kernel or
from systemd's own execution environment; nothing is taken from the charter
except the approved unit path being checked against.
"""
import json
import os
import sys
import time

import preflight


def main():
    unit_path = sys.argv[1]
    # Optional hold so the controller can read MainPID and ControlGroup from
    # systemd while this process is still the one being described. A oneshot
    # that has already exited reports MainPID=0 and an empty ControlGroup, which
    # would leave the receipt describing nothing.
    hold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    charter = {"execution_mode": "unattended", "containment": {"unit": unit_path}}
    observed = preflight.read_runtime_binding()
    refusal, binding = preflight.check_runtime_binding(charter)
    print(json.dumps({
        "approved_unit_path": unit_path,
        "pid": os.getpid(),
        "proc_self_cgroup": open("/proc/self/cgroup").read().strip(),
        "env": {key: os.environ.get(key) for key in
                ("SYSTEMD_UNIT", "INVOCATION_ID", "SYSTEMD_EXEC_PID")},
        "observed": observed,
        "refusal": refusal,
        "run_start_runtime_binding": binding,
    }, indent=2, sort_keys=True))
    sys.stdout.flush()
    if hold:
        time.sleep(hold)
    return 0 if refusal is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
