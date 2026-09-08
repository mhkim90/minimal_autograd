#!/usr/bin/env bash
# One iteration under the exclusive lock on the approved ledger FD.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: run.sh <run-dir>" >&2
    exit 2
fi
RUN_DIR="$1"
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The sentinel is not checked here. Let the component that can read the ledger
# decide what a stopped run still owes.
unset GOAL_LOOP_SUPERVISED
exec python3 "$LOOP_DIR/iterate.py" "$RUN_DIR"
