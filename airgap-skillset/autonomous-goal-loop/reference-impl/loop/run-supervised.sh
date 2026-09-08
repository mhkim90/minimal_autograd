#!/usr/bin/env bash
# Explicit human/test entry. The charter must independently approve this mode.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: run-supervised.sh <run-dir>" >&2
    exit 2
fi
RUN_DIR="$1"
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset GOAL_LOOP_SUPERVISED
exec python3 "$LOOP_DIR/iterate.py" --supervised "$RUN_DIR"
