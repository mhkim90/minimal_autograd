#!/usr/bin/env bash
# Serialize an owner stop with run.sh on the approved ledger inode.
set -euo pipefail

RUN_DIR="${1:?usage: stop.sh <run-dir>}"
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$LOOP_DIR/stop.py" "$RUN_DIR"
