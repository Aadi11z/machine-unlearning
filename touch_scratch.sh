#!/usr/bin/env bash
# Refresh timestamps in a scratch project directory.
#
# Usage:
#   ./touch_scratch.sh /scratch/<username>/<project-name>
#
# For cron, always pass the absolute scratch project path because cron may not
# load shell startup files or define $SCRATCH.

set -euo pipefail

SCRATCH_PROJECT="${1:-/<dummy>/<path>}"

if [[ ! -d "$SCRATCH_PROJECT" ]]; then
    echo "[ERROR] Scratch project directory not found: $SCRATCH_PROJECT" >&2
    exit 1
fi

echo "[INFO] Refreshing scratch project: $SCRATCH_PROJECT"
echo "[INFO] Started: $(date)"

echo "[INFO] Refreshing regular files"
find "$SCRATCH_PROJECT" -type f -exec touch {} +

echo "[INFO] Refreshing directories"
find "$SCRATCH_PROJECT" -type d -exec touch {} +

echo "[INFO] Refreshing symbolic links"
find "$SCRATCH_PROJECT" -type l -exec touch -h {} +

echo "[INFO] Oldest Access/Modify timestamps after refresh:"
find "$SCRATCH_PROJECT" -printf '%A+ %T+ %p\n' | sort | head

echo "[INFO] Finished: $(date)"
