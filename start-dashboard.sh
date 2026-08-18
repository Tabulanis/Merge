#!/usr/bin/env bash
# Launch the Forge dashboard (Merge's web UI) under the project venv.
#
# WHY THIS SCRIPT EXISTS: her human voice (piper) is discovered next to the
# running python interpreter. piper lives in ./.venv/bin, so the server MUST
# run under .venv/bin/python — a bare `python3` (system/conda) silently loses
# her voice and falls back to the robot spd-say. This pins the right
# interpreter so that footgun can't fire. Args pass through (e.g. --network,
# --new-token).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$HERE/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY" >&2
  echo "  Create it first, e.g.:  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi
if [ ! -x "$HERE/.venv/bin/piper" ]; then
  echo "WARNING: piper missing from the venv — her human voice will fall back to the robot." >&2
fi

exec "$VENV_PY" -m forge.server "$@"
