#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Internal Data Asset Platform - run only the backend API, in the foreground
#
#  This is what the systemd units in ../deploy/ run. Handy on its own too,
#  but scripts/start.sh is the one that also brings up the background worker
#  and waits for the health check.
#
#  Host and port are resolved by Python (platform.env + env vars + defaults),
#  so this script never has to parse the config file itself.
#
#  NOTE: keep LF line endings for every .sh file.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT
# shellcheck source=scripts/_env.sh
. "$SCRIPT_DIR/_env.sh"

if ! PY="$(platform_python)"; then
  echo "[platform] python not found." >&2
  echo "           install deps first:  pip install -r backend/requirements.txt" >&2
  echo "           then set PYTHON_EXE in platform.env and run this again." >&2
  exit 1
fi
export PYTHON_EXE="$PY"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

BIND_HOST="$(platform_config bind_host)"
BIND_PORT="$(platform_config bind_port)"
DATA_DIR="$(platform_config storage_root)"

echo "[platform] python    : $PY"
echo "[platform] data dir  : $DATA_DIR"
echo "[platform] listen    : $BIND_HOST:$BIND_PORT"
echo "[platform] page      : http://127.0.0.1:$BIND_PORT/   (Ctrl+C to stop)"

exec "$PY" -m uvicorn app.main:app --host "$BIND_HOST" --port "$BIND_PORT"
