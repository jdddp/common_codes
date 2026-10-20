#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Internal Data Asset Platform - show the effective configuration
#
#  Usage :  scripts/config.sh                 every option, with its source
#           scripts/config.sh storage_root    just one value
#           scripts/config.sh bind_port       (what start.sh would use)
#
#  Sources, highest priority first:
#      1) environment variables set in this shell
#      2) ../platform.env        <- edit this file, then restart the platform
#      3) built-in defaults in backend/app/config.py
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

exec "$PY" "$ROOT/scripts/show_config.py" "$@"
