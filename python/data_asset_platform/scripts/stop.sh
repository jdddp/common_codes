#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Internal Data Asset Platform - stop backend and worker
#
#  Usage :  scripts/stop.sh [port] [--all]
#
#    (no argument)   stop the instance whose port is in platform.env
#    8080            stop whatever is listening on 8080
#    --all           also stop stray "app.worker.main" processes that lost
#                    their pid file (e.g. after a crash or a reboot)
#
#  NOTE: keep LF line endings for every .sh file.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT
# shellcheck source=scripts/_env.sh
. "$SCRIPT_DIR/_env.sh"

RUN_DIR="$ROOT/run"

PORT=""
KILL_ALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) KILL_ALL=1 ;;
    -h|--help)
      echo "  usage: scripts/stop.sh [port] [--all]"
      exit 0
      ;;
    *)
      if printf '%s' "$1" | grep -Eq '^[0-9]+$'; then
        PORT="$1"
      else
        echo "[platform] unknown argument: $1" >&2
        echo "           usage: scripts/stop.sh [port] [--all]" >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if [ -z "$PORT" ]; then
  # The port itself comes from Python, so a comment or a default in
  # platform.env is handled exactly the same way as when starting.
  if PY="$(platform_python 2>/dev/null)" && [ -n "$PY" ]; then
    PORT="$(platform_config bind_port 2>/dev/null || true)"
  fi
  PORT="${PORT:-8000}"
fi

echo "[platform] stopping worker ..."
stop_pid_file "$RUN_DIR/worker.pid"

echo "[platform] stopping backend (pid file) ..."
stop_pid_file "$RUN_DIR/backend.pid"

echo "[platform] freeing port $PORT ..."
free_port "$PORT"

if [ "$KILL_ALL" = "1" ]; then
  echo "[platform] --all: stopping stray workers ..."
  pkill -f 'app\.worker\.main' 2>/dev/null || true
fi

echo "[platform] done."
