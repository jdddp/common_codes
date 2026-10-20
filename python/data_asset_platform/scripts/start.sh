#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Internal Data Asset Platform - one-click start (Linux / macOS)
#
#  Usage :  scripts/start.sh [port] [--foreground]
#  Stop  :  scripts/stop.sh [port]
#  Settings: scripts/config.sh        (edit ../platform.env, then restart)
#
#  Every setting lives in ../platform.env: data directories, port, host,
#  python interpreter, cache sizes, ...  The shell only asks Python what is
#  in effect, so there is never a second set of values to keep in sync.
#
#  Outputs:
#    run/backend.log     backend stdout/stderr
#    run/worker.log      background worker stdout/stderr
#    run/backend.pid     pid files, read back by stop.sh
#    run/worker.pid
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
mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

PORT=""
FOREGROUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    -f|--foreground) FOREGROUND=1 ;;
    -h|--help) platform_usage; exit 0 ;;
    *)
      if printf '%s' "$1" | grep -Eq '^[0-9]+$'; then
        PORT="$1"
      else
        echo "[platform] unknown argument: $1" >&2
        platform_usage >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if ! PY="$(platform_python)"; then
  echo "[platform] python not found." >&2
  echo "           install deps first:  pip install -r backend/requirements.txt" >&2
  echo "           then set PYTHON_EXE in platform.env and run this again." >&2
  exit 1
fi
export PYTHON_EXE="$PY"

BIND_PORT="$(platform_config bind_port)"
BIND_HOST="$(platform_config bind_host)"
DATA_DIR="$(platform_config storage_root)"
if [ -z "$PORT" ]; then PORT="$BIND_PORT"; fi

echo "[platform] root      : $ROOT"
echo "[platform] python    : $PY"
echo "[platform] data dir  : $DATA_DIR"
echo "[platform] listen    : $BIND_HOST:$PORT"

if platform_health_ok "$PORT"; then
  echo
  echo "[platform] port $PORT already serves the platform:"
  echo "           http://127.0.0.1:$PORT/"
  echo "           to restart it, run: scripts/stop.sh $PORT"
  exit 1
fi

echo
echo "[platform] [1/4] initializing database ..."
"$PY" "$SCRIPT_DIR/init_db.py"

echo "[platform] [2/4] starting background worker (frame index + thumbnails) ..."
nohup "$PY" -m app.worker.main >"$RUN_DIR/worker.log" 2>&1 &
WORKER_PID=$!
printf '%s\n' "$WORKER_PID" >"$RUN_DIR/worker.pid"

start_backend() {
  echo "[platform] [4/4] waiting for backend to become ready ..."
  local i ready=0
  for i in $(seq 1 90); do
    if platform_health_ok "$PORT"; then ready=1; break; fi
    sleep 0.4
  done
  if [ "$ready" != "1" ]; then
    echo "[platform] backend did not become ready in time." >&2
    echo "           check the log: $RUN_DIR/backend.log" >&2
    exit 1
  fi
  echo
  echo "  ================================================================="
  echo "   Platform is up"
  echo "     browse / upload / import :  http://127.0.0.1:$PORT/"
  echo "     api docs                 :  http://127.0.0.1:$PORT/docs"
  echo "     viewer for one file      :  http://127.0.0.1:$PORT/viewer/{id}"
  echo "     logs                     :  $RUN_DIR/backend.log"
  echo "                                 $RUN_DIR/worker.log"
  echo "   To stop: scripts/stop.sh $PORT"
  echo "  ================================================================="
}

if [ "$FOREGROUND" = "1" ]; then
  echo "[platform] [3/4] backend runs in this terminal - Ctrl+C stops everything ..."
  echo "               log: $RUN_DIR/backend.log"
  # uvicorn is a tracked child (not a pipeline element), so the trap can stop
  # it and the worker no matter how the shell is interrupted. The log is
  # streamed with tail so the terminal still shows everything live.
  "$PY" -m uvicorn app.main:app --host "$BIND_HOST" --port "$PORT" \
    >"$RUN_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!
  printf '%s\n' "$BACKEND_PID" >"$RUN_DIR/backend.pid"
  cleanup_fg() {
    echo
    echo "[platform] stopping ..."
    stop_pid_file "$RUN_DIR/backend.pid"
    stop_pid_file "$RUN_DIR/worker.pid"
  }
  trap cleanup_fg EXIT INT TERM
  tail -n +1 -f "$RUN_DIR/backend.log" &
  TAIL_PID=$!
  wait "$BACKEND_PID"
  kill "$TAIL_PID" 2>/dev/null || true
  exit 0
fi

echo "[platform] [3/4] starting backend API ..."
nohup "$PY" -m uvicorn app.main:app --host "$BIND_HOST" --port "$PORT" \
  >"$RUN_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
printf '%s\n' "$BACKEND_PID" >"$RUN_DIR/backend.pid"

start_backend
