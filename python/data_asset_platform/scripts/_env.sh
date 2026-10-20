#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Sourced by start.sh / stop.sh / config.sh / serve.sh - do not execute it.
#
#  It only answers one question for the shell: which python?
#  Every other key in platform.env (data dirs, limits, tuning) is read by
#  Python itself in backend/app/config.py, so the shell never has to parse
#  the whole file. One file, one source of truth.
#
#  The caller must set ROOT before sourcing this file.
# ---------------------------------------------------------------------------
# shellcheck shell=bash

_platform_env_file="${PLATFORM_ENV_FILE:-${ROOT:?ROOT must be set before sourcing _env.sh}/platform.env}"

if [ -f "$_platform_env_file" ] && [ -z "${PYTHON_EXE:-}" ]; then
  # Take the last PYTHON_EXE= line. Commented lines never match the grep,
  # because the key has to be the first thing on the line.
  _line="$(grep -E '^[[:space:]]*PYTHON_EXE[[:space:]]*=' "$_platform_env_file" | tail -n 1 || true)"
  if [ -n "$_line" ]; then
    _v="$(printf '%s\n' "$_line" \
      | sed -e 's/^[^=]*=//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
            -e 's/^"//' -e 's/"$//')"
    if [ -n "$_v" ]; then
      PYTHON_EXE="$_v"
      export PYTHON_EXE
    fi
    unset _v
  fi
  unset _line
fi
unset _platform_env_file

# --- interpreter ------------------------------------------------------------
# Priority: PYTHON_EXE (env or platform.env) > active conda env > python3 > python
platform_python() {
  if [ -n "${PYTHON_EXE:-}" ] && [ -x "$PYTHON_EXE" ]; then
    printf '%s' "$PYTHON_EXE"
    return 0
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    printf '%s' "$CONDA_PREFIX/bin/python"
    return 0
  fi
  local candidate
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# --- ask Python for one resolved setting ------------------------------------
# $1 = Settings field name (storage_root / db_path / bind_host / bind_port / ...)
# Going through Python keeps the shell and the running app in sync, and picks
# up the priority order (env var > platform.env > default) for free.
platform_config() {
  "$PY" "$ROOT/scripts/show_config.py" "$1"
}

# --- health probe -----------------------------------------------------------
# $1 = port
platform_health_ok() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 2 "http://127.0.0.1:$1/api/health" >/dev/null 2>&1
  else
    "$PY" -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:%s/api/health'%sys.argv[1],timeout=2)" \
      "$1" >/dev/null 2>&1
  fi
}

# --- stop the process recorded in a pid file --------------------------------
# $1 = pid file. Politely first, then by force. Always removes the file.
stop_pid_file() {
  local pidfile="$1" pid i
  [ -f "$pidfile" ] || return 0
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  rm -f "$pidfile"
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for i in $(seq 1 25); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 0.2
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "[platform] had to force-kill pid $pid"
  fi
  return 0
}

# --- free a listening port --------------------------------------------------
# $1 = port. Used as a safety net when a pid file was lost.
free_port() {
  local port="$1" pids="" pid
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -lptn "sport = :$port" 2>/dev/null \
      | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
  fi
  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    kill -9 "$pid" 2>/dev/null || true
    echo "[platform] stopped pid $pid listening on port $port"
  done
  return 0
}

platform_usage() {
  cat <<'USAGE'
  usage: scripts/start.sh [port] [--foreground]

    (no argument)        use the port from platform.env (default 8000)
    8080                 use this port for this run only
    -f, --foreground     run the backend in this terminal; Ctrl+C stops it
                         (the worker still runs in the background)

  stop:      scripts/stop.sh [port]
  settings:  scripts/config.sh
USAGE
}
