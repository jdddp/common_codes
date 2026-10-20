@echo off
rem ============================================================================
rem  Internal Data Asset Platform - start backend API only
rem  Prefer scripts\start.bat which also inits the db and starts the worker.
rem
rem  Env overrides: BIND_PORT / BIND_HOST / PYTHON_EXE
rem                 (defaults: 8000 / 0.0.0.0 / python on PATH)
rem  Config      : ..\platform.env supplies those same three keys, and Python
rem                 reads everything else (data directories, limits, ...) from
rem                 the same file. See scripts\show_config.py for effective values.
rem  NOTE: keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set ROOT=%~dp0..

rem platform.env -> PYTHON_EXE / BIND_HOST / BIND_PORT (env vars still win)
call "%ROOT%\scripts\_env.bat"

set PY=%PYTHON_EXE%
if "%PY%"=="" set PY=python
set PORT=%BIND_PORT%
if "%PORT%"=="" set PORT=8000
set HOST=%BIND_HOST%
if "%HOST%"=="" set HOST=0.0.0.0

cd /d "%ROOT%\backend"
echo [platform] python : %PY%
echo [platform] listen : %HOST%:%PORT%
echo [platform] page   : http://127.0.0.1:%PORT%/   (Ctrl+C to stop)
%PY% -m uvicorn app.main:app --host %HOST% --port %PORT%
endlocal
