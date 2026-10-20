@echo off
rem ============================================================================
rem  Internal Data Asset Platform - background worker (frame index + thumbnails)
rem  Prefer scripts\start.bat which also inits the db and starts the backend.
rem
rem  Env overrides: PYTHON_EXE (default: python on PATH)
rem  Config      : ..\platform.env supplies PYTHON_EXE; Python reads the rest
rem                 (data directories, worker tuning) from the same file.
rem  NOTE: keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set ROOT=%~dp0..

rem platform.env -> PYTHON_EXE (env var still wins)
call "%ROOT%\scripts\_env.bat"

set PY=%PYTHON_EXE%
if "%PY%"=="" set PY=python

cd /d "%ROOT%\backend"
echo [platform] worker starting, python: %PY%
%PY% -m app.worker.main
endlocal
