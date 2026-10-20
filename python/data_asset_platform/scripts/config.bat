@echo off
rem ============================================================================
rem  Internal Data Asset Platform - show the effective configuration
rem
rem  Usage :  scripts\config.bat                  show every option
rem           scripts\config.bat storage_root     show just one value
rem           scripts\config.bat bind_port        (also used by start.bat)
rem
rem  Sources, highest priority first:
rem      1) environment variables set in this terminal
rem      2) ..\platform.env      <- edit this file, then restart the platform
rem      3) built-in defaults in backend\app\config.py
rem
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
if not exist "%PY%" set PY=python

"%PY%" "%ROOT%\scripts\show_config.py" %*
endlocal
