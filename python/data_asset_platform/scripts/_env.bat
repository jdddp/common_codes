@echo off
rem ---------------------------------------------------------------------------
rem  Load a few ASCII-safe keys from platform.env into the caller's environment.
rem
rem  Called by start.bat / stop.bat / run_backend.bat / run_worker.bat.
rem
rem  Rules:
rem    - an environment variable that is already set always wins, so
rem        set BIND_PORT=9000
rem        scripts\start.bat
rem      still starts on 9000.
rem    - only ASCII keys are read here: PYTHON_EXE, BIND_HOST, BIND_PORT.
rem      Every other key (STORAGE_ROOT / DB_PATH / CACHE_DIR / ...) is read
rem      by Python from the very same file - see backend\app\config.py.
rem      Keeping the split this way means one file, one source of truth.
rem    - "delims== " (equals + space) means "KEY = VALUE" still works and
rem      values may contain spaces (e.g. C:\Program Files\...). Do not put
rem      spaces at the *end* of a value - cmd cannot trim those.
rem    - keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ---------------------------------------------------------------------------
set "_PLATFORM_ENV=%PLATFORM_ENV_FILE%"
if not defined _PLATFORM_ENV set "_PLATFORM_ENV=%~dp0..\platform.env"
if not exist "%_PLATFORM_ENV%" (
  set "_PLATFORM_ENV="
  goto :env_done
)
for /f "usebackq eol=# tokens=1,* delims== " %%A in ("%_PLATFORM_ENV%") do (
  if /i "%%A"=="PYTHON_EXE" if not "%%~B"=="" if not defined PYTHON_EXE set "PYTHON_EXE=%%~B"
  if /i "%%A"=="BIND_HOST" if not "%%~B"=="" if not defined BIND_HOST set "BIND_HOST=%%~B"
  if /i "%%A"=="BIND_PORT" if not "%%~B"=="" if not defined BIND_PORT set "BIND_PORT=%%~B"
)
set "_PLATFORM_ENV="
:env_done
