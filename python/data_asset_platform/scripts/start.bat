@echo off
rem ============================================================================
rem  Internal Data Asset Platform - one-click start
rem    1) init database   2) background worker   3) backend API   4) open browser
rem
rem  Usage :  scripts\start.bat [port] [/nobrowser]
rem           scripts\start.bat             -> port 8000, open browser
rem           scripts\start.bat 8080        -> port 8080
rem           scripts\start.bat 8080 /nobrowser
rem  Stop  :  scripts\stop.bat [port]
rem
rem  Interpreter : %PYTHON_EXE% first, then this machine's conda work env, then PATH.
rem  Config      : ..\platform.env holds PYTHON_EXE / BIND_HOST / BIND_PORT.
rem                Data directories (STORAGE_ROOT / DB_PATH / CACHE_DIR) and every
rem                other option live in the same file and are read by Python.
rem                See scripts\show_config.py for the effective values.
rem  NOTE  : keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set ROOT=%~dp0..

rem platform.env -> PYTHON_EXE / BIND_HOST / BIND_PORT (env vars still win)
call "%ROOT%\scripts\_env.bat"

rem PORT is empty until a command-line argument or platform.env fills it in
set PORT=
set OPEN_BROWSER=1

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="/nobrowser" (
  set OPEN_BROWSER=0
  shift
  goto parse_args
)
echo %~1|findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 (
  set PORT=%~1
  shift
  goto parse_args
)
echo [platform] unknown argument: %~1
echo            usage: scripts\start.bat [port] [/nobrowser]
exit /b 2
:args_done

rem ------------------------------------ port: argument > platform.env > 8000
if not defined PORT set PORT=%BIND_PORT%
if not defined PORT set PORT=8000

rem ---------------------------------------------------------------- interpreter
set PY=%PYTHON_EXE%
if defined PY if exist "%PY%" goto python_ok
if exist "C:\Users\82188\miniconda3\envs\work\python.exe" (
  set PY=C:\Users\82188\miniconda3\envs\work\python.exe
  goto python_ok
)
if exist "%USERPROFILE%\miniconda3\envs\work\python.exe" (
  set PY=%USERPROFILE%\miniconda3\envs\work\python.exe
  goto python_ok
)
set PY_FOUND=
for %%P in (python.exe) do set PY_FOUND=%%~$PATH:P
if defined PY_FOUND (
  set PY=%PY_FOUND%
  goto python_ok
)
echo [platform] Python interpreter not found.
echo            Install deps first: pip install -r backend\requirements.txt
echo            Then set the interpreter and run this script again:
echo              set PYTHON_EXE=C:\path\to\python.exe
exit /b 1

:python_ok
set BIND_PORT=%PORT%
rem export so the child windows (run_backend.bat / run_worker.bat) reuse it
set PYTHON_EXE=%PY%

rem ask Python where the data actually goes (platform.env + env vars + defaults).
rem cmd's "for /f" cannot run a command that itself needs quotes, so the value
rem goes through a temp file instead.
set DATA_DIR=
"%PY%" "%ROOT%\scripts\show_config.py" storage_root > "%TEMP%\_platform_datadir.txt" 2>nul
set /p DATA_DIR=<"%TEMP%\_platform_datadir.txt"
del "%TEMP%\_platform_datadir.txt" >nul 2>&1

echo [platform] root      : %ROOT%
echo [platform] python    : %PY%
echo [platform] data dir  : %DATA_DIR%
echo [platform] port      : %PORT%

rem ------------------------------------------------------------- port in use?
set PORT_STATUS=unknown
for /f %%C in ('powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { 'busy' } else { 'free' }"') do set PORT_STATUS=%%C
if /i "%PORT_STATUS%"=="busy" (
  echo [platform] port %PORT% is already in use - the platform may already be running:
  echo            http://127.0.0.1:%PORT%/
  echo            to restart it, run: scripts\stop.bat %PORT%
  exit /b 1
)

rem ------------------------------------------------------------------ 1) init
echo.
echo [platform] [1/4] initializing database ...
pushd "%ROOT%"
"%PY%" scripts\init_db.py
if errorlevel 1 (
  echo [platform] init failed, aborting.
  popd
  exit /b 1
)
popd

rem ---------------------------------------------------------------- 2) worker
echo [platform] [2/4] starting background worker (frame index + thumbnails) ...
start "Platform Worker" /min cmd /c call "%ROOT%\scripts\run_worker.bat"

rem --------------------------------------------------------------- 3) backend
echo [platform] [3/4] starting backend API ...
start "Platform Backend %PORT%" cmd /c call "%ROOT%\scripts\run_backend.bat"

rem ----------------------------------------------------------- 4) health wait
echo [platform] [4/4] waiting for backend to become ready ...
powershell -NoProfile -Command "for ($i = 0; $i -lt 90; $i++) { try { if ((Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2).StatusCode -eq 200) { exit 0 } } catch {} ; Start-Sleep -Milliseconds 400 } ; exit 1"
if errorlevel 1 (
  echo [platform] backend did not become ready in time.
  echo            check the log window titled "Platform Backend %PORT%".
  exit /b 1
)

if "%OPEN_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%/"

echo.
echo   ================================================================
echo    Platform is up
echo      browse / upload / import :  http://127.0.0.1:%PORT%/
echo      api docs                 :  http://127.0.0.1:%PORT%/docs
echo      viewer for one file      :  http://127.0.0.1:%PORT%/viewer/{id}
echo    Windows opened: "Platform Backend %PORT%" (log) and "Platform Worker"
echo    To stop: scripts\stop.bat %PORT%
echo   ================================================================
echo.
rem when launched by double-click, keep the window so the summary stays readable
echo %cmdcmdline% | find /i "start.bat" >nul
if not errorlevel 1 pause
exit /b 0
