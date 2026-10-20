@echo off
rem ============================================================================
rem  Internal Data Asset Platform - stop backend and worker
rem  Usage: scripts\stop.bat [port]      (default: platform.env BIND_PORT, else 8000)
rem  NOTE : keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal EnableExtensions

rem platform.env -> BIND_PORT (only used when no port argument is given)
call "%~dp0_env.bat"

set PORT=%~1
if "%PORT%"=="" set PORT=%BIND_PORT%
if "%PORT%"=="" set PORT=8000

echo [platform] stopping worker ...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match 'app\.worker\.main' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; '  stopped worker pid ' + $_.ProcessId } catch {} }"

echo [platform] stopping backend on port %PORT% ...
powershell -NoProfile -Command "$pids = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; if (-not $pids) { '  no listener on %PORT%' } else { $pids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; '  stopped backend pid ' + $_ } }"

echo [platform] done.
endlocal
