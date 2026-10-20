@echo off
rem ============================================================================
rem  Internal Data Asset Platform - Vite dev server for the Vue frontend
rem  Default http://127.0.0.1:5173 ; /api is proxied to http://127.0.0.1:8000
rem  (start the backend first: scripts\start.bat)
rem  NOTE: keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%\frontend"
if not exist "node_modules" call npm install
call npm run dev
endlocal
