@echo off
rem ============================================================================
rem  Internal Data Asset Platform - build the Vue frontend (needs Node.js)
rem  Output frontend\dist is served automatically by the backend; restart the
rem  backend afterwards and http://127.0.0.1:8000/ becomes the Vue app.
rem  Without a build you still get the dependency-free page in backend/app/static.
rem  NOTE: keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%\frontend"
call npm install
if errorlevel 1 (
  echo [platform] npm install failed. Is Node.js installed and on PATH?
  exit /b 1
)
call npm run build
if errorlevel 1 (
  echo [platform] npm run build failed.
  exit /b 1
)
echo.
echo [platform] build finished: %ROOT%\frontend\dist
echo [platform] restart the backend, then open http://127.0.0.1:8000/
endlocal
