@echo off
rem ============================================================================
rem  Internal Data Asset Platform - generate a synthetic HDY sample file
rem  Handy for demos and acceptance checks when no real sonar file is at hand.
rem  NOTE: keep this file ASCII-only (cmd mis-parses non-ASCII batch files).
rem ============================================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set ROOT=%~dp0..
set PY=%PYTHON_EXE%
if "%PY%"=="" set PY=python

cd /d "%ROOT%\backend"
echo [platform] python: %PY%
%PY% -m tests.synth_hdy "%ROOT%\samples\synthetic_sonar.HDY" --frames 120 --beams 256 --ranges 512
if errorlevel 1 (
  echo [platform] failed to generate the sample.
  exit /b 1
)
echo [platform] sample written to %ROOT%\samples\synthetic_sonar.HDY
endlocal
