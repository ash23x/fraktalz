@echo off
REM ---------------------------------------------------------------
REM  mandelfly -- double-click to fly.
REM  First run installs dependencies into a local venv (~1 minute).
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv
    echo Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

".venv\Scripts\python.exe" Fraktalz.py --fullscreen --fps 144 %*
if errorlevel 1 pause

