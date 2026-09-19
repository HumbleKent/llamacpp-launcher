@echo off
REM Convenience launcher for Windows: uses the project virtualenv if present.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m llamacpp_launcher
) else (
    echo No .venv found. Create one first:
    echo     py -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
)
