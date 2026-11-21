@echo off
setlocal
cd /d "%~dp0"
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHAINLP_OFFLINE=1
if not exist ".venv\Scripts\python.exe" (
    echo [FAIL] .venv\Scripts\python.exe not found. Please run setup_windows.ps1 first.
    pause
    exit /b 1
)
echo Starting Thai Offline AI Workbench at http://127.0.0.1:7860 ...
".venv\Scripts\python.exe" run_app.py
