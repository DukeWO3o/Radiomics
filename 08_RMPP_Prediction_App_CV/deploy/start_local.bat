@echo off
rem ===========================================================================
rem RMPP classification support system - local launcher (Windows)
rem Requires conda env rmpp-pack (fallbacks: rmpp-app, Pyradiomics)
rem Open http://localhost:8501 after startup.
rem ===========================================================================
chcp 65001 >nul
title RMPP Classification Support System
cd /d "%~dp0\.."

call conda activate rmpp-pack 2>nul
if errorlevel 1 (
    echo [Info] rmpp-pack not found, trying rmpp-app ...
    call conda activate rmpp-app 2>nul
)
if errorlevel 1 (
    echo [Info] rmpp-app not found, trying Pyradiomics ...
    call conda activate Pyradiomics 2>nul
)
if errorlevel 1 (
    echo [Error] No conda environment found. Run:
    echo         conda env create -f environment.yml
    pause
    exit /b 1
)

echo Starting RMPP classification support system ... open http://localhost:8501
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats=false

echo.
echo Service stopped.
pause
