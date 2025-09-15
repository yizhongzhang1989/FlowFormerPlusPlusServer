@echo off
setlocal enabledelayedexpansion

REM FlowFormer++ Web Server Restart Script (Batch)
REM Use this script to restart the web server after initial setup

echo === Restarting FlowFormer++ Web Server ===

REM Check if conda environment exists
conda env list | findstr /C:"flowformerpp" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] flowformerpp conda environment not found
    echo Please run 'setup_server.bat' first to complete the initial setup
    exit /b 1
)
echo [SUCCESS] Conda environment found

REM Check if model checkpoints exist
if not exist "checkpoints" (
    echo [ERROR] Model checkpoints directory not found
    echo Please run 'setup_server.bat' first to download checkpoints
    exit /b 1
)

if not exist "checkpoints\sintel.pth" (
    echo [ERROR] Model checkpoints not found
    echo Please run 'setup_server.bat' first to download checkpoints
    exit /b 1
)
echo [SUCCESS] Model checkpoints found

REM Check if config file exists
if not exist "config.json" (
    echo [ERROR] Configuration file not found
    echo Please run 'setup_server.bat' first to create config.json
    exit /b 1
)
echo [SUCCESS] Configuration file found

REM Create necessary directories if they don't exist
echo [INFO] Creating server directories...
if not exist "tmp" mkdir tmp
if not exist "tmp\uploads" mkdir tmp\uploads
if not exist "tmp\results" mkdir tmp\results
echo [SUCCESS] Server directories ready

REM Read port from config.json (default to 5000 if parsing fails)
set port=5000
if exist "config.json" (
    for /f "tokens=*" %%i in ('conda run -n flowformerpp python -c "import json; print(json.load(open('config.json'))['server']['port'])" 2^>nul') do set port=%%i
)

echo.
echo [SUCCESS] Starting web server...
echo The server will be available at: http://localhost:!port!
echo [WARNING] Press Ctrl+C to stop the server
echo.

REM Start the Flask application
conda run -n flowformerpp python app.py