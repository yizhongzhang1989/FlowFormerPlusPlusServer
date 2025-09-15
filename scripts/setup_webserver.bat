@echo off
setlocal enabledelayedexpansion

REM FlowFormer++ Web Server Setup Script
REM This script installs Flask dependencies and starts the web server

REM Source common functions
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%\common.bat" :init

call :print_info "=== Setting up Web Server ==="

REM Install Flask dependencies
call :install_flask_dependencies
if %errorlevel% neq 0 exit /b 1

REM Create necessary directories
call :create_directories

REM Create configuration file
call :create_config_file
if %errorlevel% neq 0 exit /b 1

REM Verify setup
call :verify_setup
if %errorlevel% neq 0 exit /b 1

call :print_success "Web server setup completed!"
call :print_info ""
call :print_info "Starting the web server now..."
call :print_info "After the server starts, you can:"
call :print_info "  1. Open your browser to the displayed URL"
call :print_info "  2. Upload two images to compute optical flow"
call :print_info "  3. View and download flow visualizations"
call :print_info "  4. Modify config.json to customize server settings"
call :print_info ""

REM Start the server
call :start_server
exit /b %errorlevel%

:install_flask_dependencies
call :print_info "Installing Flask dependencies..."

conda run -n flowformerpp pip install flask werkzeug
if %errorlevel% equ 0 (
    call :print_success "Flask dependencies installed successfully"
    exit /b 0
) else (
    call :print_error "Failed to install Flask dependencies"
    exit /b 1
)

:create_directories
call :print_info "Creating server directories..."

if not exist "tmp\uploads" mkdir "tmp\uploads"
if not exist "tmp\results" mkdir "tmp\results"
call :print_success "Server directories created"
exit /b 0

:create_config_file
call :print_info "Creating server configuration file..."

REM Check if config.json already exists
if exist "config.json" (
    call :print_info "Configuration file already exists, skipping creation"
    exit /b 0
)

REM Create config.json with default settings
(
echo {
echo   "server": {
echo     "host": "0.0.0.0",
echo     "port": 5000,
echo     "debug": false
echo   },
echo   "model": {
echo     "checkpoint_path": "checkpoints/sintel.pth",
echo     "device": "auto",
echo     "max_image_size": 1024
echo   },
echo   "upload": {
echo     "max_file_size_mb": 10,
echo     "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"],
echo     "upload_folder": "tmp/uploads",
echo     "results_folder": "tmp/results"
echo   },
echo   "processing": {
echo     "auto_cleanup": true,
echo     "cleanup_interval_hours": 24,
echo     "max_stored_results": 100
echo   }
echo }
) > config.json

if exist "config.json" (
    call :print_success "Configuration file created successfully"
    call :print_info "You can edit config.json to customize server settings"
    exit /b 0
) else (
    call :print_error "Failed to create configuration file"
    exit /b 1
)

:verify_setup
call :print_info "Verifying web server setup..."

REM Check if Flask is installed
conda run -n flowformerpp python -c "import flask; print(f'Flask version: {flask.__version__}')" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "✓ Flask installed and working"
) else (
    call :print_error "✗ Flask not properly installed"
    exit /b 1
)

REM Check if app.py exists
if exist "app.py" (
    call :print_success "✓ Web server app found"
) else (
    call :print_error "✗ app.py not found"
    exit /b 1
)

REM Check if config.json exists
if exist "config.json" (
    call :print_success "✓ Configuration file found"
) else (
    call :print_error "✗ config.json not found"
    exit /b 1
)

REM Check if checkpoints exist
if exist "checkpoints\sintel.pth" (
    call :print_success "✓ Model checkpoints available"
) else (
    call :print_error "✗ Model checkpoints not found"
    exit /b 1
)
exit /b 0

:start_server
call :print_info "Starting FlowFormer++ web server..."

REM Read port from config.json if it exists
set port=5000
if exist "config.json" (
    REM Simple extraction of port number - more robust than python parsing in batch
    for /f "tokens=2 delims=:" %%a in ('findstr "port" config.json') do (
        for /f "tokens=1 delims=, " %%b in ("%%a") do (
            set port=%%b
        )
    )
)

call :print_info "The server will be available at: http://localhost:!port!"
call :print_info "Press Ctrl+C to stop the server"
call :print_info ""

REM Start the Flask application
conda run -n flowformerpp python app.py
exit /b %errorlevel%

REM Include common functions inline
:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:print_step
echo === %~1 ===
goto :eof