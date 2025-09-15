@echo off
setlocal enabledelayedexpansion

REM FlowFormer++ Server Setup Script
REM This script automatically sets up the FlowFormer++ project after cloning

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPTS_DIR=%SCRIPT_DIR%scripts"

REM Call common functions
call "%SCRIPTS_DIR%\common.bat" :init

call :print_step "FlowFormer++ Server Setup"
call :print_info "Starting automatic setup process..."
call :print_info ""
call :print_info "This setup will:"
call :print_info "  1. Download model checkpoints (~325MB)"
call :print_info "  2. Create conda environment with dependencies"
call :print_info "  3. Create configuration file (config.json)"
call :print_info "  4. Install web server dependencies"
call :print_info "  5. Start the web server (default: http://localhost:5000)"
call :print_info ""
call :print_warning "Note: The web server will start automatically after setup!"
call :print_info "You can customize server settings by editing config.json"
call :print_info ""

REM Define setup steps
set step1_name=Step 1: Download Model Checkpoints
set step1_script=%SCRIPTS_DIR%\download_ckpts.bat

set step2_name=Step 2: Setup Conda Environment
set step2_script=%SCRIPTS_DIR%\setup_conda_env.bat

set step3_name=Step 3: Setup and Start Web Server
set step3_script=%SCRIPTS_DIR%\setup_webserver.bat

set total_steps=3

REM Main execution
call :print_info "Found %SCRIPTS_DIR% for step scripts"

REM Check if scripts directory exists
if not exist "%SCRIPTS_DIR%" (
    call :print_error "Scripts directory not found: %SCRIPTS_DIR%"
    exit /b 1
)

REM Run all setup steps
call :run_setup

REM Note: The script will end here because the web server keeps running
call :print_step "Setup Complete"
call :print_success "FlowFormer++ server setup completed successfully!"
call :print_info "The web server should now be running"
call :print_info "Check the output above for the server URL"
call :print_info "You can customize settings by editing config.json"

goto :eof

:run_setup
for /l %%i in (1,1,%total_steps%) do (
    set current_step=%%i
    
    call :print_info "Running !step%%i_name! (!current_step!/%total_steps%)..."
    
    call :run_step "!step%%i_name!" "!step%%i_script!"
    if !errorlevel! neq 0 (
        call :print_error "!step%%i_name! failed! Setup aborted."
        exit /b 1
    )
    
    call :print_success "!step%%i_name! completed!"
    echo.
)
goto :eof

:run_step
set step_name=%~1
set script_path=%~2

call :print_step "%step_name%"

if not exist "%script_path%" (
    call :print_error "Step script not found: %script_path%"
    exit /b 1
)

call "%script_path%"
if !errorlevel! neq 0 (
    call :print_error "%step_name% failed!"
    exit /b 1
)

call :print_success "%step_name% completed successfully!"
exit /b 0

REM Color printing functions
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