@echo off
setlocal enabledelayedexpansion

REM FlowFormer++ Checkpoint Download Script
REM This script downloads the required model checkpoints from Google Drive

REM Source the common functions
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%\common.bat" :init

call :print_info "=== Downloading Model Checkpoints ==="

set "CHECKPOINTS_DIR=checkpoints"

REM Required checkpoint files with their Google Drive file IDs
set "chairs_pth_id=1qpokIjlUhvex99-IfzRJ04HiNfvBZOtm"
set "kitti_pth_id=1i04ie6fbAii9uGFYpu99Y7rv9G7miTaH"
set "sintel_pth_id=1c5NYSh7Dbc94wppPltX2Q_ashAb4QxTB"
set "things_288960_pth_id=1lqyusjgnhtG7hc1mxbBsb8VSbuCMugyX"
set "things_pth_id=1YOJXzv80MjKg8GbA_lpbHD25JGlsuAqC"

REM File names array
set files=chairs.pth kitti.pth sintel.pth things_288960.pth things.pth

REM Check if gdown is installed
call :check_gdown
if %errorlevel% neq 0 exit /b 1

REM Download checkpoints
call :download_checkpoints
if %errorlevel% neq 0 exit /b 1

REM Verify checkpoints
call :verify_checkpoints
if %errorlevel% neq 0 exit /b 1

call :print_success "Checkpoint download completed successfully!"
exit /b 0

:check_gdown
call :command_exists gdown
if %errorlevel% neq 0 (
    call :print_warning "gdown is not installed. Installing gdown..."
    pip install gdown
    if !errorlevel! equ 0 (
        call :print_success "gdown installed successfully"
    ) else (
        call :print_error "Failed to install gdown. Please install it manually: pip install gdown"
        exit /b 1
    )
)
exit /b 0

:download_checkpoints
call :print_info "Creating checkpoints directory..."
if not exist "%CHECKPOINTS_DIR%" mkdir "%CHECKPOINTS_DIR%"

call :print_info "Downloading checkpoints from Google Drive..."

pushd "%CHECKPOINTS_DIR%"

REM Download each checkpoint file individually
for %%f in (%files%) do (
    if not exist "%%f" (
        call :print_info "Downloading %%f..."
        
        REM Get the file ID for this file
        for /f "tokens=2 delims=." %%a in ("%%f") do set ext=%%a
        for /f "tokens=1 delims=." %%b in ("%%f") do set basename=%%b
        
        REM Construct variable name and get file ID
        set var_name=!basename!_!ext!_id
        if "!basename!"=="things_288960" set var_name=things_288960_pth_id
        call set file_id=%%!var_name!%%
        
        if defined file_id (
            REM Use gdown to download the file
            gdown "https://drive.google.com/uc?id=!file_id!" -O "%%f"
            if !errorlevel! equ 0 (
                call :print_success "Successfully downloaded %%f"
            ) else (
                call :print_error "Failed to download %%f"
                call :print_info "You can manually download from: https://drive.google.com/file/d/!file_id!/view"
                popd
                exit /b 1
            )
        ) else (
            call :print_error "Could not find file ID for %%f"
            popd
            exit /b 1
        )
    ) else (
        call :print_info "%%f already exists, skipping download"
    )
)

popd
exit /b 0

:verify_checkpoints
call :print_info "Verifying downloaded checkpoints..."
set all_present=true

for %%f in (%files%) do (
    if exist "%CHECKPOINTS_DIR%\%%f" (
        call :get_file_size "%CHECKPOINTS_DIR%\%%f"
        if !file_size! gtr 1000000 (
            call :print_success "✓ %%f (!file_size! bytes)"
        ) else (
            call :print_warning "✗ %%f appears to be incomplete (!file_size! bytes)"
            set all_present=false
        )
    ) else (
        call :print_error "✗ %%f is missing"
        set all_present=false
    )
)

if "!all_present!"=="true" (
    call :print_success "All checkpoints are ready!"
    exit /b 0
) else (
    call :print_error "Some checkpoints are missing or incomplete."
    exit /b 1
)

REM Include common functions inline since we can't source them in batch
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

:command_exists
where %1 >nul 2>&1
exit /b %errorlevel%

:get_file_size
if exist "%~1" (
    for %%i in ("%~1") do set file_size=%%~zi
) else (
    set file_size=0
)
exit /b 0