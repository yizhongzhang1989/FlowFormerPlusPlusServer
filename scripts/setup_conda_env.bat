@echo off
setlocal enabledelayedexpansion

REM FlowFormer++ Conda Environment Setup Script
REM This script creates and configures the flowformerpp conda environment

REM Source common functions
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%\common.bat" :init

call :print_info "=== Setting up Conda Environment ==="

set "ENV_NAME=flowformerpp"
set "REQUIREMENTS_FILE=requirements.txt"

REM Check if conda is installed
call :check_conda
if %errorlevel% neq 0 exit /b 1

REM Check if environment already exists
call :env_exists
if %errorlevel% equ 0 (
    call :print_success "Conda environment '%ENV_NAME%' already exists"
    call :print_info "Skipping environment creation"
    
    REM Still verify that packages are installed
    call :print_info "Verifying existing environment..."
    call :verify_installation
) else (
    call :print_info "Conda environment '%ENV_NAME%' does not exist"
    
    REM Create environment
    call :create_conda_env
    if !errorlevel! neq 0 exit /b 1
    
    REM Install conda packages
    call :install_conda_packages
    if !errorlevel! neq 0 exit /b 1
    
    REM Install pip packages
    call :install_pip_packages
    if !errorlevel! neq 0 exit /b 1
    
    REM Verify installation
    call :verify_installation
)

REM Display usage information
call :display_env_info

call :print_success "Conda environment setup completed successfully!"
exit /b 0

:check_conda
call :command_exists conda
if %errorlevel% neq 0 (
    call :print_error "Conda is not installed or not in PATH"
    call :print_info "Please install Anaconda or Miniconda first:"
    call :print_info "https://docs.conda.io/en/latest/miniconda.html"
    exit /b 1
)
call :print_success "Conda is available"
exit /b 0

:env_exists
conda env list | findstr /b "%ENV_NAME% " >nul 2>&1
exit /b %errorlevel%

:create_conda_env
call :print_info "Creating conda environment: %ENV_NAME%"

conda create -n "%ENV_NAME%" python=3.8 -y
if %errorlevel% equ 0 (
    call :print_success "Conda environment '%ENV_NAME%' created successfully"
    exit /b 0
) else (
    call :print_error "Failed to create conda environment '%ENV_NAME%'"
    exit /b 1
)

:install_conda_packages
call :print_info "Installing conda packages..."

REM Activate environment and install packages
conda run -n "%ENV_NAME%" conda install pytorch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 pytorch-cuda=11.8 matplotlib tensorboard scipy opencv -c pytorch -c nvidia -y
if %errorlevel% equ 0 (
    call :print_success "Conda packages installed successfully"
    exit /b 0
) else (
    call :print_error "Failed to install conda packages"
    exit /b 1
)

:install_pip_packages
call :print_info "Installing pip packages..."

REM Check if requirements.txt exists
if not exist "%REQUIREMENTS_FILE%" (
    call :print_warning "requirements.txt not found, installing packages directly"
    conda run -n "%ENV_NAME%" pip install yacs loguru einops timm==0.4.12 imageio flask werkzeug
    if !errorlevel! equ 0 (
        call :print_success "Pip packages installed successfully"
        exit /b 0
    ) else (
        call :print_error "Failed to install pip packages"
        exit /b 1
    )
) else (
    call :print_info "Installing packages from %REQUIREMENTS_FILE%"
    conda run -n "%ENV_NAME%" pip install -r "%REQUIREMENTS_FILE%"
    if !errorlevel! equ 0 (
        call :print_success "Pip packages installed from requirements.txt"
        exit /b 0
    ) else (
        call :print_error "Failed to install packages from requirements.txt"
        exit /b 1
    )
)

:verify_installation
call :print_info "Verifying installation..."

REM Test PyTorch installation
conda run -n "%ENV_NAME%" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if %errorlevel% equ 0 (
    call :print_success "PyTorch verification passed"
) else (
    call :print_error "PyTorch verification failed"
    exit /b 1
)

REM Test other key packages
set packages=torchvision matplotlib scipy cv2 yacs loguru einops timm imageio flask

for %%p in (%packages%) do (
    conda run -n "%ENV_NAME%" python -c "import %%p" >nul 2>&1
    if !errorlevel! equ 0 (
        call :print_success "✓ %%p imported successfully"
    ) else (
        call :print_warning "✗ Failed to import %%p"
    )
)
exit /b 0

:display_env_info
call :print_info "Environment setup complete!"
call :print_info "To activate the environment, run:"
echo   conda activate %ENV_NAME%
call :print_info "To deactivate the environment, run:"
echo   conda deactivate
exit /b 0

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

:command_exists
where %1 >nul 2>&1
exit /b %errorlevel%