@echo off

REM Common utility functions for FlowFormer++ setup scripts

REM Function to print colored output (simplified for batch)
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

REM Function to check if a command exists
:command_exists
where %1 >nul 2>&1
exit /b %errorlevel%

REM Function to check if a directory exists and is not empty
:directory_not_empty
if not exist "%~1" exit /b 1
for /f %%i in ('dir /b "%~1" 2^>nul ^| find /c /v ""') do (
    if %%i gtr 0 (
        exit /b 0
    ) else (
        exit /b 1
    )
)

REM Function to get file size in bytes
:get_file_size
if exist "%~1" (
    for %%i in ("%~1") do set file_size=%%~zi
) else (
    set file_size=0
)
exit /b 0

REM Function to validate file exists and has minimum size
:validate_file
set file_path=%~1
set min_size=%~2
if "%min_size%"=="" set min_size=1000000

if not exist "%file_path%" exit /b 1

call :get_file_size "%file_path%"
if %file_size% gtr %min_size% (
    exit /b 0
) else (
    exit /b 1
)

REM Function to run a step script
:run_step
set step_name=%~1
set script_path=%~2

call :print_step "%step_name%"

if not exist "%script_path%" (
    call :print_error "Step script not found: %script_path%"
    exit /b 1
)

call "%script_path%"
if %errorlevel% neq 0 (
    call :print_error "%step_name% failed!"
    exit /b 1
)

call :print_success "%step_name% completed successfully!"
exit /b 0

:init
REM Initialize common functions - placeholder for future use
exit /b 0