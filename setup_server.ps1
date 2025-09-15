# FlowFormer++ Server Setup Script (PowerShell)
# This script automatically sets up the FlowFormer++ project after cloning

param(
    [switch]$Help
)

if ($Help) {
    Write-Host "FlowFormer++ Server Setup Script"
    Write-Host "Usage: .\setup_server.ps1"
    Write-Host ""
    Write-Host "This script will:"
    Write-Host "  1. Download model checkpoints (~325MB)"
    Write-Host "  2. Create conda environment with dependencies"
    Write-Host "  3. Create configuration file (config.json)"
    Write-Host "  4. Install web server dependencies"
    Write-Host "  5. Start the web server (default: http://localhost:5000)"
    exit 0
}

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptsDir = Join-Path $ScriptDir "scripts"

# Import common functions
. (Join-Path $ScriptsDir "common.ps1")

Write-Step "FlowFormer++ Server Setup"
Write-Info "Starting automatic setup process..."
Write-Info ""
Write-Info "This setup will:"
Write-Info "  1. Download model checkpoints (~325MB)"
Write-Info "  2. Create conda environment with dependencies"
Write-Info "  3. Create configuration file (config.json)"
Write-Info "  4. Install web server dependencies"
Write-Info "  5. Start the web server (default: http://localhost:5000)"
Write-Info ""
Write-Warning "Note: The web server will start automatically after setup!"
Write-Info "You can customize server settings by editing config.json"
Write-Info ""

# Define setup steps
$SetupSteps = @(
    @{
        Name = "Step 1: Download Model Checkpoints"
        Script = Join-Path $ScriptsDir "download_ckpts.ps1"
    },
    @{
        Name = "Step 2: Setup Conda Environment"
        Script = Join-Path $ScriptsDir "setup_conda_env.ps1"
    },
    @{
        Name = "Step 3: Setup and Start Web Server"
        Script = Join-Path $ScriptsDir "setup_webserver.ps1"
    }
)

# Function to run all setup steps
function Invoke-Setup {
    $TotalSteps = $SetupSteps.Count
    $CurrentStep = 0
    
    foreach ($Step in $SetupSteps) {
        $CurrentStep++
        
        Write-Info "Running $($Step.Name) ($CurrentStep/$TotalSteps)..."
        
        if (Invoke-Step $Step.Name $Step.Script) {
            Write-Success "$($Step.Name) completed!"
        } else {
            Write-Error "$($Step.Name) failed! Setup aborted."
            exit 1
        }
        
        Write-Host ""
    }
}

# Function to run a step script
function Invoke-Step {
    param(
        [string]$StepName,
        [string]$ScriptPath
    )
    
    Write-Step $StepName
    
    if (-not (Test-Path $ScriptPath)) {
        Write-Error "Step script not found: $ScriptPath"
        return $false
    }
    
    try {
        & $ScriptPath
        if ($LASTEXITCODE -eq 0) {
            Write-Success "$StepName completed successfully!"
            return $true
        } else {
            Write-Error "$StepName failed!"
            return $false
        }
    } catch {
        Write-Error "$StepName failed with error: $_"
        return $false
    }
}

# Main execution
Write-Info "Found $ScriptsDir for step scripts"

# Check if scripts directory exists
if (-not (Test-Path $ScriptsDir)) {
    Write-Error "Scripts directory not found: $ScriptsDir"
    exit 1
}

# Run all setup steps
Invoke-Setup

# Note: The script will end here because the web server keeps running
Write-Step "Setup Complete"
Write-Success "FlowFormer++ server setup completed successfully!"
Write-Info "The web server should now be running"
Write-Info "Check the output above for the server URL"
Write-Info "You can customize settings by editing config.json"