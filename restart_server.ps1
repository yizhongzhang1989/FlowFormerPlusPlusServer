# FlowFormer++ Web Server Restart Script (PowerShell)
# Use this script to restart the web server after initial setup

param(
    [switch]$Help
)

if ($Help) {
    Write-Host "FlowFormer++ Web Server Restart Script"
    Write-Host "Usage: .\restart_server.ps1"
    Write-Host ""
    Write-Host "This script restarts the FlowFormer++ web server after initial setup."
    Write-Host "Prerequisites:"
    Write-Host "  - flowformerpp conda environment (created by setup_server.ps1)"
    Write-Host "  - Model checkpoints (downloaded by setup_server.ps1)"
    Write-Host "  - Configuration file (config.json)"
    exit 0
}

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptsDir = Join-Path $ScriptDir "scripts"

# Import common functions if available
$CommonScript = Join-Path $ScriptsDir "common.ps1"
if (Test-Path $CommonScript) {
    . $CommonScript
} else {
    # Basic functions if common.ps1 not available
    function Write-Step { param([string]$Message) Write-Host "=== $Message ===" -ForegroundColor Cyan }
    function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Blue }
    function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
    function Write-Error { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
    function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
}

Write-Step "Restarting FlowFormer++ Web Server"

# Function to check if conda environment exists
function Test-CondaEnvironment {
    param([string]$EnvName)
    
    try {
        $envList = conda env list 2>$null
        return $envList -match $EnvName
    } catch {
        return $false
    }
}

# Function to get port from config.json
function Get-ServerPort {
    $defaultPort = 5000
    
    if (-not (Test-Path "config.json")) {
        return $defaultPort
    }
    
    try {
        $port = conda run -n flowformerpp python -c "import json; print(json.load(open('config.json'))['server']['port'])" 2>$null
        if ($port -and $port -match '^\d+$') {
            return [int]$port
        }
    } catch {
        Write-Warning "Could not read port from config.json, using default port $defaultPort"
    }
    
    return $defaultPort
}

# Check if conda environment exists
Write-Info "Checking conda environment..."
if (-not (Test-CondaEnvironment "flowformerpp")) {
    Write-Error "flowformerpp conda environment not found"
    Write-Error "Please run '.\setup_server.ps1' first to complete the initial setup"
    exit 1
}
Write-Success "Conda environment found"

# Check if model checkpoints exist
Write-Info "Checking model checkpoints..."
if (-not (Test-Path "checkpoints")) {
    Write-Error "Model checkpoints directory not found"
    Write-Error "Please run '.\setup_server.ps1' first to download checkpoints"
    exit 1
}

if (-not (Test-Path "checkpoints\sintel.pth")) {
    Write-Error "Model checkpoints not found"
    Write-Error "Please run '.\setup_server.ps1' first to download checkpoints"
    exit 1
}
Write-Success "Model checkpoints found"

# Check if config file exists
Write-Info "Checking configuration file..."
if (-not (Test-Path "config.json")) {
    Write-Error "Configuration file not found"
    Write-Error "Please run '.\setup_server.ps1' first to create config.json"
    exit 1
}
Write-Success "Configuration file found"

# Create necessary directories if they don't exist
Write-Info "Creating server directories..."
$directories = @("tmp", "tmp\uploads", "tmp\results")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "Server directories ready"

# Get server port
$port = Get-ServerPort

Write-Host ""
Write-Success "Starting web server..."
Write-Info "The server will be available at: http://localhost:$port"
Write-Warning "Press Ctrl+C to stop the server"
Write-Host ""

# Start the Flask application
try {
    conda run -n flowformerpp python app.py
} catch {
    Write-Error "Failed to start the web server: $_"
    exit 1
}