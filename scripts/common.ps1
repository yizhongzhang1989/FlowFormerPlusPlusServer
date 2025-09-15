# Common utility functions for FlowFormer++ setup scripts (PowerShell)

# Function to print colored output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "=== $Message ===" -ForegroundColor Blue
}

# Function to check if a command exists
function Test-Command {
    param([string]$CommandName)
    
    try {
        Get-Command $CommandName -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to check if a directory exists and is not empty
function Test-DirectoryNotEmpty {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return $false
    }
    
    $Items = Get-ChildItem $Path -Force -ErrorAction SilentlyContinue
    return ($Items.Count -gt 0)
}

# Function to get file size in bytes
function Get-FileSize {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        return (Get-Item $FilePath).Length
    } else {
        return 0
    }
}

# Function to validate file exists and has minimum size
function Test-ValidateFile {
    param(
        [string]$FilePath,
        [int]$MinSize = 1000000  # Default 1MB
    )
    
    if (-not (Test-Path $FilePath)) {
        return $false
    }
    
    $Size = Get-FileSize $FilePath
    return ($Size -gt $MinSize)
}

# Function to run a step script
function Invoke-StepScript {
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