# FlowFormer++ Checkpoint Download Script (PowerShell)
# This script downloads the required model checkpoints from Google Drive

# Source the common functions
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "common.ps1")

Write-Info "=== Downloading Model Checkpoints ==="

$CheckpointsDir = "checkpoints"

# Required checkpoint files with their Google Drive file IDs
$CheckpointFiles = @{
    "chairs.pth" = "1qpokIjlUhvex99-IfzRJ04HiNfvBZOtm"
    "kitti.pth" = "1i04ie6fbAii9uGFYpu99Y7rv9G7miTaH"
    "sintel.pth" = "1c5NYSh7Dbc94wppPltX2Q_ashAb4QxTB"
    "things_288960.pth" = "1lqyusjgnhtG7hc1mxbBsb8VSbuCMugyX"
    "things.pth" = "1YOJXzv80MjKg8GbA_lpbHD25JGlsuAqC"
}

# Function to check if gdown is installed
function Test-GdownInstallation {
    if (-not (Test-Command "gdown")) {
        Write-Warning "gdown is not installed. Installing gdown..."
        try {
            & pip install gdown
            if ($LASTEXITCODE -eq 0) {
                Write-Success "gdown installed successfully"
                return $true
            } else {
                Write-Error "Failed to install gdown. Please install it manually: pip install gdown"
                return $false
            }
        } catch {
            Write-Error "Failed to install gdown. Please install it manually: pip install gdown"
            return $false
        }
    }
    return $true
}

# Function to download checkpoints
function Invoke-DownloadCheckpoints {
    Write-Info "Creating checkpoints directory..."
    if (-not (Test-Path $CheckpointsDir)) {
        New-Item -ItemType Directory -Path $CheckpointsDir | Out-Null
    }
    
    Write-Info "Downloading checkpoints from Google Drive..."
    
    Push-Location $CheckpointsDir
    
    try {
        # Download each checkpoint file individually
        foreach ($File in $CheckpointFiles.Keys) {
            if (-not (Test-Path $File)) {
                Write-Info "Downloading $File..."
                $FileId = $CheckpointFiles[$File]
                
                # Use gdown to download the file
                try {
                    & gdown "https://drive.google.com/uc?id=$FileId" -O $File
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Successfully downloaded $File"
                    } else {
                        Write-Error "Failed to download $File"
                        Write-Info "You can manually download from: https://drive.google.com/file/d/$FileId/view"
                        Pop-Location
                        return $false
                    }
                } catch {
                    Write-Error "Failed to download $File : $_"
                    Write-Info "You can manually download from: https://drive.google.com/file/d/$FileId/view"
                    Pop-Location
                    return $false
                }
            } else {
                Write-Info "$File already exists, skipping download"
            }
        }
        
        Pop-Location
        return $true
    } catch {
        Pop-Location
        throw
    }
}

# Function to verify downloaded checkpoints
function Test-CheckpointVerification {
    Write-Info "Verifying downloaded checkpoints..."
    $AllPresent = $true
    
    foreach ($File in $CheckpointFiles.Keys) {
        $FilePath = Join-Path $CheckpointsDir $File
        if (Test-Path $FilePath) {
            $FileSize = Get-FileSize $FilePath
            if ($FileSize -gt 1000000) {  # Check if file is larger than 1MB
                Write-Success "✓ $File ($FileSize bytes)"
            } else {
                Write-Warning "✗ $File appears to be incomplete ($FileSize bytes)"
                $AllPresent = $false
            }
        } else {
            Write-Error "✗ $File is missing"
            $AllPresent = $false
        }
    }
    
    if ($AllPresent) {
        Write-Success "All checkpoints are ready!"
        return $true
    } else {
        Write-Error "Some checkpoints are missing or incomplete."
        return $false
    }
}

# Main execution
function Main {
    # Check if checkpoints directory exists and has the required files
    if (Test-Path $CheckpointsDir) {
        Write-Info "Checkpoints directory exists. Checking for required files..."
        
        $MissingFiles = @()
        foreach ($File in $CheckpointFiles.Keys) {
            $FilePath = Join-Path $CheckpointsDir $File
            if (-not (Test-Path $FilePath)) {
                $MissingFiles += $File
            }
        }
        
        if ($MissingFiles.Count -eq 0) {
            Write-Success "All checkpoint files are present!"
        } else {
            Write-Warning "Missing checkpoint files: $($MissingFiles -join ', ')"
            Write-Info "Downloading missing checkpoints..."
            if (-not (Test-GdownInstallation)) {
                exit 1
            }
            if (-not (Invoke-DownloadCheckpoints)) {
                Write-Error "Failed to download some checkpoints automatically."
                exit 1
            }
        }
    } else {
        Write-Info "Checkpoints directory does not exist. Creating and downloading..."
        if (-not (Test-GdownInstallation)) {
            exit 1
        }
        if (-not (Invoke-DownloadCheckpoints)) {
            Write-Error "Failed to download checkpoints automatically."
            exit 1
        }
    }

    # Verify checkpoints
    if (Test-CheckpointVerification) {
        Write-Success "Checkpoint download completed successfully!"
        exit 0
    } else {
        Write-Error "Checkpoint verification failed."
        exit 1
    }
}

# Run main function
Main