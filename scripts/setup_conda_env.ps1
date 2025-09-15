# FlowFormer++ Conda Environment Setup Script (PowerShell)
# This script creates and configures the flowformerpp conda environment

# Source common functions
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "common.ps1")

Write-Info "=== Setting up Conda Environment ==="

$EnvName = "flowformerpp"
$RequirementsFile = "requirements.txt"

# Function to check if conda is installed
function Test-CondaInstallation {
    if (-not (Test-Command "conda")) {
        Write-Error "Conda is not installed or not in PATH"
        Write-Info "Please install Anaconda or Miniconda first:"
        Write-Info "https://docs.conda.io/en/latest/miniconda.html"
        return $false
    }
    Write-Success "Conda is available"
    return $true
}

# Function to check if environment exists
function Test-EnvironmentExists {
    try {
        $EnvList = & conda env list 2>$null
        foreach ($Line in $EnvList) {
            if ($Line -match "^$EnvName\s") {
                return $true
            }
        }
        return $false
    } catch {
        return $false
    }
}

# Function to create conda environment
function New-CondaEnvironment {
    Write-Info "Creating conda environment: $EnvName"
    
    try {
        & conda create -n $EnvName python=3.8 -y
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Conda environment '$EnvName' created successfully"
            return $true
        } else {
            Write-Error "Failed to create conda environment '$EnvName'"
            return $false
        }
    } catch {
        Write-Error "Failed to create conda environment '$EnvName': $_"
        return $false
    }
}

# Function to install conda packages
function Install-CondaPackages {
    Write-Info "Installing conda packages..."
    
    try {
        # Activate environment and install packages
        $Command = "conda install pytorch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 pytorch-cuda=11.8 matplotlib tensorboard scipy opencv -c pytorch -c nvidia -y"
        & conda run -n $EnvName conda install pytorch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 pytorch-cuda=11.8 matplotlib tensorboard scipy opencv -c pytorch -c nvidia -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Conda packages installed successfully"
            return $true
        } else {
            Write-Error "Failed to install conda packages"
            return $false
        }
    } catch {
        Write-Error "Failed to install conda packages: $_"
        return $false
    }
}

# Function to install pip packages
function Install-PipPackages {
    Write-Info "Installing pip packages..."
    
    try {
        # Check if requirements.txt exists
        if (-not (Test-Path $RequirementsFile)) {
            Write-Warning "requirements.txt not found, installing packages directly"
            & conda run -n $EnvName pip install yacs loguru einops timm==0.4.12 imageio flask werkzeug
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Pip packages installed successfully"
                return $true
            } else {
                Write-Error "Failed to install pip packages"
                return $false
            }
        } else {
            Write-Info "Installing packages from $RequirementsFile"
            & conda run -n $EnvName pip install -r $RequirementsFile
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Pip packages installed from requirements.txt"
                return $true
            } else {
                Write-Error "Failed to install packages from requirements.txt"
                return $false
            }
        }
    } catch {
        Write-Error "Failed to install pip packages: $_"
        return $false
    }
}

# Function to verify installation
function Test-InstallationVerification {
    Write-Info "Verifying installation..."
    
    try {
        # Test PyTorch installation
        & conda run -n $EnvName python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "PyTorch verification passed"
        } else {
            Write-Error "PyTorch verification failed"
            return $false
        }
        
        # Test other key packages
        $Packages = @("torchvision", "matplotlib", "scipy", "cv2", "yacs", "loguru", "einops", "timm", "imageio", "flask")
        foreach ($Package in $Packages) {
            try {
                & conda run -n $EnvName python -c "import $Package" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "✓ $Package imported successfully"
                } else {
                    Write-Warning "✗ Failed to import $Package"
                }
            } catch {
                Write-Warning "✗ Failed to import $Package"
            }
        }
        
        return $true
    } catch {
        Write-Error "Installation verification failed: $_"
        return $false
    }
}

# Function to display environment info
function Show-EnvironmentInfo {
    Write-Info "Environment setup complete!"
    Write-Info "To activate the environment, run:"
    Write-Host "  conda activate $EnvName"
    Write-Info "To deactivate the environment, run:"
    Write-Host "  conda deactivate"
}

# Main execution
function Main {
    # Check if conda is available
    if (-not (Test-CondaInstallation)) {
        exit 1
    }
    
    # Check if environment already exists
    if (Test-EnvironmentExists) {
        Write-Success "Conda environment '$EnvName' already exists"
        Write-Info "Skipping environment creation"
        
        # Still verify that packages are installed
        Write-Info "Verifying existing environment..."
        if (-not (Test-InstallationVerification)) {
            exit 1
        }
    } else {
        Write-Info "Conda environment '$EnvName' does not exist"
        
        # Create environment
        if (-not (New-CondaEnvironment)) {
            exit 1
        }
        
        # Install conda packages
        if (-not (Install-CondaPackages)) {
            exit 1
        }
        
        # Install pip packages
        if (-not (Install-PipPackages)) {
            exit 1
        }
        
        # Verify installation
        if (-not (Test-InstallationVerification)) {
            exit 1
        }
    }
    
    # Display usage information
    Show-EnvironmentInfo
    
    Write-Success "Conda environment setup completed successfully!"
}

# Run main function
Main