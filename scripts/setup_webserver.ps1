# FlowFormer++ Web Server Setup Script (PowerShell)
# This script installs Flask dependencies and starts the web server

# Source common functions
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "common.ps1")

Write-Info "=== Setting up Web Server ==="

# Function to install Flask dependencies
function Install-FlaskDependencies {
    Write-Info "Installing Flask dependencies..."
    
    try {
        & conda run -n flowformerpp pip install flask werkzeug
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Flask dependencies installed successfully"
            return $true
        } else {
            Write-Error "Failed to install Flask dependencies"
            return $false
        }
    } catch {
        Write-Error "Failed to install Flask dependencies: $_"
        return $false
    }
}

# Function to create necessary directories
function New-ServerDirectories {
    Write-Info "Creating server directories..."
    
    try {
        if (-not (Test-Path "tmp\uploads")) {
            New-Item -ItemType Directory -Path "tmp\uploads" -Force | Out-Null
        }
        if (-not (Test-Path "tmp\results")) {
            New-Item -ItemType Directory -Path "tmp\results" -Force | Out-Null
        }
        Write-Success "Server directories created"
        return $true
    } catch {
        Write-Error "Failed to create server directories: $_"
        return $false
    }
}

# Function to create config file
function New-ConfigFile {
    Write-Info "Creating server configuration file..."
    
    # Check if config.json already exists
    if (Test-Path "config.json") {
        Write-Info "Configuration file already exists, skipping creation"
        return $true
    }
    
    try {
        # Create config.json with default settings
        $ConfigContent = @'
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  },
  "model": {
    "checkpoint_path": "checkpoints/sintel.pth",
    "device": "auto",
    "max_image_size": 1024
  },
  "upload": {
    "max_file_size_mb": 10,
    "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"],
    "upload_folder": "tmp/uploads",
    "results_folder": "tmp/results"
  },
  "processing": {
    "auto_cleanup": true,
    "cleanup_interval_hours": 24,
    "max_stored_results": 100
  }
}
'@
        
        $ConfigContent | Out-File -FilePath "config.json" -Encoding UTF8
        
        if (Test-Path "config.json") {
            Write-Success "Configuration file created successfully"
            Write-Info "You can edit config.json to customize server settings"
            return $true
        } else {
            Write-Error "Failed to create configuration file"
            return $false
        }
    } catch {
        Write-Error "Failed to create configuration file: $_"
        return $false
    }
}

# Function to verify web server setup
function Test-ServerSetup {
    Write-Info "Verifying web server setup..."
    
    try {
        # Check if Flask is installed
        & conda run -n flowformerpp python -c "import flask; print(f'Flask version: {flask.__version__}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✓ Flask installed and working"
        } else {
            Write-Error "✗ Flask not properly installed"
            return $false
        }
        
        # Check if app.py exists
        if (Test-Path "app.py") {
            Write-Success "✓ Web server app found"
        } else {
            Write-Error "✗ app.py not found"
            return $false
        }
        
        # Check if config.json exists
        if (Test-Path "config.json") {
            Write-Success "✓ Configuration file found"
        } else {
            Write-Error "✗ config.json not found"
            return $false
        }
        
        # Check if checkpoints exist
        if (Test-Path "checkpoints\sintel.pth") {
            Write-Success "✓ Model checkpoints available"
        } else {
            Write-Error "✗ Model checkpoints not found"
            return $false
        }
        
        return $true
    } catch {
        Write-Error "Server setup verification failed: $_"
        return $false
    }
}

# Function to start the web server
function Start-WebServer {
    Write-Info "Starting FlowFormer++ web server..."
    
    # Read port from config.json if it exists
    $Port = 5000
    if (Test-Path "config.json") {
        try {
            $Config = Get-Content "config.json" | ConvertFrom-Json
            $Port = $Config.server.port
        } catch {
            Write-Warning "Could not read port from config.json, using default port 5000"
        }
    }
    
    Write-Info "The server will be available at: http://localhost:$Port"
    Write-Info "Press Ctrl+C to stop the server"
    Write-Info ""
    
    try {
        # Start the Flask application
        & conda run -n flowformerpp python app.py
        return ($LASTEXITCODE -eq 0)
    } catch {
        Write-Error "Failed to start web server: $_"
        return $false
    }
}

# Main execution
function Main {
    # Install Flask dependencies
    if (-not (Install-FlaskDependencies)) {
        exit 1
    }
    
    # Create necessary directories
    if (-not (New-ServerDirectories)) {
        exit 1
    }
    
    # Create configuration file
    if (-not (New-ConfigFile)) {
        exit 1
    }
    
    # Verify setup
    if (-not (Test-ServerSetup)) {
        exit 1
    }
    
    Write-Success "Web server setup completed!"
    Write-Info ""
    Write-Info "Starting the web server now..."
    Write-Info "After the server starts, you can:"
    Write-Info "  1. Open your browser to the displayed URL"
    Write-Info "  2. Upload two images to compute optical flow"
    Write-Info "  3. View and download flow visualizations"
    Write-Info "  4. Modify config.json to customize server settings"
    Write-Info ""
    
    # Start the server
    if (-not (Start-WebServer)) {
        exit 1
    }
}

# Run main function
Main