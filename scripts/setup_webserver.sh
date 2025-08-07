#!/bin/bash

# FlowFormer++ Web Server Setup Script
# This script installs Flask dependencies and starts the web server

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/common.sh"

print_info "=== Setting up Web Server ==="

# Function to install Flask dependencies
install_flask_dependencies() {
    print_info "Installing Flask dependencies..."
    
    if conda run -n flowformerpp pip install flask werkzeug; then
        print_success "Flask dependencies installed successfully"
    else
        print_error "Failed to install Flask dependencies"
        exit 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_info "Creating server directories..."
    
    mkdir -p tmp/uploads tmp/results
    print_success "Server directories created"
}

# Function to create config file
create_config_file() {
    print_info "Creating server configuration file..."
    
    # Check if config.json already exists
    if [ -f "config.json" ]; then
        print_info "Configuration file already exists, skipping creation"
        return 0
    fi
    
    # Create config.json with default settings
    cat > config.json << 'EOF'
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
EOF
    
    if [ -f "config.json" ]; then
        print_success "Configuration file created successfully"
        print_info "You can edit config.json to customize server settings"
    else
        print_error "Failed to create configuration file"
        exit 1
    fi
}

# Function to verify web server setup
verify_setup() {
    print_info "Verifying web server setup..."
    
    # Check if Flask is installed
    if conda run -n flowformerpp python -c "import flask; print(f'Flask version: {flask.__version__}')" 2>/dev/null; then
        print_success "✓ Flask installed and working"
    else
        print_error "✗ Flask not properly installed"
        exit 1
    fi
    
    # Check if app.py exists
    if [ -f "app.py" ]; then
        print_success "✓ Web server app found"
    else
        print_error "✗ app.py not found"
        exit 1
    fi
    
    # Check if config.json exists
    if [ -f "config.json" ]; then
        print_success "✓ Configuration file found"
    else
        print_error "✗ config.json not found"
        exit 1
    fi
    
    # Check if checkpoints exist
    if [ -f "checkpoints/sintel.pth" ]; then
        print_success "✓ Model checkpoints available"
    else
        print_error "✗ Model checkpoints not found"
        exit 1
    fi
}

# Function to start the web server
start_server() {
    print_info "Starting FlowFormer++ web server..."
    
    # Read port from config.json if it exists
    local port=5000
    if [ -f "config.json" ] && command -v python3 &> /dev/null; then
        port=$(python3 -c "import json; print(json.load(open('config.json'))['server']['port'])" 2>/dev/null || echo "5000")
    fi
    
    print_info "The server will be available at: http://localhost:$port"
    print_info "Press Ctrl+C to stop the server"
    print_info ""
    
    # Start the Flask application
    conda run -n flowformerpp python app.py
}

# Main execution
main() {
    # Install Flask dependencies
    install_flask_dependencies
    
    # Create necessary directories
    create_directories
    
    # Create configuration file
    create_config_file
    
    # Verify setup
    verify_setup
    
    print_success "Web server setup completed!"
    print_info ""
    print_info "Starting the web server now..."
    print_info "After the server starts, you can:"
    print_info "  1. Open your browser to the displayed URL"
    print_info "  2. Upload two images to compute optical flow"
    print_info "  3. View and download flow visualizations"
    print_info "  4. Modify config.json to customize server settings"
    print_info ""
    
    # Start the server
    start_server
}

# Run main function
main "$@"
