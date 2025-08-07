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
    print_info "The server will be available at: http://localhost:5000"
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
    
    # Verify setup
    verify_setup
    
    print_success "Web server setup completed!"
    print_info ""
    print_info "Starting the web server now..."
    print_info "After the server starts, you can:"
    print_info "  1. Open your browser to http://localhost:5000"
    print_info "  2. Upload two images to compute optical flow"
    print_info "  3. View and download flow visualizations"
    print_info ""
    
    # Start the server
    start_server
}

# Run main function
main "$@"
