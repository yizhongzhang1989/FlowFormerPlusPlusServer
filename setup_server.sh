#!/bin/bash

# FlowFormer++ Server Setup Script
# This script automatically sets up the FlowFormer++ project after cloning

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# Source common functions
if [ -f "$SCRIPTS_DIR/common.sh" ]; then
    source "$SCRIPTS_DIR/common.sh"
else
    echo "Error: common.sh not found in $SCRIPTS_DIR"
    exit 1
fi

print_step "FlowFormer++ Server Setup"
print_info "Starting automatic setup process..."
print_info ""
print_info "This setup will:"
print_info "  1. Download model checkpoints (~325MB)"
print_info "  2. Create conda environment with dependencies"
print_info "  3. Create configuration file (config.json)"
print_info "  4. Install web server dependencies"
print_info "  5. Start the web server (default: http://localhost:5000)"
print_info ""
print_warning "Note: The web server will start automatically after setup!"
print_info "You can customize server settings by editing config.json"
print_info ""

# Define setup steps
declare -a SETUP_STEPS=(
    "Step 1: Download Model Checkpoints|$SCRIPTS_DIR/download_ckpts.sh"
    "Step 2: Setup Conda Environment|$SCRIPTS_DIR/setup_conda_env.sh"
    "Step 3: Setup and Start Web Server|$SCRIPTS_DIR/setup_webserver.sh"
)

# Function to run all setup steps
run_setup() {
    local total_steps=${#SETUP_STEPS[@]}
    local current_step=0
    
    for step_info in "${SETUP_STEPS[@]}"; do
        current_step=$((current_step + 1))
        
        # Parse step info
        IFS='|' read -r step_name script_path <<< "$step_info"
        
        print_info "Running $step_name ($current_step/$total_steps)..."
        
        if run_step "$step_name" "$script_path"; then
            print_success "$step_name completed!"
        else
            print_error "$step_name failed! Setup aborted."
            exit 1
        fi
        
        echo ""
    done
}

# Main execution
main() {
    print_info "Found $SCRIPTS_DIR for step scripts"
    
    # Check if scripts directory exists
    if [ ! -d "$SCRIPTS_DIR" ]; then
        print_error "Scripts directory not found: $SCRIPTS_DIR"
        exit 1
    fi
    
    # Run all setup steps
    run_setup
    
    # Note: The script will end here because the web server keeps running
    print_step "Setup Complete"
    print_success "FlowFormer++ server setup completed successfully!"
    print_info "The web server should now be running"
    print_info "Check the output above for the server URL"
    print_info "You can customize settings by editing config.json"
}

# Run main function
main "$@"
