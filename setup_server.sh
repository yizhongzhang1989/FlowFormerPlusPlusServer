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

# Define setup steps
declare -a SETUP_STEPS=(
    "Step 1: Download Model Checkpoints|$SCRIPTS_DIR/download_ckpts.sh"
    # Add more steps here as needed:
    # "Step 2: Install Dependencies|$SCRIPTS_DIR/install_deps.sh"
    # "Step 3: Setup Environment|$SCRIPTS_DIR/setup_env.sh"
    # "Step 4: Compile CUDA Extensions|$SCRIPTS_DIR/compile_cuda.sh"
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
    
    print_step "Setup Complete"
    print_success "FlowFormer++ server setup completed successfully!"
    print_info "You can now start using the FlowFormer++ project."
}

# Run main function
main "$@"
