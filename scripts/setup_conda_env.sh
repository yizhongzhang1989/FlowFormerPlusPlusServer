#!/bin/bash

# FlowFormer++ Conda Environment Setup Script
# This script creates and configures the flowformerpp conda environment

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/common.sh"

print_info "=== Setting up Conda Environment ==="

ENV_NAME="flowformerpp"
REQUIREMENTS_FILE="requirements.txt"

# Function to check if conda is installed
check_conda() {
    if ! command_exists conda; then
        print_error "Conda is not installed or not in PATH"
        print_info "Please install Anaconda or Miniconda first:"
        print_info "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda is available"
}

# Function to check if environment exists
env_exists() {
    conda env list | grep -q "^$ENV_NAME "
}

# Function to create conda environment
create_conda_env() {
    print_info "Creating conda environment: $ENV_NAME"
    
    if conda create -n "$ENV_NAME" python=3.8 -y; then
        print_success "Conda environment '$ENV_NAME' created successfully"
    else
        print_error "Failed to create conda environment '$ENV_NAME'"
        exit 1
    fi
}

# Function to install conda packages
install_conda_packages() {
    print_info "Installing conda packages..."
    
    # Activate environment and install packages
    if conda run -n "$ENV_NAME" conda install \
        pytorch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 pytorch-cuda=11.8 \
        matplotlib tensorboard scipy opencv \
        -c pytorch -c nvidia -y; then
        print_success "Conda packages installed successfully"
    else
        print_error "Failed to install conda packages"
        exit 1
    fi
}

# Function to install pip packages
install_pip_packages() {
    print_info "Installing pip packages..."
    
    # Check if requirements.txt exists
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_warning "requirements.txt not found, installing packages directly"
        if conda run -n "$ENV_NAME" pip install yacs loguru einops timm==0.4.12 imageio; then
            print_success "Pip packages installed successfully"
        else
            print_error "Failed to install pip packages"
            exit 1
        fi
    else
        print_info "Installing packages from $REQUIREMENTS_FILE"
        if conda run -n "$ENV_NAME" pip install -r "$REQUIREMENTS_FILE"; then
            print_success "Pip packages installed from requirements.txt"
        else
            print_error "Failed to install packages from requirements.txt"
            exit 1
        fi
    fi
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test PyTorch installation
    if conda run -n "$ENV_NAME" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
        print_success "PyTorch verification passed"
    else
        print_error "PyTorch verification failed"
        exit 1
    fi
    
    # Test other key packages
    local packages=("torchvision" "matplotlib" "scipy" "cv2" "yacs" "loguru" "einops" "timm" "imageio")
    for pkg in "${packages[@]}"; do
        if conda run -n "$ENV_NAME" python -c "import $pkg" 2>/dev/null; then
            print_success "✓ $pkg imported successfully"
        else
            print_warning "✗ Failed to import $pkg"
        fi
    done
}

# Function to display environment info
display_env_info() {
    print_info "Environment setup complete!"
    print_info "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    print_info "To deactivate the environment, run:"
    echo "  conda deactivate"
}

# Main execution
main() {
    # Check if conda is available
    check_conda
    
    # Check if environment already exists
    if env_exists; then
        print_success "Conda environment '$ENV_NAME' already exists"
        print_info "Skipping environment creation"
        
        # Still verify that packages are installed
        print_info "Verifying existing environment..."
        verify_installation
    else
        print_info "Conda environment '$ENV_NAME' does not exist"
        
        # Create environment
        create_conda_env
        
        # Install conda packages
        install_conda_packages
        
        # Install pip packages
        install_pip_packages
        
        # Verify installation
        verify_installation
    fi
    
    # Display usage information
    display_env_info
    
    print_success "Conda environment setup completed successfully!"
}

# Run main function
main "$@"
