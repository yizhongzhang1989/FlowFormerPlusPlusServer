#!/bin/bash

# Common utility functions for FlowFormer++ setup scripts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a directory exists and is not empty
directory_not_empty() {
    [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]
}

# Function to get file size in bytes
get_file_size() {
    if [ -f "$1" ]; then
        stat -c%s "$1" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to validate file exists and has minimum size
validate_file() {
    local file="$1"
    local min_size="${2:-1000000}"  # Default 1MB
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    local size=$(get_file_size "$file")
    [ "$size" -gt "$min_size" ]
}

# Function to run a step script
run_step() {
    local step_name="$1"
    local script_path="$2"
    
    print_step "$step_name"
    
    if [ ! -f "$script_path" ]; then
        print_error "Step script not found: $script_path"
        return 1
    fi
    
    if [ ! -x "$script_path" ]; then
        print_info "Making script executable: $script_path"
        chmod +x "$script_path"
    fi
    
    if bash "$script_path"; then
        print_success "$step_name completed successfully!"
        return 0
    else
        print_error "$step_name failed!"
        return 1
    fi
}
