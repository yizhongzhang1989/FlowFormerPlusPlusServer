#!/bin/bash

# FlowFormer++ Server Setup Script
# This script automatically sets up the FlowFormer++ project after cloning

set -e  # Exit on any error

echo "=== FlowFormer++ Server Setup ==="
echo "Starting automatic setup process..."

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

# Step 1: Check and download checkpoints
print_info "Step 1: Checking and downloading model checkpoints..."

CHECKPOINTS_DIR="checkpoints"
GOOGLE_DRIVE_URL="https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI"

# Required checkpoint files with their Google Drive URLs
declare -A CHECKPOINT_FILES
CHECKPOINT_FILES["chairs.pth"]="https://drive.google.com/file/d/1qpokIjlUhvex99-IfzRJ04HiNfvBZOtm/view?usp=drive_link"
CHECKPOINT_FILES["kitti.pth"]="https://drive.google.com/file/d/1i04ie6fbAii9uGFYpu99Y7rv9G7miTaH/view?usp=drive_link"
CHECKPOINT_FILES["sintel.pth"]="https://drive.google.com/file/d/1c5NYSh7Dbc94wppPltX2Q_ashAb4QxTB/view?usp=drive_link"
CHECKPOINT_FILES["things_288960.pth"]="https://drive.google.com/file/d/1lqyusjgnhtG7hc1mxbBsb8VSbuCMugyX/view?usp=drive_link"
CHECKPOINT_FILES["things.pth"]="https://drive.google.com/file/d/1YOJXzv80MjKg8GbA_lpbHD25JGlsuAqC/view?usp=drive_link"

# Function to check if gdown is installed
check_gdown() {
    if ! command -v gdown &> /dev/null; then
        print_warning "gdown is not installed. Installing gdown..."
        pip install gdown
        if [ $? -eq 0 ]; then
            print_success "gdown installed successfully"
        else
            print_error "Failed to install gdown. Please install it manually: pip install gdown"
            exit 1
        fi
    fi
}

# Function to download checkpoints
download_checkpoints() {
    print_info "Creating checkpoints directory..."
    mkdir -p "$CHECKPOINTS_DIR"
    
    print_info "Downloading checkpoints from Google Drive..."
    
    cd "$CHECKPOINTS_DIR"
    
    # Download each checkpoint file individually
    for file in "${!CHECKPOINT_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_info "Downloading $file..."
            url="${CHECKPOINT_FILES[$file]}"
            
            # Extract file ID from Google Drive URL
            file_id=$(echo "$url" | sed -n 's/.*\/d\/\([a-zA-Z0-9_-]*\)\/.*/\1/p')
            
            if [ -n "$file_id" ]; then
                # Use gdown to download the file
                if gdown "https://drive.google.com/uc?id=$file_id" -O "$file"; then
                    print_success "Successfully downloaded $file"
                else
                    print_error "Failed to download $file"
                    print_info "You can manually download from: $url"
                    cd ..
                    return 1
                fi
            else
                print_error "Could not extract file ID from URL: $url"
                print_info "You can manually download from: $url"
                cd ..
                return 1
            fi
        else
            print_info "$file already exists, skipping download"
        fi
    done
    
    cd ..
    return 0
}

# Check if checkpoints directory exists and has the required files
if [ -d "$CHECKPOINTS_DIR" ]; then
    print_info "Checkpoints directory exists. Checking for required files..."
    
    missing_files=()
    for file in "${!CHECKPOINT_FILES[@]}"; do
        if [ ! -f "$CHECKPOINTS_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All checkpoint files are present!"
    else
        print_warning "Missing checkpoint files: ${missing_files[*]}"
        print_info "Downloading missing checkpoints..."
        check_gdown
        if ! download_checkpoints; then
            print_error "Failed to download some checkpoints automatically."
            exit 1
        fi
    fi
else
    print_info "Checkpoints directory does not exist. Creating and downloading..."
    check_gdown
    if ! download_checkpoints; then
        print_error "Failed to download checkpoints automatically."
        exit 1
    fi
fi

# Verify checkpoints after download
print_info "Verifying downloaded checkpoints..."
all_present=true
for file in "${!CHECKPOINT_FILES[@]}"; do
    if [ -f "$CHECKPOINTS_DIR/$file" ]; then
        file_size=$(stat -c%s "$CHECKPOINTS_DIR/$file" 2>/dev/null || echo "0")
        if [ "$file_size" -gt 1000000 ]; then  # Check if file is larger than 1MB
            print_success "✓ $file (${file_size} bytes)"
        else
            print_warning "✗ $file appears to be incomplete (${file_size} bytes)"
            all_present=false
        fi
    else
        print_error "✗ $file is missing"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    print_success "Step 1 completed: All checkpoints are ready!"
else
    print_error "Step 1 failed: Some checkpoints are missing or incomplete."
    print_info "Please manually download the checkpoints from: $GOOGLE_DRIVE_URL"
    exit 1
fi

echo ""
print_success "=== Setup Step 1 Complete ==="
print_info "Next steps will be added to this script..."
