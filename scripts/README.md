# FlowFormer++ Setup Scripts

This directory contains modular setup scripts for the FlowFormer++ project.

## Scripts Overview

### Core Scripts
- `common.sh` - Common utility functions and color definitions used by all scripts
- `download_ckpts.sh` - Downloads model checkpoints from Google Drive

### Usage

Run the main setup script from the project root:
```bash
./setup_server.sh
```

Or run individual scripts:
```bash
./scripts/download_ckpts.sh
```

## Adding New Steps

To add a new setup step:

1. Create a new script in this directory (e.g., `install_deps.sh`)
2. Make it executable: `chmod +x install_deps.sh`
3. Source the common functions: `source "$(dirname "$0")/common.sh"`
4. Add the step to the `SETUP_STEPS` array in `setup_server.sh`

### Script Template

```bash
#!/bin/bash

# Description of what this script does

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/common.sh"

print_info "=== Step Description ==="

# Your setup logic here

print_success "Step completed successfully!"
```

## Current Steps

1. **Download Checkpoints** (`download_ckpts.sh`)
   - Downloads all required model checkpoint files from Google Drive
   - Verifies file integrity and completeness
   - Skips already downloaded files

## Planned Steps

Future setup steps can include:
- Dependency installation
- Environment setup
- CUDA extension compilation
- Dataset preparation
- Configuration validation
