# FlowFormer++ Server Setup

This repository includes an automated setup script that prepares the FlowFormer++ environment after cloning.

## Quick Start

Simply run the setup script:

```bash
./setup_server.sh
```

## What the Setup Script Does

### Step 1: Download Model Checkpoints
- Downloads 5 pre-trained model checkpoints from Google Drive:
  - `chairs.pth` (65 MB)
  - `kitti.pth` (65 MB) 
  - `sintel.pth` (65 MB)
  - `things.pth` (65 MB)
  - `things_288960.pth` (65 MB)
- Verifies file integrity
- Skips download if files already exist

### Step 2: Setup Conda Environment
- Creates `flowformerpp` conda environment (if it doesn't exist)
- Installs PyTorch 2.4.1 with CUDA 11.8 support
- Installs required dependencies:
  - **Conda packages**: torchvision, torchaudio, matplotlib, tensorboard, scipy, opencv
  - **Pip packages**: yacs, loguru, einops, timm==0.4.12, imageio
- Verifies all packages are properly installed

## Manual Setup

If you prefer to set up manually:

### 1. Download Checkpoints
```bash
./scripts/download_ckpts.sh
```

### 2. Setup Environment
```bash
./scripts/setup_conda_env.sh
```

## Environment Usage

After setup, activate the environment:
```bash
conda activate flowformerpp
```

Then you can run the FlowFormer++ scripts:
```bash
# Visualize flow on Sintel dataset
python visualize_flow.py --eval_type sintel --keep_size

# Visualize flow on image sequences
python visualize_flow.py --eval_type seq

# Evaluate on validation sets
python evaluate_FlowFormer_tile.py --eval sintel_validation
```

## Requirements

- **Conda/Anaconda**: Required for environment management
- **CUDA**: GPU with CUDA support (tested with CUDA 11.8)
- **Python**: 3.8+ (automatically installed in conda environment)
- **Internet**: For downloading checkpoints and packages

## Troubleshooting

- **Conda not found**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- **CUDA issues**: Ensure your GPU drivers support CUDA 11.8
- **Download fails**: Manually download checkpoints from the [Google Drive folder](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI)

## File Structure

```
├── setup_server.sh          # Main setup script
├── requirements.txt         # Python package requirements  
├── checkpoints/            # Downloaded model checkpoints
│   ├── chairs.pth
│   ├── kitti.pth
│   ├── sintel.pth
│   ├── things.pth
│   └── things_288960.pth
└── scripts/                # Individual setup scripts
    ├── common.sh           # Shared utilities
    ├── download_ckpts.sh   # Checkpoint downloader
    └── setup_conda_env.sh  # Environment setup
```
