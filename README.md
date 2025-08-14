# FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation

> **🚀 Server Fork**: This is a forked version of the original FlowFormer++ repository. The main enhancement is the addition of a **web server and API** that hosts the pre-trained model, enabling online optical flow computation through a modern web interface or programmatic API access.

> FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation   
> Xiaoyu Shi<sup>\*</sup>, [Zhaoyang Huang](https://drinkingcoder.github.io)<sup>\*</sup>, [Dasong Li](https://dasongli1.github.io/), [Manyuan Zhang](https://manyuan97.github.io/), Ka Chun Cheung, Simon See, [Hongwei Qin](http://qinhongwei.com/academic/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> CVPR 2023  

<img src="assets/demo.png">

## Quick Start

### 🚀 One-Command Setup
```bash
./setup_server.sh
```

This single command will:
- Download pre-trained checkpoints (~325MB)
- Create conda environment with all dependencies
- **Start web server automatically at http://localhost:5000**

⚠️ **Note**: The setup script starts the web server and keeps it running. Open your browser to http://localhost:5000 to use the interface.

### 🔄 Restart Server (After Initial Setup)
```bash
./restart_server.sh
```

## Table of Contents
1. [Web Server Interface](#web-server-interface)
2. [API Access](#api-access)
3. [Training and Evaluation](#training-and-evaluation)
4. [Configuration](#configuration)
5. [Command Line Tools](#command-line-tools)
6. [Data Preparation](#data-preparation)
7. [Troubleshooting](#troubleshooting)
8. [Citation](#citation)

## Web Server Interface

### Features
- **Modern Web UI**: Drag-and-drop image upload with live previews
- **Real-time Processing**: Compute optical flow between any two images
- **Visual Results**: Color-coded flow visualization with detailed statistics
- **Dimension Validation**: Automatic checking that images have matching dimensions
- **Memory Efficient**: Memory-only processing without temporary files
- **Configurable**: Fully customizable via `config.json`

### Usage
1. **Upload Images**: Drag and drop or click to upload two images
2. **Dimension Check**: Automatic validation that images have same dimensions
3. **Compute Flow**: Click "Compute Optical Flow" button  
4. **View Results**: Flow visualization with statistics and download option

### Supported Formats
- **Image Types**: PNG, JPG, JPEG, BMP, TIFF
- **File Size**: Up to 10MB per image (configurable)
- **Dimensions**: Both images must have identical dimensions

## API Access

### HTTP API Endpoints

#### Check Server Status
```bash
curl http://localhost:5000/status
```

#### Compute Flow (Upload Images)
```bash
curl -X POST \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg" \
  http://localhost:5000/upload
```

#### Download Flow Visualization
```bash
curl http://localhost:5000/result/SESSION_ID -o flow_result.png
```

#### Download Raw Flow Data
```bash
curl http://localhost:5000/flow/SESSION_ID -o flow_data.npy
```

### Python API Client

#### Simple Usage
```python
from flowformer_api import FlowFormerClient
import numpy as np
from PIL import Image

# Initialize client
client = FlowFormerClient()
if client.setup():
    # Load images as numpy arrays
    img1 = np.array(Image.open('img1.jpg'))
    img2 = np.array(Image.open('img2.jpg'))
    
    # Compute raw optical flow
    flow = client.compute_flow(img1, img2)  # Returns (H, W, 2) numpy array
    print(f"Flow shape: {flow.shape}")
    
    # Create flow visualization
    flow_vis = client.visualize_flow(flow)  # Returns (H, W, 3) RGB image
    Image.fromarray(flow_vis).save('flow_visualization.png')
```

#### Complete Example
```python
from flowformer_api import FlowFormerClient
import numpy as np
from PIL import Image

# Initialize and test connection
client = FlowFormerClient()
if not client.setup():
    print("Server not available")
    exit(1)

# Load sample images
img1 = np.array(Image.open('sample_data/img1.jpg'))
img2 = np.array(Image.open('sample_data/img2.jpg'))

# Compute optical flow (returns raw flow field)
flow = client.compute_flow(img1, img2)

# Visualize flow (returns RGB visualization)
flow_visualization = client.visualize_flow(flow)

# Save results
Image.fromarray(flow_visualization).save('tmp/flow_output.png')
np.save('tmp/flow_data.npy', flow)
```

#### API Methods
- **`compute_flow(img1, img2)`**: Takes two numpy arrays (H,W,3), returns raw flow (H,W,2)
- **`visualize_flow(flow)`**: Takes flow array (H,W,2), returns RGB visualization (H,W,3)
- **`setup()`**: Test server connection and readiness
```

#### Run Example
```bash
# Run the working example (requires server running)
python flowformer_api.py
```

This example will:
- Test server connection
- Load sample images from `sample_data/`
- Compute optical flow
- Create visualization
- Save results to `tmp/` directory

### API Response Format
```json
{
  "success": true,
  "session_id": "a19f69d9-b1f8-4ada-a90e-aa5a5af20497",
  "computation_time": 0.91,
  "flow_stats": {
    "shape": [1276, 1702, 2],
    "magnitude": {
      "min": 0.0,
      "max": 334.12,
      "mean": 15.67
    }
  },
  "endpoints": {
    "visualization": "/result/SESSION_ID",
    "raw_flow": "/flow/SESSION_ID",
    "flow_info": "/flow/SESSION_ID/info"
  }
}
```

## Training and Evaluation

### Requirements
```bash
conda create --name flowformerpp
conda activate flowformerpp
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio
```

### Pre-training
```bash
python -u pretrain_FlowFormer_maemask.py --stage youtube
```

### Training Pipeline
```bash
./run_train.sh
```

### Models
Download [pre-trained models](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI?usp=sharing):
```
├── checkpoints
    ├── chairs.pth
    ├── things.pth
    ├── sintel.pth
    ├── kitti.pth
    └── things_288960.pth
```

### Evaluation
```bash
# Sintel validation
python evaluate_FlowFormer_tile.py --eval sintel_validation

# KITTI validation  
python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth

# Generate submissions
python evaluate_FlowFormer_tile.py --eval sintel_submission
python evaluate_FlowFormer_tile.py --eval kitti_submission
```

## Configuration

### Server Configuration (`config.json`)

The server behavior can be fully customized via `config.json`:

```json
{
  "server": {
    "host": "0.0.0.0",        // Server host
    "port": 5000,             // Server port
    "debug": false            // Debug mode
  },
  "model": {
    "checkpoint_path": "checkpoints/sintel.pth",
    "device": "auto",         // "auto", "cuda", "cpu"
    "max_image_size": 1024    // Max processing size
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
```

### Common Configurations

#### Change Port
```json
{
  "server": {
    "port": 8080
  }
}
```

#### CPU-Only Mode
```json
{
  "model": {
    "device": "cpu",
    "max_image_size": 512
  }
}
```

#### High-Resolution Mode
```json
{
  "model": {
    "max_image_size": 2048
  },
  "upload": {
    "max_file_size_mb": 50
  }
}
```

### Apply Configuration Changes
```bash
# Edit config.json, then restart server
./restart_server.sh
```

## Command Line Tools

### Image Pair Processing
```bash
# Basic usage with sample images
python visualize_flow_img_pair.py

# Custom image pair
python visualize_flow_img_pair.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg

# Keep original size (no resizing)
python visualize_flow_img_pair.py --keep_size

# Use tiling for large images
python visualize_flow_img_pair.py --use_tiling
```

### Dataset Visualization
```bash
# Sintel dataset
python visualize_flow.py --eval_type sintel --keep_size

# Image sequence from video
python visualize_flow.py --eval_type seq
```

## Data Preparation

For training and evaluation, you'll need these datasets:
* [YoutubeVOS](https://youtube-vos.org/dataset/) (for pre-training)
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

### Directory Structure
```
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Troubleshooting

### Server Issues

#### Port Already in Use
```bash
# Change port in config.json
{
  "server": {
    "port": 8080
  }
}
# Then restart
./restart_server.sh
```

#### GPU Not Detected
```bash
# Force GPU usage
{
  "model": {
    "device": "cuda"
  }
}
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Model Not Found
```bash
# Re-download checkpoints
./scripts/download_ckpts.sh
```

### Common Errors

#### Dimension Mismatch
**Error**: "Image dimension mismatch: Image 1 is 1920×1080, but Image 2 is 1280×720"
**Solution**: Resize both images to the same dimensions before uploading

#### CUDA Out of Memory
**Solutions**:
- Reduce `max_image_size` in config.json
- Switch to CPU mode: `"device": "cpu"`
- Use smaller images

#### File Size Too Large
**Solution**: Increase `max_file_size_mb` in config.json

### Validation
```bash
# Test configuration syntax
python3 -c "import json; json.load(open('config.json')); print('Config OK')"

# Check current settings
python3 -c "import json; c=json.load(open('config.json')); print(f'Port: {c[\"server\"][\"port\"]}')"
```

## Development

### Project Structure
```
├── app.py                          # Main Flask application
├── flowformer_api.py              # Python API client library with example
├── visualize_flow_img_pair.py     # Core flow computation module
├── config.json                    # Server configuration
├── setup_server.sh               # One-command setup script
├── restart_server.sh              # Server restart script
├── web/                           # Web interface files
│   ├── templates/index.html       # Main HTML template
│   └── static/                    # CSS and JavaScript
├── scripts/                       # Setup scripts
│   ├── download_ckpts.sh          # Download model checkpoints
│   ├── setup_conda_env.sh         # Environment setup
│   └── setup_webserver.sh         # Web server setup
├── core/                          # FlowFormer++ model code
├── configs/                       # Training configurations
└── checkpoints/                   # Pre-trained models
```

### Key Features Implemented

#### ✅ Flow Dimension Fix
- **Problem**: Flow output had wrong dimensions compared to input images
- **Solution**: Adaptive image processing with proper backscaling and vector scaling
- **Result**: Flow output now correctly matches input image dimensions

#### ✅ Dimension Validation  
- **Frontend**: Real-time dimension checking with visual indicators
- **Backend**: Validation before processing with specific error messages
- **User Experience**: Clear feedback and guidance for dimension issues

#### ✅ Memory-Only Processing
- **No File I/O**: All processing done in memory without temporary files
- **Automatic Cleanup**: Results automatically cleaned up after use
- **Raw Flow Access**: API provides both visualization and raw flow data

#### ✅ Comprehensive API
- **HTTP Endpoints**: Full REST API for programmatic access
- **Python Client**: High-level client library with convenience functions
- **Multiple Formats**: Support for bytes, numpy arrays, and PIL images

## Security Considerations

- **Host Setting**: Use `"127.0.0.1"` for local access only, `"0.0.0.0"` for network access
- **Debug Mode**: Never enable debug mode in production
- **File Uploads**: Consider security implications of allowing file uploads
- **Network Access**: Use proper firewall configuration when exposing to network

## License
FlowFormer++ is released under the Apache License

## Citation
```bibtex
@inproceedings{shi2023flowformer++,
  title={Flowformer++: Masked cost volume autoencoding for pretraining optical flow estimation},
  author={Shi, Xiaoyu and Huang, Zhaoyang and Li, Dasong and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1599--1610},
  year={2023}
}
```

## Acknowledgement

In this project, we use parts of codes in:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
