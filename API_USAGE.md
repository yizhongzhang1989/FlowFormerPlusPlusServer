# FlowFormer++ API Usage Guide

## Simple Flow Computation Example

The `api_client_examples.py` file provides a **simple, single-function example** for computing optical flow on a remote machine.

### Basic Usage

```bash
# Compute flow between two images
python api_client_examples.py

# Specify custom output path
python api_client_examples.py --output my_flow_result.png

# Use custom images
python api_client_examples.py --img1 path/to/first.jpg --img2 path/to/second.jpg --output result.png
```

### What it does

The `compute_flow_example()` function:
1. **Reads two images** from disk (default: sample_data/img1.jpg, img2.jpg)
2. **Computes raw optical flow** using FlowFormer++ via HTTP API
3. **Converts flow to color visualization** locally using flow_viz
4. **Exports the result** as a PNG image

### Key Features

- ✅ **Single function**: One simple function handles the entire workflow
- ✅ **Raw flow access**: Gets actual flow data (not just visualization)
- ✅ **Local visualization**: Creates color flow images using standard flow_viz
- ✅ **Remote computation**: Offloads heavy computation to server
- ✅ **Memory efficient**: Uses bytes-based processing (no temporary files)

### For Remote Machine Usage

1. Start the FlowFormer++ server on your GPU machine:
   ```bash
   python app.py
   ```

2. Run the example from any client machine:
   ```bash
   python api_client_examples.py --output my_flow.png
   ```

The example is designed to be **simple and easy to understand** - perfect for integrating optical flow computation into your own projects.
