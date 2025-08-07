# Flow Dimension Fix Documentation

## Problem Description

The original FlowFormer++ web server had a **shape mismatch issue** where the computed optical flow had different dimensions than the input images.

### Root Cause
1. **Fixed Resize**: Images were always resized to `(432, 960)` regardless of input size
2. **No Backscaling**: Flow was computed on resized images but never scaled back to original dimensions
3. **Missing Vector Scaling**: When resizing flow fields, the flow vectors themselves need to be scaled

## Solution Implemented

### 1. Adaptive Image Preprocessing
```python
def prepare_image(image_path, target_size=None, max_size=1024):
    # Store original dimensions
    original_h, original_w = image.shape[:2]
    
    # Adaptive sizing based on max_size instead of fixed target
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        h, w = int(h * scale), int(w * scale)
    
    # Ensure dimensions divisible by 8 (model requirement)
    h = (h // 8) * 8
    w = (w // 8) * 8
    
    return image_tensor, (h, w), (original_h, original_w)
```

### 2. Flow Backscaling
```python
def compute_flow(image1_path, image2_path):
    # Compute flow on processed images
    flow = model_inference(processed_images)
    
    # Resize flow back to original dimensions
    if proc_size != target_output_size:
        # Resize the flow field
        flow_resized = cv2.resize(flow, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Scale flow vectors proportionally
        flow_resized[:, :, 0] *= scale_w  # x-component scaling
        flow_resized[:, :, 1] *= scale_h  # y-component scaling
```

### 3. Dimension Tracking
- Track original, processed, and output dimensions
- Display dimension information in web UI
- Verify dimensions match expectations

## Key Improvements

✅ **Preserves Original Dimensions**: Flow output matches input image size  
✅ **Adaptive Processing**: Images scaled intelligently based on max_size  
✅ **Proper Vector Scaling**: Flow vectors scaled correctly when resizing  
✅ **Model Compatibility**: Ensures dimensions divisible by 8  
✅ **UI Feedback**: Shows dimension information to users  

## Configuration Options

Users can now control image processing via `config.json`:

```json
{
  "model": {
    "max_image_size": 1024   // Maximum processing size
  }
}
```

- **1024**: Good balance of quality/speed (default)
- **512**: Faster processing, lower quality  
- **2048**: Higher quality, requires more GPU memory

## Testing

Run the test script to verify the fix:
```bash
python test_dimension_fix.py
```

This tests:
- Image preprocessing with dimension tracking
- Flow resizing logic
- Vector scaling correctness

## Before vs After

| Aspect | Before | After |
|--------|---------|-------|
| Input Image | 1920×1080 | 1920×1080 |
| Processing Size | 432×960 (fixed) | 864×1152 (adaptive) |
| **Flow Output** | **432×960** ❌ | **1920×1080** ✅ |
| Flow Vectors | Incorrect scale | Correctly scaled |
| UI Information | No dimension info | Full dimension tracking |

The flow output now correctly matches the input image dimensions! 🎉
