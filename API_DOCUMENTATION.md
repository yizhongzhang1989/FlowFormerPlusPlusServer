# FlowFormer++ API Documentation

This document describes how to use the FlowFormer++ Web Server API to compute optical flow programmatically.

## API Endpoints

### 1. Server Status
**GET** `/status`

Check if the server is running and the model is loaded.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "device": "cuda",
  "memory_mode": true,
  "cached_results": 0
}
```

### 2. Compute Optical Flow
**POST** `/upload`

Upload two images and compute optical flow between them.

**Request:**
- Content-Type: `multipart/form-data`
- Form fields:
  - `image1`: First image file
  - `image2`: Second image file

**Response:**
```json
{
  "success": true,
  "session_id": "a19f69d9-b1f8-4ada-a90e-aa5a5af20497",
  "computation_time": 0.91,
  "flow_stats": {
    "shape": [1276, 1702, 2],
    "min": -289.95,
    "max": 209.87,
    "mean": 2.34,
    "std": 15.67
  },
  "image_info": {
    "image1_dimensions": [1276, 1702],
    "image2_dimensions": [1276, 1702],
    "flow_dimensions": [1276, 1702],
    "dimensions_match": true
  },
  "note": "Result stored in memory - use /result/<session_id> to download"
}
```

### 3. Download Result
**GET** `/result/<session_id>`

Download the flow visualization image.

**Response:**
- Content-Type: `image/png`
- Body: PNG image data

### 4. Cleanup Session
**POST** `/cleanup/<session_id>`

Remove the result from server memory (optional - results expire automatically).

**Response:**
```json
{
  "success": true,
  "message": "Session cleaned up from memory"
}
```

## Usage Examples

### Python with requests library

```python
import requests

# 1. Check server status
response = requests.get('http://localhost:5000/status')
status = response.json()
print(f"Server status: {status['status']}")

# 2. Upload images and compute flow
with open('image1.jpg', 'rb') as f1, open('image2.jpg', 'rb') as f2:
    files = {
        'image1': ('img1.jpg', f1, 'image/jpeg'),
        'image2': ('img2.jpg', f2, 'image/jpeg')
    }
    response = requests.post('http://localhost:5000/upload', files=files)
    result = response.json()

session_id = result['session_id']
print(f"Flow computed in {result['computation_time']}s")

# 3. Download result
response = requests.get(f'http://localhost:5000/result/{session_id}')
with open('flow_result.png', 'wb') as f:
    f.write(response.content)

# 4. Cleanup (optional)
requests.post(f'http://localhost:5000/cleanup/{session_id}')
```

### curl commands

```bash
# Check status
curl http://localhost:5000/status

# Upload images and compute flow
curl -X POST \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg" \
  http://localhost:5000/upload

# Download result (replace SESSION_ID with actual ID)
curl http://localhost:5000/result/SESSION_ID -o flow_result.png

# Cleanup
curl -X POST http://localhost:5000/cleanup/SESSION_ID
```

## Testing Tools

### 1. Comprehensive API Test Script
```bash
# Test with sample data (saves to tmp/api_test_results/)
python test_api.py --test-sample-data

# Test with custom images (saves to tmp/api_test_results/)
python test_api.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg

# Test with custom output directory
python test_api.py --test-sample-data --output-dir tmp/my_results

# Check server status only
python test_api.py --status-only
```

### 2. Simple API Example
```bash
# Run the simple example (saves to tmp/example_flow_result.png)
python api_example.py
```

### 3. Managing Test Results
```bash
# List tmp directory contents
ls -la tmp/

# Clean up all test results
rm -rf tmp/api_test_results tmp/example_flow_result.png

# Clean up entire tmp directory (will be recreated as needed)
rm -rf tmp/
```

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Image dimension mismatch: Image 1 is 1920×1080, but Image 2 is 1280×720. Both images must have the same dimensions for optical flow computation.",
  "image1_dimensions": [1080, 1920],
  "image2_dimensions": [720, 1280],
  "suggestion": "Please resize both images to the same dimensions before uploading."
}
```

**404 Not Found:**
```json
{
  "error": "Result not found or expired"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Error processing images: CUDA out of memory"
}
```

### Best Practices

1. **Check server status** before making requests
2. **Handle timeouts** - flow computation can take 1-10 seconds depending on image size
3. **Validate image dimensions** - both images must have the same size
4. **Use appropriate file formats** - PNG, JPG, JPEG, BMP, TIFF are supported
5. **Clean up sessions** when done to free server memory
6. **Handle errors gracefully** - check response status codes

### Performance Notes

- **Memory-only processing**: No files are saved to disk, everything is processed in memory
- **Model caching**: The FlowFormer++ model stays loaded for fast subsequent requests
- **GPU acceleration**: Uses CUDA when available, falls back to CPU
- **Automatic sizing**: Large images are automatically resized to maintain reasonable memory usage
- **Session expiration**: Results are automatically cleaned up after 1 hour

## Server Configuration

The server behavior can be configured via `config.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  },
  "model": {
    "checkpoint_path": "checkpoints/sintel.pth",
    "device": "auto",
    "max_image_size": 1024
  },
  "upload": {
    "max_file_size_mb": 10,
    "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"]
  }
}
```
