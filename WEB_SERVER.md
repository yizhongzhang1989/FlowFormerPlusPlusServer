# FlowFormer++ Web Server

A Flask-based web application for computing optical flow between image pairs using the FlowFormer++ model.

## Features

- **Web Interface**: Modern, responsive web UI for easy image upload
- **Real-time Processing**: Compute optical flow between any two images
- **Pre-loaded Model**: Model loads once at startup for fast processing
- **Visual Results**: Color-coded flow visualization with statistics
- **File Management**: Automatic cleanup of temporary files
- **Error Handling**: Comprehensive error handling and user feedback

## Quick Start

1. **Setup and Start Server** (all-in-one):
   ```bash
   ./setup_server.sh
   ```

   This single command will:
   - Download model checkpoints
   - Setup conda environment
   - Install all dependencies
   - **Start the web server automatically**

2. **Access the Web Interface**:
   Open your browser and go to: `http://localhost:5000`

⚠️ **Note**: The setup script starts the web server and keeps it running. You don't need any additional commands.

## Web Interface

### Upload Section
- Upload two images (PNG, JPG, JPEG, BMP, TIFF)
- Maximum file size: 16MB per image
- Live image previews
- Drag and drop support

### Results Section
- Flow visualization as color image
- Computation time statistics
- Flow statistics (range, mean, std deviation)
- Download functionality

## API Endpoints

### `GET /`
Main web interface

### `POST /upload`
Upload image pair and compute flow
- **Input**: `multipart/form-data` with `image1` and `image2` files
- **Output**: JSON with flow statistics and result filename

### `GET /result/<filename>`
Download flow visualization image

### `GET /status`
Server status and health check

### `POST /cleanup/<session_id>`
Clean up session files

## File Structure

```
├── app.py                 # Main Flask application
├── start_server.sh        # Server startup script
├── web/                   # Web interface files
│   ├── templates/
│   │   └── index.html     # Main HTML template
│   └── static/
│       ├── css/
│       │   └── style.css  # Styling
│       └── js/
│           └── app.js     # JavaScript functionality
└── tmp/                   # Temporary files
    ├── uploads/           # Uploaded images
    └── results/           # Flow visualizations
```

## Configuration

The server can be configured by modifying variables in `app.py`:

- `UPLOAD_FOLDER`: Directory for uploaded images
- `RESULT_FOLDER`: Directory for flow results
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Supported image formats

## Model Information

- **Default Model**: `checkpoints/sintel.pth`
- **Architecture**: FlowFormer++ transformer-based model
- **Performance**: Optimized for accuracy on various image types
- **Device**: Automatically uses GPU if available, fallback to CPU

## Performance Notes

- **Model Loading**: Model loads once at startup (not per request)
- **GPU Acceleration**: Automatically detected and used if available
- **Memory Management**: Efficient memory usage with automatic cleanup
- **Image Processing**: Adaptive resizing for optimal performance

## Usage Examples

### Basic Usage
1. Run the setup: `./setup_server.sh`
2. The server starts automatically and shows: "Server ready! Access the web interface at: http://localhost:5000"
3. Open browser to `http://localhost:5000`
4. Upload two images
5. Click "Compute Optical Flow"
6. View and download results

### Manual Server Management
If you need to start/stop the server manually:

```bash
# Start server (after setup is complete)
conda activate flowformerpp
python app.py

# Or run individual setup steps
./scripts/download_ckpts.sh
./scripts/setup_conda_env.sh
./scripts/setup_webserver.sh
```

## Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure checkpoints are downloaded: `./scripts/download_ckpts.sh`

2. **Environment issues**:
   - Activate conda environment: `conda activate flowformerpp`
   - Install dependencies: `pip install flask werkzeug`

3. **CUDA issues**:
   - Server automatically falls back to CPU if GPU unavailable
   - Check CUDA installation and compatibility

4. **Memory issues**:
   - Large images are automatically resized
   - Increase system memory or use smaller images

### Log Messages
- Server logs show model loading, processing times, and errors
- Check console output for debugging information

## Development

To modify the web interface:

1. **HTML**: Edit `web/templates/index.html`
2. **CSS**: Edit `web/static/css/style.css`
3. **JavaScript**: Edit `web/static/js/app.js`
4. **Backend**: Edit `app.py`

The server supports hot reloading in debug mode for development.

## Security Notes

- Files are stored temporarily and cleaned up automatically
- File type validation prevents malicious uploads
- Session-based file isolation prevents conflicts
- Regular cleanup removes old files
