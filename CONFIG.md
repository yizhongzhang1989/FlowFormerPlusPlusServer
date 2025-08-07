# Configuration Guide

The FlowFormer++ server uses a `config.json` file to manage server settings. This file is automatically created during the initial setup, but you can modify it to customize the server behavior.

## Configuration File Location

The configuration file is located at the root of the project:
```
config.json
```

## Configuration Options

### Server Settings
```json
"server": {
  "host": "0.0.0.0",        // Server host (0.0.0.0 for all interfaces)
  "port": 5000,             // Server port number
  "debug": false            // Enable Flask debug mode
}
```

**Common port configurations:**
- `5000` - Default Flask port
- `8080` - Alternative web port
- `3000` - Common development port

### Model Settings
```json
"model": {
  "checkpoint_path": "checkpoints/sintel.pth",  // Path to model checkpoint
  "device": "auto",                             // Device: "auto", "cuda", "cpu"
  "max_image_size": 1024                        // Maximum image size for processing
}
```

**Device options:**
- `"auto"` - Automatically detect best device (GPU if available, otherwise CPU)
- `"cuda"` - Force GPU usage (requires CUDA-compatible GPU)
- `"cpu"` - Force CPU usage

### Upload Settings
```json
"upload": {
  "max_file_size_mb": 10,                                    // Maximum file size in MB
  "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"], // Allowed image formats
  "upload_folder": "tmp/uploads",                            // Temporary upload directory
  "results_folder": "tmp/results"                            // Results storage directory
}
```

### Processing Settings
```json
"processing": {
  "auto_cleanup": true,           // Automatically clean up old files
  "cleanup_interval_hours": 24,   // How often to clean up (in hours)
  "max_stored_results": 100       // Maximum number of results to keep
}
```

## Example Configurations

### Development Setup (Local Access Only)
```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 3000,
    "debug": true
  }
}
```

### Production Setup (Network Access)
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false
  },
  "upload": {
    "max_file_size_mb": 20
  }
}
```

### CPU-Only Setup
```json
{
  "model": {
    "device": "cpu",
    "max_image_size": 512
  }
}
```

## Applying Configuration Changes

After modifying `config.json`:

1. **Stop the server** (if running) with `Ctrl+C`
2. **Restart the server**:
   ```bash
   ./restart_server.sh
   ```

The new configuration will be loaded automatically.

## Troubleshooting

### Port Already in Use
If you get a "port already in use" error:
1. Change the port number in `config.json`
2. Restart the server
3. Use the new URL (e.g., `http://localhost:8080`)

### GPU Not Detected
If the server doesn't use your GPU:
1. Set `"device": "cuda"` in the model configuration
2. Ensure CUDA and PyTorch are properly installed
3. Check that your GPU is CUDA-compatible

### File Size Limits
To allow larger images:
1. Increase `max_file_size_mb` in upload settings
2. Consider reducing `max_image_size` for performance
3. Restart the server to apply changes

## Security Considerations

- **Host Setting**: Use `"127.0.0.1"` for local access only, `"0.0.0.0"` for network access
- **Debug Mode**: Never enable debug mode in production (`"debug": false`)
- **File Uploads**: Consider the security implications of allowing file uploads
- **Network Access**: If using `"0.0.0.0"`, ensure proper firewall configuration
