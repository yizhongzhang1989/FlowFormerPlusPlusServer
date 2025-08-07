# Configuration Examples

This file contains example configurations for different use cases.

## Example 1: Change Server Port

To run the server on port 8080 instead of 5000:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false
  }
}
```

Then restart the server:
```bash
./restart_server.sh
```

Access at: `http://localhost:8080`

## Example 2: Development Mode

For development with debug enabled and local access only:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 3000,
    "debug": true
  }
}
```

## Example 3: Production Setup

For production with increased file limits and optimized settings:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 80,
    "debug": false
  },
  "upload": {
    "max_file_size_mb": 25,
    "upload_folder": "tmp/uploads",
    "results_folder": "tmp/results"
  },
  "processing": {
    "auto_cleanup": true,
    "cleanup_interval_hours": 12,
    "max_stored_results": 50
  }
}
```

## Example 4: CPU-Only Mode

For systems without GPU or when you want to force CPU usage:

```json
{
  "model": {
    "checkpoint_path": "checkpoints/sintel.pth",
    "device": "cpu",
    "max_image_size": 512
  }
}
```

## Example 5: High-Resolution Mode

For processing larger images (requires more GPU memory):

```json
{
  "model": {
    "checkpoint_path": "checkpoints/sintel.pth",
    "device": "cuda",
    "max_image_size": 2048
  },
  "upload": {
    "max_file_size_mb": 50
  }
}
```

## Quick Commands

```bash
# Test configuration syntax
python3 -c "import json; json.load(open('config.json')); print('Config OK')"

# View current port
python3 -c "import json; print('Port:', json.load(open('config.json'))['server']['port'])"

# Restart server after config changes
./restart_server.sh
```
