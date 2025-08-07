#!/bin/bash

# FlowFormer++ Web Server Restart Script
# Use this script to restart the web server after initial setup

echo "=== Restarting FlowFormer++ Web Server ==="

# Check if conda environment exists
if ! conda env list | grep -q "flowformerpp"; then
    echo "Error: flowformerpp conda environment not found"
    echo "Please run './setup_server.sh' first to complete the initial setup"
    exit 1
fi

# Check if model checkpoints exist
if [ ! -d "checkpoints" ] || [ ! -f "checkpoints/sintel.pth" ]; then
    echo "Error: Model checkpoints not found"
    echo "Please run './setup_server.sh' first to download checkpoints"
    exit 1
fi

# Check if config file exists
if [ ! -f "config.json" ]; then
    echo "Error: Configuration file not found"
    echo "Please run './setup_server.sh' first to create config.json"
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p tmp/uploads tmp/results

# Read port from config.json
port=5000
if command -v python3 &> /dev/null; then
    port=$(python3 -c "import json; print(json.load(open('config.json'))['server']['port'])" 2>/dev/null || echo "5000")
fi

echo "Starting web server..."
echo "The server will be available at: http://localhost:$port"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
conda run -n flowformerpp python app.py
