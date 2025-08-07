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

# Create necessary directories if they don't exist
mkdir -p tmp/uploads tmp/results

echo "Starting web server..."
echo "The server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
conda run -n flowformerpp python app.py
