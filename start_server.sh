#!/bin/bash

# FlowFormer++ Web Server Startup Script

echo "=== Starting FlowFormer++ Web Server ==="

# Check if conda environment exists
if ! conda env list | grep -q "flowformerpp"; then
    echo "Error: flowformerpp conda environment not found"
    echo "Please run setup_server.sh first to create the environment"
    exit 1
fi

# Check if model checkpoints exist
if [ ! -d "checkpoints" ] || [ ! -f "checkpoints/sintel.pth" ]; then
    echo "Error: Model checkpoints not found"
    echo "Please run setup_server.sh first to download checkpoints"
    exit 1
fi

# Create necessary directories
mkdir -p tmp/uploads tmp/results

# Install Flask if not already installed
echo "Checking Flask installation..."
conda run -n flowformerpp pip install flask werkzeug

echo "Starting web server..."
echo "The server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
conda run -n flowformerpp python app.py
