#!/usr/bin/env python3
"""
FlowFormer++ Web Server
A Flask web application for computing optical flow between image pairs.
"""

import os
import sys
import uuid
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import torch
import numpy as np
import cv2

# Add core to path
sys.path.append('core')

# Import FlowFormer++ modules from visualize_flow_img_pair to avoid duplication
import visualize_flow_img_pair as flow_utils

# Load configuration from config.json
def load_config():
    """Load configuration from config.json file."""
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback configuration if file doesn't exist
        print(f"Warning: {config_path} not found, using default configuration")
        return {
            "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
            "model": {"checkpoint_path": "checkpoints/sintel.pth", "device": "auto", "max_image_size": 1024},
            "upload": {"max_file_size_mb": 10, "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"], 
                      "upload_folder": "tmp/uploads", "results_folder": "tmp/results"},
            "processing": {"auto_cleanup": True, "cleanup_interval_hours": 24, "max_stored_results": 100}
        }

# Load configuration
config = load_config()

# Extract configuration values
UPLOAD_FOLDER = config['upload']['upload_folder']
RESULT_FOLDER = config['upload']['results_folder']
MAX_FILE_SIZE_MB = config['upload']['max_file_size_mb']
ALLOWED_EXTENSIONS = set(config['upload']['allowed_extensions'])
CHECKPOINT_PATH = config['model']['checkpoint_path']
MAX_IMAGE_SIZE = config['model']['max_image_size']
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

# Global model variable - loaded once at startup
model = None
device = None

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'flowformer-plus-plus-server-2023'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load FlowFormer++ model at startup"""
    global model, device
    
    print("Loading FlowFormer++ model...")
    
    # Determine device based on config
    device_config = config['model']['device']
    
    # Build model using the shared function
    model, device = flow_utils.build_model(
        model_path=CHECKPOINT_PATH,
        device_config=device_config,
        use_cache=True  # Cache for web server
    )
    
    print("Model loaded successfully for web server!")


def prepare_image(image_path, target_size=None, max_size=1024):
    """Load and prepare image for flow computation"""
    # Use the shared function from flow_utils
    return flow_utils.prepare_single_image(image_path)


def compute_flow(image1_path, image2_path):
    """Compute optical flow between two images"""
    global model, device
    
    # Use the shared function from flow_utils with web server configuration
    return flow_utils.compute_flow_between_images(
        image1_path, 
        image2_path, 
        model_path=CHECKPOINT_PATH,
        device_config=config['model']['device'],
        max_size=MAX_IMAGE_SIZE,
        return_original_size=True,
        use_cache=True  # Web server should cache the model
    )


def save_flow_visualization(flow, output_path):
    """Save flow as color visualization"""
    return flow_utils.save_flow_visualization(flow, output_path)


def get_image_dimensions(image_path):
    """Get original image dimensions"""
    return flow_utils.get_image_dimensions(image_path)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and flow computation"""
    try:
        # Check if files are in request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both image1 and image2 files are required'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # Check if files are selected
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        # Check file types
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded files
        filename1 = f"{session_id}_img1.{file1.filename.rsplit('.', 1)[1].lower()}"
        filename2 = f"{session_id}_img2.{file2.filename.rsplit('.', 1)[1].lower()}"
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        # Get original image dimensions
        img1_h, img1_w = get_image_dimensions(filepath1)
        img2_h, img2_w = get_image_dimensions(filepath2)
        
        # Validate that input images have matching dimensions
        if img1_h != img2_h or img1_w != img2_w:
            error_msg = f"Image dimension mismatch: Image 1 is {img1_w}×{img1_h}, but Image 2 is {img2_w}×{img2_h}. Both images must have the same dimensions for optical flow computation."
            return jsonify({
                'error': error_msg,
                'image1_dimensions': [img1_h, img1_w],
                'image2_dimensions': [img2_h, img2_w],
                'suggestion': 'Please resize both images to the same dimensions before uploading.'
            }), 400
        
        # Compute optical flow
        start_time = time.time()
        flow = compute_flow(filepath1, filepath2)
        computation_time = time.time() - start_time
        
        # Save flow visualization
        result_filename = f"{session_id}_flow.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        save_flow_visualization(flow, result_path)
        
        # Get flow statistics
        flow_stats = {
            'shape': flow.shape,
            'min': float(flow.min()),
            'max': float(flow.max()),
            'mean': float(flow.mean()),
            'std': float(flow.std())
        }
        
        # Image dimension information
        image_info = {
            'image1_dimensions': [img1_h, img1_w],
            'image2_dimensions': [img2_h, img2_w], 
            'flow_dimensions': list(flow.shape[:2]),
            'dimensions_match': (flow.shape[0] == img1_h and flow.shape[1] == img1_w)
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'result_filename': result_filename,
            'computation_time': round(computation_time, 2),
            'flow_stats': flow_stats,
            'image_info': image_info
        })
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


@app.route('/result/<filename>')
def get_result(filename):
    """Serve result images"""
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if os.path.exists(result_path):
            return send_file(result_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Result not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload/<filename>')
def get_upload(filename):
    """Serve uploaded images"""
    try:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(upload_path):
            return send_file(upload_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Server status endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'result_folder': app.config['RESULT_FOLDER']
    })


@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up session files"""
    try:
        # Remove uploaded files
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
            for file in os.listdir(folder):
                if file.startswith(session_id):
                    os.remove(os.path.join(folder, file))
        
        return jsonify({'success': True, 'message': 'Session cleaned up'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def cleanup_old_files(max_age_hours=24):
    """Clean up old files periodically"""
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        print(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        print(f"Error cleaning up {filename}: {e}")


if __name__ == '__main__':
    print("Starting FlowFormer++ Web Server...")
    
    # Load model at startup
    load_model()
    
    # Clean up old files
    cleanup_old_files()
    
    # Get server configuration
    server_config = config['server']
    host = server_config['host']
    port = server_config['port']
    debug = server_config['debug']
    
    print("Server ready!")
    print(f"Access the web interface at: http://localhost:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)
