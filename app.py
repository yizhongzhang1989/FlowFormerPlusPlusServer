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
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'flowformer-plus-plus-server-2023'

# Note: No need to create directories in memory-only mode


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


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and flow computation (memory-only)"""
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
        
        # Read files into memory
        img1_bytes = file1.read()
        img2_bytes = file2.read()
        
        # Get original image dimensions from bytes
        img1_h, img1_w = flow_utils.get_image_dimensions_from_bytes(img1_bytes)
        img2_h, img2_w = flow_utils.get_image_dimensions_from_bytes(img2_bytes)
        
        # Validate that input images have matching dimensions
        if img1_h != img2_h or img1_w != img2_w:
            error_msg = f"Image dimension mismatch: Image 1 is {img1_w}×{img1_h}, but Image 2 is {img2_w}×{img2_h}. Both images must have the same dimensions for optical flow computation."
            return jsonify({
                'error': error_msg,
                'image1_dimensions': [img1_h, img1_w],
                'image2_dimensions': [img2_h, img2_w],
                'suggestion': 'Please resize both images to the same dimensions before uploading.'
            }), 400
        
        # Compute optical flow from bytes (no file I/O)
        start_time = time.time()
        flow = flow_utils.compute_flow_from_bytes(
            img1_bytes,
            img2_bytes,
            model_path=CHECKPOINT_PATH,
            device_config=config['model']['device'],
            max_size=MAX_IMAGE_SIZE,
            use_cache=True
        )
        computation_time = time.time() - start_time
        
        # Convert flow to image bytes (no file I/O)
        result_bytes = flow_utils.flow_to_image_bytes(flow, format='PNG')
        
        # Generate unique session ID for caching the result in memory
        session_id = str(uuid.uuid4())
        
        # Store result in memory cache (you could use Redis or similar for production)
        if not hasattr(app, 'result_cache'):
            app.result_cache = {}
        
        # Store both raw flow data and visualization
        app.result_cache[session_id] = {
            'flow_data': flow,  # Raw flow array
            'result_bytes': result_bytes,  # Visualization image
            'timestamp': time.time(),
            'flow_stats': flow_utils.get_flow_statistics(flow),
            'image_info': {
                'image1_dimensions': [img1_h, img1_w],
                'image2_dimensions': [img2_h, img2_w], 
                'flow_dimensions': list(flow.shape[:2]),
                'dimensions_match': (flow.shape[0] == img1_h and flow.shape[1] == img1_w)
            }
        }
        
        # Clean up old cache entries (keep only recent results)
        current_time = time.time()
        max_age = 3600  # 1 hour
        expired_keys = [k for k, v in app.result_cache.items() 
                       if current_time - v['timestamp'] > max_age]
        for key in expired_keys:
            del app.result_cache[key]
        
        # Get flow statistics and image info from cached data
        cached_result = app.result_cache[session_id]
        flow_stats = cached_result['flow_stats']
        image_info = cached_result['image_info']
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'computation_time': round(computation_time, 2),
            'flow_stats': flow_stats,
            'image_info': image_info,
            'endpoints': {
                'visualization': f'/result/{session_id}',
                'raw_flow': f'/flow/{session_id}',
                'flow_info': f'/flow/{session_id}/info'
            },
            'note': 'Result stored in memory - use endpoints to access data'
        })
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


@app.route('/result/<session_id>')
def get_result(session_id):
    """Serve result images from memory cache"""
    try:
        # Check if we have the result in memory cache
        if not hasattr(app, 'result_cache') or session_id not in app.result_cache:
            return jsonify({'error': 'Result not found or expired'}), 404
        
        # Get result from cache
        cached_result = app.result_cache[session_id]
        result_bytes = cached_result['result_bytes']
        
        # Create response with image data
        from flask import Response
        return Response(
            result_bytes,
            mimetype='image/png',
            headers={
                'Content-Disposition': f'inline; filename=flow_result_{session_id}.png',
                'Cache-Control': 'no-cache'
            }
        )
        
    except Exception as e:
        print(f"Error serving result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/flow/<session_id>')
def get_raw_flow(session_id):
    """Serve raw flow data as numpy array bytes"""
    try:
        # Check if we have the result in memory cache
        if not hasattr(app, 'result_cache') or session_id not in app.result_cache:
            return jsonify({'error': 'Flow data not found or expired'}), 404
        
        # Get raw flow from cache
        cached_result = app.result_cache[session_id]
        flow_data = cached_result['flow_data']
        
        # Convert flow to bytes
        flow_bytes = flow_utils.flow_to_bytes(flow_data, dtype=np.float32)
        
        # Create response with binary data
        from flask import Response
        return Response(
            flow_bytes,
            mimetype='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename=flow_data_{session_id}.npy',
                'Content-Type': 'application/octet-stream',
                'X-Flow-Shape': f"{flow_data.shape[0]},{flow_data.shape[1]},{flow_data.shape[2]}",
                'X-Flow-Dtype': str(flow_data.dtype),
                'Cache-Control': 'no-cache'
            }
        )
        
    except Exception as e:
        print(f"Error serving raw flow: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/flow/<session_id>/info')
def get_flow_info(session_id):
    """Get detailed flow information and statistics"""
    try:
        # Check if we have the result in memory cache
        if not hasattr(app, 'result_cache') or session_id not in app.result_cache:
            return jsonify({'error': 'Flow data not found or expired'}), 404
        
        # Get cached data
        cached_result = app.result_cache[session_id]
        
        return jsonify({
            'session_id': session_id,
            'timestamp': cached_result['timestamp'],
            'flow_stats': cached_result['flow_stats'],
            'image_info': cached_result['image_info'],
            'endpoints': {
                'visualization': f'/result/{session_id}',
                'raw_flow': f'/flow/{session_id}',
                'flow_info': f'/flow/{session_id}/info'
            }
        })
        
    except Exception as e:
        print(f"Error getting flow info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Server status endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown',
        'memory_mode': True,
        'cached_results': len(getattr(app, 'result_cache', {}))
    })


@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up session from memory cache"""
    try:
        if hasattr(app, 'result_cache') and session_id in app.result_cache:
            del app.result_cache[session_id]
            return jsonify({'success': True, 'message': 'Session cleaned up from memory'})
        else:
            return jsonify({'success': False, 'message': 'Session not found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def cleanup_old_results(max_age_hours=1):
    """Clean up old results from memory cache"""
    if not hasattr(app, 'result_cache'):
        return
    
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    
    expired_keys = [k for k, v in app.result_cache.items() 
                   if current_time - v['timestamp'] > max_age_seconds]
    
    for key in expired_keys:
        del app.result_cache[key]
        print(f"Cleaned up expired result: {key}")
    
    if expired_keys:
        print(f"Cleaned up {len(expired_keys)} expired results from memory")


if __name__ == '__main__':
    print("Starting FlowFormer++ Web Server (Memory Mode)...")
    
    # Load model at startup
    load_model()
    
    # Initialize result cache
    app.result_cache = {}
    
    # Clean up old results from memory
    cleanup_old_results()
    
    # Get server configuration
    server_config = config['server']
    host = server_config['host']
    port = server_config['port']
    debug = server_config['debug']
    
    print("Server ready! (Memory-only mode - no files saved to disk)")
    print(f"Access the web interface at: http://localhost:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)
