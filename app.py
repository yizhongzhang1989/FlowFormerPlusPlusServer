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

# Import FlowFormer++ modules
from configs.submissions import get_cfg
from core.FlowFormer import build_flowformer
from core.utils import flow_viz, frame_utils
from core.utils.utils import InputPadder
import torch.nn.functional as F

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
    cfg = get_cfg()
    cfg.model = CHECKPOINT_PATH
    
    # Determine device based on config
    device_config = config['model']['device']
    if device_config == "auto":
        # Auto-detect device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        # Use specified device
        device = torch.device(device_config)
        print(f"Using device: {device}")
    
    # Build and load model
    model = torch.nn.DataParallel(build_flowformer(cfg))
    
    try:
        model.load_state_dict(torch.load(cfg.model, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def prepare_image(image_path, target_size=None, max_size=1024):
    """Load and prepare image for flow computation"""
    image = frame_utils.read_gen(image_path)
    image = np.array(image).astype(np.uint8)[..., :3]
    
    # Store original dimensions
    original_h, original_w = image.shape[:2]
    
    # If no target size specified, use adaptive sizing based on max_size
    if target_size is None:
        h, w = original_h, original_w
        
        # Scale down if image is too large
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            h, w = int(h * scale), int(w * scale)
        
        # Ensure dimensions are divisible by 8 (common requirement for optical flow models)
        h = (h // 8) * 8
        w = (w // 8) * 8
        
        if h != original_h or w != original_w:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        # Use specified target size
        h, w = target_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return image_tensor, (h, w), (original_h, original_w)


def compute_flow(image1_path, image2_path):
    """Compute optical flow between two images"""
    global model, device
    
    # Load and prepare images
    img1_tensor, img1_proc_size, img1_orig_size = prepare_image(image1_path, max_size=MAX_IMAGE_SIZE)
    img2_tensor, img2_proc_size, img2_orig_size = prepare_image(image2_path, max_size=MAX_IMAGE_SIZE)
    
    # Use the original size of the first image as target output size
    target_output_size = img1_orig_size
    
    # Ensure processed images have the same size for flow computation
    if img1_proc_size != img2_proc_size:
        # Resize to common size (use the smaller dimensions to preserve detail)
        common_h = min(img1_proc_size[0], img2_proc_size[0])
        common_w = min(img1_proc_size[1], img2_proc_size[1])
        
        # Ensure dimensions are divisible by 8
        common_h = (common_h // 8) * 8
        common_w = (common_w // 8) * 8
        
        img1_tensor = F.interpolate(
            img1_tensor.unsqueeze(0), 
            size=(common_h, common_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        img2_tensor = F.interpolate(
            img2_tensor.unsqueeze(0), 
            size=(common_h, common_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        proc_size = (common_h, common_w)
    else:
        proc_size = img1_proc_size
    
    # Move to device and add batch dimension
    img1_tensor = img1_tensor.unsqueeze(0).to(device)
    img2_tensor = img2_tensor.unsqueeze(0).to(device)
    
    # Compute flow
    with torch.no_grad():
        padder = InputPadder(img1_tensor.shape)
        img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)
        
        flow_pred, _ = model(img1_padded, img2_padded)
        flow_pred = padder.unpad(flow_pred)
        
        # Convert to numpy
        flow = flow_pred[0].permute(1, 2, 0).cpu().numpy()
    
    # Resize flow back to original image dimensions
    if proc_size != target_output_size:
        # Calculate scaling factors
        scale_h = target_output_size[0] / proc_size[0]
        scale_w = target_output_size[1] / proc_size[1]
        
        # Resize flow field
        flow_resized = cv2.resize(flow, (target_output_size[1], target_output_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Scale flow vectors according to the resize ratio
        flow_resized[:, :, 0] *= scale_w  # x-component
        flow_resized[:, :, 1] *= scale_h  # y-component
        
        flow = flow_resized
    
    return flow


def save_flow_visualization(flow, output_path):
    """Save flow as color visualization"""
    flow_img = flow_viz.flow_to_image(flow)
    cv2.imwrite(output_path, flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
    return output_path


def get_image_dimensions(image_path):
    """Get original image dimensions"""
    image = frame_utils.read_gen(image_path)
    image = np.array(image).astype(np.uint8)[..., :3]
    h, w = image.shape[:2]
    return h, w


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
