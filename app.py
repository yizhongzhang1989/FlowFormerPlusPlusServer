#!/usr/bin/env python3
"""
FlowFormer++ Web Server
A Flask web application for computing optical flow between image pairs.
"""

import os
import sys
import uuid
import time
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

# Configuration
UPLOAD_FOLDER = 'tmp/uploads'
RESULT_FOLDER = 'tmp/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

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
    cfg.model = "checkpoints/sintel.pth"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
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


def prepare_image(image_path, target_size=(432, 960)):
    """Load and prepare image for flow computation"""
    image = frame_utils.read_gen(image_path)
    image = np.array(image).astype(np.uint8)[..., :3]
    
    # Adaptive resize to target size
    h, w = image.shape[:2]
    scale_h = target_size[0] / h
    scale_w = target_size[1] / w
    scale = max(scale_h, scale_w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return image_tensor, (new_h, new_w)


def compute_flow(image1_path, image2_path):
    """Compute optical flow between two images"""
    global model, device
    
    # Load and prepare images
    img1_tensor, img1_size = prepare_image(image1_path)
    img2_tensor, img2_size = prepare_image(image2_path)
    
    # Ensure images have the same size
    if img1_size != img2_size:
        # Resize to common size
        common_h = min(img1_size[0], img2_size[0])
        common_w = min(img1_size[1], img2_size[1])
        
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
    
    return flow


def save_flow_visualization(flow, output_path):
    """Save flow as color visualization"""
    flow_img = flow_viz.flow_to_image(flow)
    cv2.imwrite(output_path, flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
    return output_path


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
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'result_filename': result_filename,
            'computation_time': round(computation_time, 2),
            'flow_stats': flow_stats
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
    
    print("Server ready!")
    print("Access the web interface at: http://localhost:5000")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
