import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from configs.submissions import get_cfg
from core.utils.misc import process_cfg
from core.utils import flow_viz
from core.utils import frame_utils
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer
from core.utils.utils import InputPadder

TRAIN_SIZE = [432, 960]

# Global variables for model caching (used by web server)
_cached_model = None
_cached_device = None


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size}).")
    if image_shape[0] == TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian', device='cuda'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    
    # Move to appropriate device
    if device == 'cuda' and torch.cuda.is_available():
        weights = weights.cuda()
    else:
        weights = weights.cpu()
        
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


def compute_flow_with_model(model, device, image1, image2, use_tiling=False):
    """Compute optical flow using provided model and device
    
    Args:
        model: Loaded FlowFormer model
        device: Device (cuda/cpu)
        image1: First image tensor
        image2: Second image tensor  
        use_tiling: Whether to use tiling for large images
        
    Returns:
        numpy array: Flow field
    """
    print(f"Computing optical flow on {device}...")

    image_size = image1.shape[1:]
    image1, image2 = image1[None].to(device), image2[None].to(device)

    if not use_tiling:
        # Simple padding approach for smaller images
        padder = InputPadder(image1.shape)
        image1_padded, image2_padded = padder.pad(image1, image2)

        flow_pre, _ = model(image1_padded, image2_padded)
        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
        
        # Clear intermediate tensors from GPU memory
        del image1_padded, image2_padded, flow_pre
    else:
        # Tiling approach for larger images
        hws = compute_grid_indices(image_size)
        weights = compute_weight(hws, image_size, device=device.type)
        
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
            
            # Clear intermediate tensors for each tile
            del image1_tile, image2_tile, flow_pre

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
        
        # Clear large tensors from GPU memory
        del flows, flow_count, flow_pre, weights
    
    # Clear input tensors from GPU memory
    del image1, image2

    return flow


def compute_flow_between_images(img1_path, img2_path, model_path=None, device_config="auto", 
                               max_size=None, use_tiling=False, return_original_size=False, use_cache=False):
    """High-level function to compute flow between two images with automatic backscaling
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        model_path: Path to model checkpoint
        device_config: Device configuration
        max_size: Maximum image size for processing (None for original behavior)
        use_tiling: Whether to use tiling
        return_original_size: Whether to resize flow back to original image size
        use_cache: Whether to cache the model (True for web server, False for CLI)
        
    Returns:
        numpy array: Flow field (resized to original size if return_original_size=True)
    """
    # Build model
    model, device = build_model(model_path, device_config, use_cache=use_cache)
    
    # Load and prepare images
    if max_size is not None:
        # Use new adaptive sizing for web server compatibility
        image1, image2, orig_size1, orig_size2, proc_size = load_and_prepare_images(
            img1_path, img2_path, keep_size=False, max_size=max_size)
    else:
        # Use original behavior for command line tool
        image1, image2, orig_size1, orig_size2, proc_size = load_and_prepare_images(
            img1_path, img2_path, keep_size=True)
    
    # Validate dimensions match (for web server compatibility)
    if orig_size1 != orig_size2:
        raise ValueError(f"Image dimension mismatch: Image 1 is {orig_size1[1]}×{orig_size1[0]}, "
                        f"but Image 2 is {orig_size2[1]}×{orig_size2[0]}. "
                        f"Both images must have the same dimensions for optical flow computation.")
    
    # Compute flow
    with torch.no_grad():
        flow = compute_flow_with_model(model, device, image1, image2, use_tiling)
    
    # Clean up GPU memory if not caching model
    if not use_cache and device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    
    # Resize flow back to original dimensions if requested
    if return_original_size and proc_size != orig_size1:
        target_size = orig_size1  # (height, width)
        
        print(f"Resizing flow from {proc_size} to {target_size}")
        
        # Calculate scaling factors
        scale_h = target_size[0] / proc_size[0]
        scale_w = target_size[1] / proc_size[1]
        
        # Resize flow field (cv2.resize expects (width, height))
        flow_resized = cv2.resize(flow, (target_size[1], target_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Scale flow vectors according to the resize ratio
        flow_resized[:, :, 0] *= scale_w  # x-component
        flow_resized[:, :, 1] *= scale_h  # y-component
        
        flow = flow_resized
    
    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))
    return image_size


def load_and_prepare_images(img1_path, img2_path, keep_size=True, max_size=None):
    """Load and prepare images for flow computation
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image  
        keep_size: If True, keep original size. If False, use adaptive sizing.
        max_size: Maximum size for adaptive sizing (used when keep_size=False)
        
    Returns:
        tuple: (image1_tensor, image2_tensor, original_size1, original_size2, processed_size)
    """
    print(f"Loading images: {img1_path}, {img2_path}")
    
    # Load images
    image1 = frame_utils.read_gen(img1_path)
    image2 = frame_utils.read_gen(img2_path)
    
    # Convert to numpy arrays
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    
    # Store original dimensions
    original_size1 = image1.shape[:2]  # (height, width)
    original_size2 = image2.shape[:2]
    
    print(f"Original image sizes: {original_size1}, {original_size2}")
    
    processed_size = original_size1  # Default to first image size
    
    if not keep_size:
        if max_size is None:
            # Use original behavior with TRAIN_SIZE
            dsize = compute_adaptive_image_size(image1.shape[0:2])
            print(f"Resizing to: {dsize}")
            image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            processed_size = (dsize[1], dsize[0])  # cv2 uses (width, height), we want (height, width)
        else:
            # Use max_size constraint with divisible by 8 requirement
            h, w = original_size1
            
            # Scale down if image is too large
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                h, w = int(h * scale), int(w * scale)
            
            # Ensure dimensions are divisible by 8
            h = (h // 8) * 8
            w = (w // 8) * 8
            processed_size = (h, w)
            
            if (h, w) != original_size1:
                print(f"Resizing to: {w}x{h}")
                image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_CUBIC)
                image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to torch tensors
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    
    return image1, image2, original_size1, original_size2, processed_size


def prepare_single_image(image_path, target_size=None, max_size=1024):
    """Prepare a single image for flow computation (compatible with app.py)
    
    Args:
        image_path: Path to image
        target_size: Target size tuple (height, width) or None for adaptive
        max_size: Maximum size for adaptive sizing
        
    Returns:
        tuple: (image_tensor, processed_size, original_size)
    """
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


def build_model(model_path=None, device_config="auto", use_cache=False):
    """Build and load FlowFormer++ model
    
    Args:
        model_path: Path to model checkpoint
        device_config: Device configuration ("auto", "cuda", "cpu", or torch.device)
        use_cache: Whether to cache the model for reuse (useful for web server)
        
    Returns:
        tuple: (model, device)
    """
    global _cached_model, _cached_device
    
    # Check cache first
    if use_cache and _cached_model is not None:
        print("Using cached model...")
        return _cached_model, _cached_device
    
    print("Building FlowFormer++ model...")
    cfg = get_cfg()
    
    # Use provided model path or default to sintel checkpoint
    if model_path is None:
        model_path = "checkpoints/sintel.pth"
    
    # Override the config model path
    cfg.model = model_path
    
    print(f"Loading model from: {model_path}")
    
    # Determine device
    if isinstance(device_config, torch.device):
        device = device_config
    elif device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_config)
        print(f"Using device: {device}")
    
    model = torch.nn.DataParallel(build_flowformer(cfg))
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Available models in checkpoints/:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith(".pth"):
                    print(f"  - {f}")
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load model with error handling for CUDA compatibility
    try:
        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.to(device)
    except Exception as e:
        print(f"Warning: Failed to load model on {device}: {e}")
        if device.type == 'cuda':
            print("Trying to load on CPU...")
            device = torch.device('cpu')
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.to(device)
            print("Model loaded on CPU. GPU acceleration disabled.")
        else:
            raise
    
    model.eval()
    
    # Cache the model if requested
    if use_cache:
        _cached_model = model
        _cached_device = device
        print("Model cached for future use")
    
    return model, device


def get_image_dimensions(image_path):
    """Get original image dimensions
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (height, width) of the image
    """
    image = frame_utils.read_gen(image_path)
    image = np.array(image).astype(np.uint8)[..., :3]
    h, w = image.shape[:2]
    return h, w


def save_flow_visualization(flow, output_path):
    """Save flow as color visualization (compatible with app.py)
    
    Args:
        flow: Flow field as numpy array
        output_path: Output file path
        
    Returns:
        str: Path to saved file
    """
    flow_img = flow_viz.flow_to_image(flow)
    cv2.imwrite(output_path, flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
    return output_path


def save_visualization(flow, img1_path, output_dir='tmp'):
    """Save flow visualization with automatic filename generation
    
    Args:
        flow: Flow field as numpy array
        img1_path: Path to first input image (used for filename)
        output_dir: Output directory
        
    Returns:
        str: Path to saved file
    """
    print(f"Saving flow visualization...")
    
    # Create output directory if it doesn't exist
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate output filename based on input image
    filename = osp.splitext(osp.basename(img1_path))[0]
    viz_fn = osp.join(output_dir, f"{filename}_flow.png")
    
    # Save using the shared function
    save_flow_visualization(flow, viz_fn)
    
    print(f"Flow visualization saved to: {viz_fn}")
    return viz_fn
    print(f"Saving flow visualization...")
    
    # Create output directory if it doesn't exist
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate output filename based on input image
    filename = osp.splitext(osp.basename(img1_path))[0]
    viz_fn = osp.join(output_dir, f"{filename}_flow.png")
    
    # Convert flow to color image
    flow_img = flow_viz.flow_to_image(flow)
    
    # Save as PNG (OpenCV uses BGR, so we need to convert from RGB)
    cv2.imwrite(viz_fn, flow_img[:, :, [2, 1, 0]])
    
    print(f"Flow visualization saved to: {viz_fn}")
    return viz_fn


# Memory-based functions for web server (no file I/O)
def prepare_image_from_bytes(image_bytes, max_size=1024):
    """Prepare image from bytes for flow computation (memory-only)
    
    Args:
        image_bytes: Image data as bytes
        max_size: Maximum size for adaptive sizing
        
    Returns:
        tuple: (image_tensor, processed_size, original_size)
    """
    import io
    from PIL import Image
    
    # Load image from bytes
    image_pil = Image.open(io.BytesIO(image_bytes))
    image = np.array(image_pil).astype(np.uint8)[..., :3]
    
    # Store original dimensions
    original_h, original_w = image.shape[:2]
    
    # Use adaptive sizing
    h, w = original_h, original_w
    
    # Scale down if image is too large
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        h, w = int(h * scale), int(w * scale)
    
    # Ensure dimensions are divisible by 8
    h = (h // 8) * 8
    w = (w // 8) * 8
    processed_size = (h, w)
    
    if h != original_h or w != original_w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return image_tensor, processed_size, (original_h, original_w)


def get_image_dimensions_from_bytes(image_bytes):
    """Get image dimensions from bytes
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        tuple: (height, width) of the image
    """
    import io
    from PIL import Image
    
    image_pil = Image.open(io.BytesIO(image_bytes))
    return image_pil.size[::-1]  # PIL returns (width, height), we want (height, width)


def compute_flow_from_bytes(img1_bytes, img2_bytes, model_path=None, device_config="auto", 
                           max_size=1024, use_cache=True):
    """Compute flow from image bytes (memory-only)
    
    Args:
        img1_bytes: First image as bytes
        img2_bytes: Second image as bytes
        model_path: Path to model checkpoint
        device_config: Device configuration
        max_size: Maximum image size for processing
        use_cache: Whether to cache the model
        
    Returns:
        numpy array: Flow field resized to original image size
    """
    # Build model
    model, device = build_model(model_path, device_config, use_cache=use_cache)
    
    # Prepare images from bytes
    image1, proc_size1, orig_size1 = prepare_image_from_bytes(img1_bytes, max_size)
    image2, proc_size2, orig_size2 = prepare_image_from_bytes(img2_bytes, max_size)
    
    # Ensure processed images have the same size for flow computation
    if proc_size1 != proc_size2:
        # Resize to common size (use the smaller dimensions to preserve detail)
        common_h = min(proc_size1[0], proc_size2[0])
        common_w = min(proc_size1[1], proc_size2[1])
        
        # Ensure dimensions are divisible by 8
        common_h = (common_h // 8) * 8
        common_w = (common_w // 8) * 8
        
        image1 = torch.nn.functional.interpolate(
            image1.unsqueeze(0), 
            size=(common_h, common_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        image2 = torch.nn.functional.interpolate(
            image2.unsqueeze(0), 
            size=(common_h, common_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        proc_size = (common_h, common_w)
    else:
        proc_size = proc_size1
    
    # Compute flow using the model
    with torch.no_grad():
        flow = compute_flow_with_model(model, device, image1, image2, use_tiling=False)
    
    # Clean up GPU memory if not caching model
    if not use_cache and device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Resize flow back to original image dimensions
    target_output_size = orig_size1  # Use first image's original size
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


def flow_to_image_bytes(flow, format='PNG'):
    """Convert flow to image bytes (memory-only)
    
    Args:
        flow: Flow field as numpy array
        format: Image format ('PNG', 'JPEG', etc.)
        
    Returns:
        bytes: Image data as bytes
    """
    import io
    from PIL import Image
    
    # Convert flow to color image
    flow_img = flow_viz.flow_to_image(flow)
    
    # Convert to PIL Image (flow_viz returns RGB)
    flow_pil = Image.fromarray(flow_img.astype(np.uint8))
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    flow_pil.save(img_buffer, format=format)
    img_buffer.seek(0)
    
    return img_buffer.getvalue()


def flow_to_bytes(flow, dtype=None):
    """Convert flow array to bytes for transmission
    
    Args:
        flow: Flow field as numpy array (H, W, 2)
        dtype: Target data type (default: float32)
        
    Returns:
        bytes: Serialized flow data
    """
    import io
    
    if dtype is not None:
        flow = flow.astype(dtype)
    
    buffer = io.BytesIO()
    np.save(buffer, flow)
    buffer.seek(0)
    return buffer.getvalue()


def flow_from_bytes(flow_bytes):
    """Load flow array from bytes
    
    Args:
        flow_bytes: Serialized flow data as bytes
        
    Returns:
        numpy array: Flow field (H, W, 2)
    """
    import io
    
    buffer = io.BytesIO(flow_bytes)
    buffer.seek(0)
    return np.load(buffer)


def get_flow_statistics(flow):
    """Get comprehensive flow statistics
    
    Args:
        flow: Flow field as numpy array (H, W, 2)
        
    Returns:
        dict: Flow statistics
    """
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    
    return {
        'shape': list(flow.shape),
        'dtype': str(flow.dtype),
        'flow_x': {
            'min': float(flow[:, :, 0].min()),
            'max': float(flow[:, :, 0].max()),
            'mean': float(flow[:, :, 0].mean()),
            'std': float(flow[:, :, 0].std())
        },
        'flow_y': {
            'min': float(flow[:, :, 1].min()),
            'max': float(flow[:, :, 1].max()),
            'mean': float(flow[:, :, 1].mean()),
            'std': float(flow[:, :, 1].std())
        },
        'magnitude': {
            'min': float(flow_magnitude.min()),
            'max': float(flow_magnitude.max()),
            'mean': float(flow_magnitude.mean()),
            'std': float(flow_magnitude.std())
        },
        'total_pixels': int(flow.shape[0] * flow.shape[1]),
        'size_bytes': int(flow.nbytes)
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize optical flow between two images')
    parser.add_argument('--img1', type=str, default='sample_data/img1.jpg', 
                       help='Path to first image')
    parser.add_argument('--img2', type=str, default='sample_data/img2.jpg', 
                       help='Path to second image')
    parser.add_argument('--viz_root_dir', type=str, default='tmp', 
                       help='Root directory for visualization output')
    parser.add_argument('--model', type=str, default='checkpoints/sintel.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--keep_size', action='store_true', 
                       help='Keep original image size (default: adaptive resize)')
    parser.add_argument('--use_tiling', action='store_true', 
                       help='Use tiling for large images (slower but more accurate)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size for adaptive processing')
    
    args = parser.parse_args()
    
    # Check if input images exist
    if not os.path.exists(args.img1):
        print(f"Error: Image {args.img1} not found!")
        return
    if not os.path.exists(args.img2):
        print(f"Error: Image {args.img2} not found!")
        return
    
    print(f"Processing {args.img1}, {args.img2}...")
    
    # Use the high-level function for flow computation
    try:
        if args.keep_size:
            # Original behavior - keep image size
            flow = compute_flow_between_images(
                args.img1, args.img2, 
                model_path=args.model,
                max_size=None,
                use_tiling=args.use_tiling
            )
        else:
            # Use adaptive sizing with reasonable default max_size for CLI
            max_size = args.max_size if args.max_size else 1024  # Default to 1024 for CLI
            flow = compute_flow_between_images(
                args.img1, args.img2,
                model_path=args.model, 
                max_size=max_size,
                use_tiling=args.use_tiling
            )
        
        # Save visualization
        output_file = save_visualization(flow, args.img1, args.viz_root_dir)
        
        print(f"Flow computation complete!")
        print(f"Flow shape: {flow.shape}")
        print(f"Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
        
    except Exception as e:
        print(f"Error during flow computation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
