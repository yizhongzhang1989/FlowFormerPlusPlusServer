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


def compute_flow(model, image1, image2, use_tiling=False):
    print(f"Computing optical flow...")

    image_size = image1.shape[1:]
    
    # Check if CUDA is available and model is on CUDA
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    image1, image2 = image1[None].to(device), image2[None].to(device)

    if not use_tiling:
        # Simple padding approach for smaller images
        padder = InputPadder(image1.shape)
        image1_padded, image2_padded = padder.pad(image1, image2)

        flow_pre, _ = model(image1_padded, image2_padded)
        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
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

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

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


def load_and_prepare_images(img1_path, img2_path, keep_size=True):
    print(f"Loading images: {img1_path}, {img2_path}")
    
    # Load images
    image1 = frame_utils.read_gen(img1_path)
    image2 = frame_utils.read_gen(img2_path)
    
    # Convert to numpy arrays
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    
    print(f"Original image size: {image1.shape}")
    
    # Resize if needed
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        print(f"Resizing to: {dsize}")
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    
    # Convert to torch tensors
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    
    return image1, image2


def build_model(model_path=None):
    print("Building FlowFormer++ model...")
    cfg = get_cfg()
    
    # Use provided model path or default to sintel checkpoint
    if model_path is None:
        model_path = "checkpoints/sintel.pth"
    
    # Override the config model path
    cfg.model = model_path
    
    print(f"Loading model from: {model_path}")
    
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
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.cuda()
    except Exception as e:
        print(f"Warning: Failed to load model on CUDA: {e}")
        print("Trying to load on CPU...")
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print("Model loaded on CPU. GPU acceleration disabled.")
    
    model.eval()
    return model


def save_visualization(flow, img1_path, output_dir='tmp'):
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
    
    args = parser.parse_args()
    
    # Check if input images exist
    if not os.path.exists(args.img1):
        print(f"Error: Image {args.img1} not found!")
        return
    if not os.path.exists(args.img2):
        print(f"Error: Image {args.img2} not found!")
        return
    
    print(f"Processing {args.img1}, {args.img2}...")
    
    # Build model
    model = build_model(args.model)
    
    # Load and prepare images
    image1, image2 = load_and_prepare_images(args.img1, args.img2, args.keep_size)
    
    # Compute optical flow
    with torch.no_grad():
        flow = compute_flow(model, image1, image2, args.use_tiling)
    
    # Save visualization
    output_file = save_visualization(flow, args.img1, args.viz_root_dir)
    
    print(f"Flow computation complete!")
    print(f"Flow shape: {flow.shape}")
    print(f"Flow range: [{flow.min():.2f}, {flow.max():.2f}]")


if __name__ == '__main__':
    main()
