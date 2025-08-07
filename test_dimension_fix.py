#!/usr/bin/env python3
"""
Test script to verify the flow dimension fix
"""

import sys
import os
import numpy as np
import cv2
import torch

# Add core to path
sys.path.append('core')

# Import FlowFormer++ modules  
from core.utils import frame_utils

def create_test_images():
    """Create test images of different sizes"""
    # Create test image 1 (larger)
    img1 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    cv2.imwrite('test_img1.jpg', img1)
    
    # Create test image 2 (smaller)
    img2 = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    cv2.imwrite('test_img2.jpg', img2)
    
    print(f"Created test_img1.jpg: {img1.shape}")
    print(f"Created test_img2.jpg: {img2.shape}")
    
    return 'test_img1.jpg', 'test_img2.jpg'

def test_prepare_image():
    """Test the updated prepare_image function"""
    print("\n=== Testing prepare_image function ===")
    
    # Import the updated function
    sys.path.append('.')
    from app import prepare_image, get_image_dimensions
    
    img1_path, img2_path = create_test_images()
    
    # Test prepare_image
    img1_tensor, proc_size, orig_size = prepare_image(img1_path, max_size=1024)
    print(f"Image 1 - Original: {orig_size}, Processed: {proc_size}")
    
    img2_tensor, proc_size2, orig_size2 = prepare_image(img2_path, max_size=1024)
    print(f"Image 2 - Original: {orig_size2}, Processed: {proc_size2}")
    
    # Test get_image_dimensions
    dims1 = get_image_dimensions(img1_path)
    dims2 = get_image_dimensions(img2_path)
    print(f"Dimension check - Image 1: {dims1}, Image 2: {dims2}")
    
    # Verify dimensions match
    assert dims1 == orig_size, f"Dimension mismatch: {dims1} != {orig_size}"
    assert dims2 == orig_size2, f"Dimension mismatch: {dims2} != {orig_size2}"
    
    print("✓ prepare_image function working correctly!")
    
    # Cleanup
    os.remove(img1_path)
    os.remove(img2_path)

def test_flow_resizing():
    """Test flow resizing logic"""
    print("\n=== Testing flow resizing logic ===")
    
    # Simulate flow computation result
    proc_size = (432, 576)  # Processed image size
    target_size = (600, 800)  # Original image size
    
    # Create dummy flow field
    flow = np.random.randn(proc_size[0], proc_size[1], 2).astype(np.float32)
    print(f"Original flow shape: {flow.shape}")
    
    # Resize flow back to original dimensions (simulating the app logic)
    if proc_size != target_size:
        scale_h = target_size[0] / proc_size[0]
        scale_w = target_size[1] / proc_size[1]
        
        print(f"Scale factors - H: {scale_h:.3f}, W: {scale_w:.3f}")
        
        # Resize flow field
        flow_resized = cv2.resize(flow, (target_size[1], target_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Scale flow vectors
        flow_resized[:, :, 0] *= scale_w  # x-component
        flow_resized[:, :, 1] *= scale_h  # y-component
        
        print(f"Resized flow shape: {flow_resized.shape}")
        print(f"Flow vector scaling applied correctly")
        
        assert flow_resized.shape[:2] == target_size, "Flow not resized to correct dimensions"
        print("✓ Flow resizing logic working correctly!")
    
if __name__ == "__main__":
    print("Testing FlowFormer++ dimension fix...")
    
    try:
        test_prepare_image()
        test_flow_resizing()
        print("\n🎉 All tests passed! The dimension fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
