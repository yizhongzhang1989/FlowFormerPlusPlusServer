#!/usr/bin/env python3
"""
Example usage of FlowFormer++ API Client

This script demonstrates different ways to use the FlowFormer API client
for memory-based optical flow computation.
"""

import numpy as np
from PIL import Image
import io
import time
from flowformer_api import FlowFormerClient, compute_flow_quick


def example_basic_usage():
    """Example 1: Basic usage with image files"""
    print("Example 1: Basic usage with image files")
    print("-" * 40)
    
    try:
        # Load images as bytes
        with open('sample_data/img1.jpg', 'rb') as f:
            img1_bytes = f.read()
        with open('sample_data/img2.jpg', 'rb') as f:
            img2_bytes = f.read()
        
        # Quick computation
        flow_result = compute_flow_quick(img1_bytes, img2_bytes)
        
        # Save result
        with open('tmp/example_basic_flow.png', 'wb') as f:
            f.write(flow_result)
        
        print(f"✅ Flow computed: {len(flow_result)} bytes")
        print(f"   Result saved to: tmp/example_basic_flow.png")
        
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")


def example_client_class():
    """Example 2: Using the client class with setup"""
    print("\nExample 2: Using client class with setup")
    print("-" * 40)
    
    try:
        # Initialize and setup client
        client = FlowFormerClient()
        if not client.setup():
            print("❌ Server not ready")
            return
        
        # Load images
        with open('sample_data/img1.jpg', 'rb') as f:
            img1_bytes = f.read()
        with open('sample_data/img2.jpg', 'rb') as f:
            img2_bytes = f.read()
        
        # Compute flow with detailed control
        start_time = time.time()
        flow_result = client.compute_flow(
            img1_bytes, 
            img2_bytes, 
            auto_cleanup=False  # Manual cleanup for demonstration
        )
        computation_time = time.time() - start_time
        
        print(f"✅ Flow computed in {computation_time:.2f}s")
        print(f"   Session ID: {client.session_id}")
        
        # Manual cleanup
        if client.cleanup_session(client.session_id):
            print("✅ Session cleaned up")
        
        # Save result
        with open('tmp/example_client_flow.png', 'wb') as f:
            f.write(flow_result)
        print(f"   Result saved to: tmp/example_client_flow.png")
        
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")


def example_numpy_arrays():
    """Example 3: Using numpy arrays"""
    print("\nExample 3: Using numpy arrays")
    print("-" * 40)
    
    try:
        client = FlowFormerClient()
        if not client.setup():
            print("❌ Server not ready")
            return
        
        # Load images as numpy arrays
        img1 = np.array(Image.open('sample_data/img1.jpg'))
        img2 = np.array(Image.open('sample_data/img2.jpg'))
        
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # Compute flow from arrays
        flow_result = client.compute_flow_from_arrays(
            img1, 
            img2,
            image_format='JPEG',
            quality=90
        )
        
        # Get result as numpy array
        flow_array = client.get_flow_as_array(
            client._array_to_bytes(img1),
            client._array_to_bytes(img2)
        )
        
        print(f"✅ Flow result shape: {flow_array.shape}")
        print(f"   Flow data type: {flow_array.dtype}")
        
        # Save results
        with open('tmp/example_numpy_flow.png', 'wb') as f:
            f.write(flow_result)
        
        Image.fromarray(flow_array).save('tmp/example_numpy_array.png')
        print(f"   Results saved to tmp/example_numpy_*.png")
        
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")


def example_pil_images():
    """Example 4: Using PIL Images"""
    print("\nExample 4: Using PIL Images")
    print("-" * 40)
    
    try:
        client = FlowFormerClient()
        if not client.setup():
            print("❌ Server not ready")
            return
        
        # Load as PIL Images
        img1_pil = Image.open('sample_data/img1.jpg')
        img2_pil = Image.open('sample_data/img2.jpg')
        
        print(f"Image 1 size: {img1_pil.size}")
        print(f"Image 2 size: {img2_pil.size}")
        
        # Compute flow from PIL images
        flow_result = client.compute_flow_from_pil(
            img1_pil,
            img2_pil,
            image_format='PNG'  # Use PNG for lossless transfer
        )
        
        # Get result as PIL Image
        with open('sample_data/img1.jpg', 'rb') as f1, open('sample_data/img2.jpg', 'rb') as f2:
            flow_pil = client.get_flow_as_pil(f1.read(), f2.read())
        
        print(f"✅ Flow PIL image size: {flow_pil.size}")
        print(f"   Flow PIL mode: {flow_pil.mode}")
        
        # Save results
        with open('tmp/example_pil_flow.png', 'wb') as f:
            f.write(flow_result)
        
        flow_pil.save('tmp/example_pil_image.png')
        print(f"   Results saved to tmp/example_pil_*.png")
        
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")


def example_memory_efficiency():
    """Example 5: Memory efficiency demonstration"""
    print("\nExample 5: Memory efficiency with compressed transfer")
    print("-" * 50)
    
    try:
        client = FlowFormerClient()
        if not client.setup():
            print("❌ Server not ready")
            return
        
        # Load original images
        img1_pil = Image.open('sample_data/img1.jpg')
        img2_pil = Image.open('sample_data/img2.jpg')
        
        # Test different compression levels
        qualities = [50, 75, 95]
        
        for quality in qualities:
            print(f"\nTesting JPEG quality: {quality}")
            
            # Convert to bytes with different qualities
            img1_bytes = client._pil_to_bytes(img1_pil, 'JPEG', quality)
            img2_bytes = client._pil_to_bytes(img2_pil, 'JPEG', quality)
            
            print(f"  Image 1 size: {len(img1_bytes):,} bytes")
            print(f"  Image 2 size: {len(img2_bytes):,} bytes")
            
            # Compute flow
            start_time = time.time()
            flow_result = client.compute_flow(img1_bytes, img2_bytes)
            computation_time = time.time() - start_time
            
            print(f"  ✅ Computation time: {computation_time:.2f}s")
            print(f"  Result size: {len(flow_result):,} bytes")
            
            # Save result
            with open(f'tmp/example_quality_{quality}.png', 'wb') as f:
                f.write(flow_result)
        
        print(f"\n✅ Quality comparison results saved to tmp/example_quality_*.png")
        
    except Exception as e:
        print(f"❌ Example 5 failed: {e}")


def main():
    """Run all examples"""
    print("FlowFormer++ API Client Examples")
    print("================================")
    
    # Create output directory
    import os
    os.makedirs('tmp', exist_ok=True)
    
    # Run examples
    example_basic_usage()
    example_client_class()
    example_numpy_arrays()
    example_pil_images()
    example_memory_efficiency()
    
    print("\n🎉 All examples completed!")
    print("Check the tmp/ directory for results.")


if __name__ == "__main__":
    main()
