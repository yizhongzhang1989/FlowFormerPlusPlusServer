#!/usr/bin/env python3
"""
Simple FlowFormer++ API Usage Example

This script demonstrates how to use the FlowFormer++ API programmatically
using the provided FlowFormer API client.
"""

import os
from flowformer_api import FlowFormerClient


def compute_flow_via_api(server_url, image1_path, image2_path, output_path=None):
    """
    Compute optical flow using FlowFormer++ API client
    
    Args:
        server_url: URL of the FlowFormer++ server (e.g., "http://localhost:5000")
        image1_path: Path to the first image
        image2_path: Path to the second image
        output_path: Optional path to save the result image
        
    Returns:
        dict: Flow information and statistics
    """
    
    # 1. Initialize client and check server
    print("🔧 Initializing FlowFormer++ client...")
    client = FlowFormerClient(server_url)
    
    if not client.setup():
        raise Exception("Server not reachable or model not loaded")
    
    print(f"✅ Connected to FlowFormer++ server at {server_url}")
    
    # 2. Read image files
    print("� Reading image files...")
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        image1_data = f1.read()
        image2_data = f2.read()
    
    print(f"   Image 1: {len(image1_data)} bytes")
    print(f"   Image 2: {len(image2_data)} bytes")
    
    # 3. Compute optical flow (without auto-cleanup to get statistics)
    print("🔄 Computing optical flow...")
    flow_image_data = client.compute_flow(image1_data, image2_data, auto_cleanup=False)
    
    # 4. Get flow statistics
    if client.session_id:
        print("📊 Getting flow statistics...")
        flow_info = client.get_flow_info(client.session_id)
        
        stats = flow_info.get('flow_stats', {})
        print(f"   Flow shape: {stats.get('shape', 'N/A')}")
        if 'magnitude' in stats:
            mag_stats = stats['magnitude']
            print(f"   Flow magnitude range: [{mag_stats.get('min', 0):.2f}, {mag_stats.get('max', 0):.2f}]")
            print(f"   Average magnitude: {mag_stats.get('mean', 0):.2f} pixels")
        
        # Clean up session
        client.cleanup_session(client.session_id)
        print("🧹 Session cleaned up")
    
    # 5. Save result if output path provided
    if output_path:
        print("💾 Saving result...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(flow_image_data)
        print(f"   Saved to: {output_path}")
    
    print(f"✅ Downloaded {len(flow_image_data)} bytes")
    
    return flow_info if client.session_id else {'success': True}


# Example usage
if __name__ == '__main__':
    server_url = "http://localhost:5000"
    image1 = "sample_data/img1.jpg"
    image2 = "sample_data/img2.jpg"
    output = "tmp/example_flow_result.png"
    
    try:
        result = compute_flow_via_api(server_url, image1, image2, output)
        print("\n🎉 Flow computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
