#!/usr/bin/env python3
"""
Simple FlowFormer++ API Usage Example

This script demonstrates how to use the FlowFormer++ API programmatically.
"""

import requests
import os


def compute_flow_via_api(server_url, image1_path, image2_path, output_path=None):
    """
    Compute optical flow using FlowFormer++ API
    
    Args:
        server_url: URL of the FlowFormer++ server (e.g., "http://localhost:5000")
        image1_path: Path to the first image
        image2_path: Path to the second image
        output_path: Optional path to save the result image
        
    Returns:
        dict: API response with flow statistics
    """
    
    # 1. Check if server is running
    try:
        status_response = requests.get(f"{server_url}/status", timeout=5)
        status_response.raise_for_status()
        status = status_response.json()
        
        if not status.get('model_loaded'):
            raise Exception("FlowFormer++ model not loaded on server")
            
        print(f"✅ Server is running on {status.get('device', 'unknown')} device")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Server not reachable: {e}")
    
    # 2. Upload images and compute flow
    print("🔄 Computing optical flow...")
    
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {
            'image1': ('image1.jpg', f1, 'image/jpeg'),
            'image2': ('image2.jpg', f2, 'image/jpeg')
        }
        
        upload_response = requests.post(f"{server_url}/upload", files=files, timeout=120)
        upload_response.raise_for_status()
        result = upload_response.json()
    
    if not result.get('success'):
        raise Exception(f"Flow computation failed: {result.get('error', 'Unknown error')}")
    
    session_id = result['session_id']
    print(f"✅ Flow computed successfully (Session: {session_id})")
    print(f"   Computation time: {result['computation_time']}s")
    print(f"   Flow shape: {result['flow_stats']['shape']}")
    print(f"   Flow range: [{result['flow_stats']['min']:.2f}, {result['flow_stats']['max']:.2f}]")
    
    # 3. Download result image
    print("📥 Downloading result...")
    
    result_response = requests.get(f"{server_url}/result/{session_id}", timeout=30)
    result_response.raise_for_status()
    
    image_data = result_response.content
    print(f"✅ Downloaded {len(image_data)} bytes")
    
    # 4. Save result if output path provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"💾 Saved result to: {output_path}")
    
    # 5. Clean up (optional)
    cleanup_response = requests.post(f"{server_url}/cleanup/{session_id}", timeout=10)
    if cleanup_response.status_code == 200:
        print("🧹 Session cleaned up")
    
    return result


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
