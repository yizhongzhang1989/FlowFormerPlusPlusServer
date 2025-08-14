#!/usr/bin/env python3
"""
FlowFormer++ Remote API Example

This script demonstrates how to use the FlowFormer++ API client
to connect to a remote server and compute optical flow.

Server: msraig-ubuntu-3:5000
Usage: python flow_api_example.py
"""

import os
import time
import numpy as np
from PIL import Image
from flowformer_api import FlowFormerClient


def main():
    print("FlowFormer++ Remote API Example")
    print("===============================")
    print("Server: msraig-ubuntu-3:5000")
    print()
    
    # Initialize client with remote server
    server_url = "http://msraig-ubuntu-3:5000"
    client = FlowFormerClient(server_url=server_url, timeout=180)  # Longer timeout for remote
    
    # Test server connection
    print("🔗 Connecting to remote server...")
    if not client.setup():
        print("❌ Failed to connect to remote server")
        print("   Please ensure:")
        print("   1. Server is running on msraig-ubuntu-3:5000")
        print("   2. Network connectivity is available")
        print("   3. No firewall blocking the connection")
        return False
    
    print("✅ Successfully connected to remote server!")
    print()
    
    # Check for sample images
    img1_path = "sample_data/img1.jpg"
    img2_path = "sample_data/img2.jpg"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Sample images not found locally")
        print(f"   Looking for: {img1_path}, {img2_path}")
        print("   Please ensure sample images are available in sample_data/ directory")
        return False
    
    try:
        # Load images as numpy arrays
        print("📷 Loading local images...")
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))
        print(f"   Image 1 shape: {img1.shape}")
        print(f"   Image 2 shape: {img2.shape}")
        print(f"   Image 1 size: {os.path.getsize(img1_path)} bytes")
        print(f"   Image 2 size: {os.path.getsize(img2_path)} bytes")
        print()
        
        # Compute optical flow on remote server
        print("🚀 Computing optical flow on remote server...")
        print("   (This may take longer due to network transfer)")
        start_time = time.time()
        
        flow = client.compute_flow(img1, img2)
        
        total_time = time.time() - start_time
        print(f"✅ Flow computed successfully in {total_time:.2f}s")
        print(f"   Flow shape: {flow.shape}")
        print(f"   Flow data type: {flow.dtype}")
        print(f"   Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
        print()
        
        # Create flow visualization
        print("🎨 Creating flow visualization...")
        flow_vis = client.visualize_flow(flow)
        print(f"   Visualization shape: {flow_vis.shape}")
        print(f"   Visualization data type: {flow_vis.dtype}")
        print()
        
        # Create output directory and save results
        os.makedirs("tmp", exist_ok=True)
        
        # Save visualization
        vis_output_path = "tmp/remote_flow_output.png"
        Image.fromarray(flow_vis).save(vis_output_path)
        print(f"✅ Flow visualization saved to: {vis_output_path}")
        
        # Save raw flow data
        flow_data_path = "tmp/remote_flow_data.npy"
        np.save(flow_data_path, flow)
        print(f"✅ Raw flow data saved to: {flow_data_path}")
        
        # Display statistics
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        print()
        print("📊 Flow Statistics:")
        print(f"   Mean magnitude: {flow_magnitude.mean():.2f} pixels")
        print(f"   Max magnitude: {flow_magnitude.max():.2f} pixels")
        print(f"   Std magnitude: {flow_magnitude.std():.2f} pixels")
        
        print()
        print("🎉 Remote API example completed successfully!")
        print("   Results saved in tmp/ directory")
        return True
        
    except Exception as e:
        print(f"❌ Error during remote computation: {e}")
        print("   This could be due to:")
        print("   - Network connectivity issues")
        print("   - Server overload or timeout")
        print("   - Image format compatibility")
        return False


def test_connection_only():
    """Quick connection test without computation"""
    print("Quick Connection Test")
    print("====================")
    
    server_url = "http://msraig-ubuntu-3:5000"
    client = FlowFormerClient(server_url=server_url, timeout=30)
    
    if client.setup():
        print("✅ Connection successful!")
        return True
    else:
        print("❌ Connection failed!")
        return False


if __name__ == "__main__":
    # You can uncomment this line to do a quick connection test first
    # test_connection_only()
    
    # Run the full example
    success = main()
    
    if success:
        print("\n💡 Usage Summary:")
        print("   1. client = FlowFormerClient('http://msraig-ubuntu-3:5000')")
        print("   2. flow = client.compute_flow(img1, img2)")
        print("   3. vis = client.visualize_flow(flow)")
    else:
        print("\n🔧 Troubleshooting:")
        print("   - Check server availability: curl http://msraig-ubuntu-3:5000/status")
        print("   - Verify network connectivity: ping msraig-ubuntu-3")
        print("   - Ensure sample images exist in sample_data/")
