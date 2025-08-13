#!/usr/bin/env python3
"""
FlowFormer++ API Client

A simplified Python client library for the FlowFormer++ web server API.

Usage:
    from flowformer_api import FlowFormerClient
    
    # Initialize client
    client = FlowFormerClient()
    
    # Compute raw flow from numpy arrays
    raw_flow = client.compute_flow(img1_array, img2_array)
    
    # Visualize flow as image
    flow_image = client.visualize_flow(raw_flow)
"""

import requests
import time
import io
import numpy as np
from PIL import Image
from core.utils import flow_viz


class FlowFormerClient:
    """Simplified client for FlowFormer++ Web API"""
    
    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 120):
        """Initialize FlowFormer API client
        
        Args:
            server_url: Base URL of the FlowFormer++ server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
    def setup(self) -> bool:
        """Setup and verify server connection
        
        Returns:
            bool: True if server is ready, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/status", timeout=10)
            response.raise_for_status()
            status = response.json()
            
            if status.get('model_loaded'):
                print(f"✅ FlowFormer++ server ready at {self.server_url}")
                print(f"   Device: {status.get('device', 'unknown')}")
                return True
            else:
                print(f"❌ Server model not loaded: {status}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False
    
    def compute_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Compute raw optical flow from numpy arrays
        
        Args:
            image1: First image as numpy array (H, W, 3) uint8
            image2: Second image as numpy array (H, W, 3) uint8
            
        Returns:
            np.ndarray: Raw flow field (H, W, 2) float32
            
        Raises:
            Exception: If computation fails
        """
        try:
            # Convert numpy arrays to bytes
            img1_bytes = self._array_to_bytes(image1)
            img2_bytes = self._array_to_bytes(image2)
            
            # Prepare files for upload
            files = {
                'image1': ('image1.jpg', io.BytesIO(img1_bytes), 'image/jpeg'),
                'image2': ('image2.jpg', io.BytesIO(img2_bytes), 'image/jpeg')
            }
            
            # Upload and compute
            response = requests.post(
                f"{self.server_url}/upload",
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get('success'):
                raise Exception(f"Computation failed: {result.get('error', 'Unknown error')}")
            
            session_id = result.get('session_id')
            
            # Download raw flow data
            flow_response = requests.get(
                f"{self.server_url}/flow/{session_id}",
                timeout=30
            )
            flow_response.raise_for_status()
            
            # Parse flow data from bytes
            buffer = io.BytesIO(flow_response.content)
            buffer.seek(0)
            flow_data = np.load(buffer)
            
            # Cleanup session
            try:
                requests.post(f"{self.server_url}/cleanup/{session_id}", timeout=10)
            except:
                pass  # Ignore cleanup errors
            
            return flow_data
            
        except requests.exceptions.Timeout:
            raise Exception("Request timeout - computation took too long")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Visualize optical flow as color image
        
        Args:
            flow: Raw flow field (H, W, 2) as numpy array
            
        Returns:
            np.ndarray: Flow visualization as RGB image (H, W, 3) uint8
        """
        # Use flow_viz to convert flow to color image
        flow_rgb = flow_viz.flow_to_image(flow)
        return flow_rgb
    
    def _array_to_bytes(self, image_array: np.ndarray, quality: int = 95) -> bytes:
        """Convert numpy array to JPEG bytes
        
        Args:
            image_array: Image as numpy array (H, W, 3) uint8
            quality: JPEG quality (1-100)
            
        Returns:
            bytes: JPEG image data
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
            
        pil_image = Image.fromarray(image_array)
        
        # Convert RGBA to RGB for JPEG
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
            
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        return buffer.getvalue()


if __name__ == "__main__":
    # Working example of FlowFormer++ API Client
    print("FlowFormer++ API Client Example")
    print("===============================")
    
    # Initialize client
    client = FlowFormerClient()
    
    # Test server connection
    if not client.setup():
        print("❌ Server not available. Please start the FlowFormer++ server first.")
        print("   Run: python app.py")
        exit(1)
    
    # Check for sample images
    import os
    img1_path = "sample_data/img1.jpg"
    img2_path = "sample_data/img2.jpg"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Sample images not found. Please ensure sample_data/img1.jpg and img2.jpg exist.")
        exit(1)
    
    try:
        # Load images as numpy arrays
        print("\n📷 Loading sample images...")
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))
        print(f"   Image 1 shape: {img1.shape}")
        print(f"   Image 2 shape: {img2.shape}")
        
        # Compute raw optical flow
        print("\n🔄 Computing optical flow...")
        start_time = time.time()
        flow = client.compute_flow(img1, img2)
        computation_time = time.time() - start_time
        print(f"✅ Flow computed in {computation_time:.2f}s")
        print(f"   Flow shape: {flow.shape}")
        print(f"   Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
        
        # Visualize flow
        print("\n🎨 Creating flow visualization...")
        flow_vis = client.visualize_flow(flow)
        print(f"   Visualization shape: {flow_vis.shape}")
        
        # Create tmp directory if it doesn't exist
        os.makedirs("tmp", exist_ok=True)
        
        # Save results to tmp directory
        output_path = "tmp/flow_output.png"
        Image.fromarray(flow_vis).save(output_path)
        print(f"✅ Flow visualization saved to: {output_path}")
        
        # Save raw flow data (optional)
        flow_data_path = "tmp/flow_data.npy"
        np.save(flow_data_path, flow)
        print(f"✅ Raw flow data saved to: {flow_data_path}")
        
        print("\n🎉 Example completed successfully!")
        print("\nUsage summary:")
        print("1. flow = client.compute_flow(img1, img2)  # Get raw flow")
        print("2. vis = client.visualize_flow(flow)       # Visualize flow")
        
    except Exception as e:
        print(f"❌ Error during computation: {e}")
        exit(1)
