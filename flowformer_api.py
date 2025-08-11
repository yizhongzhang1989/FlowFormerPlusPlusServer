#!/usr/bin/env python3
"""
FlowFormer++ API Client

A Python client library for the FlowFormer++ web server API.
Supports memory-based image transfer without file I/O.

Usage:
    from flowformer_api import FlowFormerClient
    
    # Initialize client
    client = FlowFormerClient()
    
    # Compute flow from image data
    flow_image = client.compute_flow(img1_data, img2_data)
    
    # Or use with numpy arrays
    flow_image = client.compute_flow_from_arrays(img1_array, img2_array)
"""

import requests
import json
import time
import io
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


class FlowFormerClient:
    """Client for FlowFormer++ Web API with memory-based image processing"""
    
    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 120):
        """Initialize FlowFormer API client
        
        Args:
            server_url: Base URL of the FlowFormer++ server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session_id = None
        
    def setup(self) -> bool:
        """Setup and verify server connection
        
        Returns:
            bool: True if server is ready, False otherwise
        """
        try:
            status = self.get_status()
            if status.get('model_loaded'):
                print(f"✅ FlowFormer++ server ready at {self.server_url}")
                print(f"   Device: {status.get('device', 'unknown')}")
                print(f"   Memory mode: {status.get('memory_mode', 'unknown')}")
                return True
            else:
                print(f"❌ Server model not loaded: {status}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get server status
        
        Returns:
            dict: Server status information
        
        Raises:
            Exception: If server is not reachable
        """
        try:
            response = requests.get(f"{self.server_url}/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to check server status: {e}")
    
    def compute_flow(self, 
                    image1_data: bytes, 
                    image2_data: bytes,
                    auto_cleanup: bool = True) -> bytes:
        """Compute optical flow from image bytes
        
        Args:
            image1_data: First image as bytes (JPEG, PNG, etc.)
            image2_data: Second image as bytes (JPEG, PNG, etc.)
            auto_cleanup: Whether to automatically cleanup session after computation
            
        Returns:
            bytes: Flow visualization image as PNG bytes
            
        Raises:
            Exception: If computation fails
        """
        try:
            # Prepare files for upload
            files = {
                'image1': ('image1.jpg', io.BytesIO(image1_data), 'image/jpeg'),
                'image2': ('image2.jpg', io.BytesIO(image2_data), 'image/jpeg')
            }
            
            # Upload and compute
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/upload",
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get('success'):
                raise Exception(f"Computation failed: {result.get('error', 'Unknown error')}")
            
            self.session_id = result.get('session_id')
            computation_time = time.time() - start_time
            
            # Download result
            flow_image_data = self.download_result(self.session_id)
            
            # Optional cleanup
            if auto_cleanup:
                self.cleanup_session(self.session_id)
                
            print(f"✅ Flow computed in {computation_time:.2f}s")
            return flow_image_data
            
        except requests.exceptions.Timeout:
            raise Exception("Request timeout - computation took too long")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def compute_flow_from_arrays(self, 
                                image1: np.ndarray, 
                                image2: np.ndarray,
                                image_format: str = 'JPEG',
                                quality: int = 95,
                                auto_cleanup: bool = True) -> bytes:
        """Compute optical flow from numpy arrays
        
        Args:
            image1: First image as numpy array (H, W, 3) uint8
            image2: Second image as numpy array (H, W, 3) uint8
            image_format: Image format for transfer ('JPEG', 'PNG')
            quality: JPEG quality (1-100, ignored for PNG)
            auto_cleanup: Whether to automatically cleanup session after computation
            
        Returns:
            bytes: Flow visualization image as PNG bytes
            
        Raises:
            Exception: If computation fails
        """
        # Convert numpy arrays to bytes
        img1_bytes = self._array_to_bytes(image1, image_format, quality)
        img2_bytes = self._array_to_bytes(image2, image_format, quality)
        
        return self.compute_flow(img1_bytes, img2_bytes, auto_cleanup)
    
    def compute_flow_from_pil(self,
                             image1: Image.Image,
                             image2: Image.Image,
                             image_format: str = 'JPEG',
                             quality: int = 95,
                             auto_cleanup: bool = True) -> bytes:
        """Compute optical flow from PIL Images
        
        Args:
            image1: First image as PIL Image
            image2: Second image as PIL Image
            image_format: Image format for transfer ('JPEG', 'PNG')
            quality: JPEG quality (1-100, ignored for PNG)
            auto_cleanup: Whether to automatically cleanup session after computation
            
        Returns:
            bytes: Flow visualization image as PNG bytes
        """
        # Convert PIL images to bytes
        img1_bytes = self._pil_to_bytes(image1, image_format, quality)
        img2_bytes = self._pil_to_bytes(image2, image_format, quality)
        
        return self.compute_flow(img1_bytes, img2_bytes, auto_cleanup)
    
    def download_result(self, session_id: str) -> bytes:
        """Download flow visualization result
        
        Args:
            session_id: Session ID from computation
            
        Returns:
            bytes: Flow visualization image data
            
        Raises:
            Exception: If download fails
        """
        try:
            response = requests.get(
                f"{self.server_url}/result/{session_id}",
                timeout=30
            )
            response.raise_for_status()
            return response.content
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download result: {e}")
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up session from server memory
        
        Args:
            session_id: Session ID to cleanup
            
        Returns:
            bool: True if cleanup successful
        """
        try:
            response = requests.post(
                f"{self.server_url}/cleanup/{session_id}",
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result.get('success', False)
            
        except requests.exceptions.RequestException:
            return False
    
    def get_flow_as_array(self,
                         image1_data: bytes,
                         image2_data: bytes,
                         auto_cleanup: bool = True) -> np.ndarray:
        """Compute flow and return result as numpy array
        
        Args:
            image1_data: First image as bytes
            image2_data: Second image as bytes
            auto_cleanup: Whether to automatically cleanup session
            
        Returns:
            np.ndarray: Flow visualization as RGB array (H, W, 3)
        """
        flow_bytes = self.compute_flow(image1_data, image2_data, auto_cleanup)
        
        # Convert bytes to PIL Image then to numpy array
        flow_image = Image.open(io.BytesIO(flow_bytes))
        return np.array(flow_image)
    
    def get_flow_as_pil(self,
                       image1_data: bytes,
                       image2_data: bytes,
                       auto_cleanup: bool = True) -> Image.Image:
        """Compute flow and return result as PIL Image
        
        Args:
            image1_data: First image as bytes
            image2_data: Second image as bytes
            auto_cleanup: Whether to automatically cleanup session
            
        Returns:
            PIL.Image: Flow visualization as PIL Image
        """
        flow_bytes = self.compute_flow(image1_data, image2_data, auto_cleanup)
        return Image.open(io.BytesIO(flow_bytes))
    
    def _array_to_bytes(self, 
                       image_array: np.ndarray, 
                       format: str = 'JPEG', 
                       quality: int = 95) -> bytes:
        """Convert numpy array to image bytes
        
        Args:
            image_array: Image as numpy array (H, W, 3) uint8
            format: Image format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)
            
        Returns:
            bytes: Image data
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
            
        pil_image = Image.fromarray(image_array)
        return self._pil_to_bytes(pil_image, format, quality)
    
    def _pil_to_bytes(self, 
                     pil_image: Image.Image, 
                     format: str = 'JPEG', 
                     quality: int = 95) -> bytes:
        """Convert PIL Image to bytes
        
        Args:
            pil_image: PIL Image
            format: Image format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)
            
        Returns:
            bytes: Image data
        """
        buffer = io.BytesIO()
        
        if format.upper() == 'JPEG':
            # Convert RGBA to RGB for JPEG
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            pil_image.save(buffer, format='JPEG', quality=quality)
        else:
            pil_image.save(buffer, format=format)
            
        return buffer.getvalue()


# Convenience functions for quick usage
def compute_flow_quick(image1_data: bytes, 
                      image2_data: bytes, 
                      server_url: str = "http://localhost:5000") -> bytes:
    """Quick function to compute optical flow
    
    Args:
        image1_data: First image as bytes
        image2_data: Second image as bytes
        server_url: Server URL
        
    Returns:
        bytes: Flow visualization image as PNG bytes
    """
    client = FlowFormerClient(server_url)
    return client.compute_flow(image1_data, image2_data)


def compute_flow_from_files(image1_path: str,
                           image2_path: str,
                           server_url: str = "http://localhost:5000") -> bytes:
    """Compute flow from image files (convenience function)
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        server_url: Server URL
        
    Returns:
        bytes: Flow visualization image as PNG bytes
    """
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        img1_data = f1.read()
        img2_data = f2.read()
    
    return compute_flow_quick(img1_data, img2_data, server_url)


if __name__ == "__main__":
    # Example usage
    print("FlowFormer++ API Client")
    print("======================")
    
    # Test server connection
    client = FlowFormerClient()
    if client.setup():
        print("✅ Client ready for use")
        
        # Example with sample data (if available)
        try:
            result = compute_flow_from_files(
                "sample_data/img1.jpg",
                "sample_data/img2.jpg"
            )
            print(f"✅ Flow computation successful: {len(result)} bytes")
        except Exception as e:
            print(f"ℹ️  Sample test failed (expected if no sample data): {e}")
    else:
        print("❌ Client setup failed")
