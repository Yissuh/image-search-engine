import requests
import time
import io
import numpy as np
from PIL import Image
import cv2

class RotatedImageSearchFrameClient:
    def __init__(self, api_url="http://localhost:8000"):
        """
        Initialize the client for rotated image search that works with image frames
        
        Parameters:
        - api_url: Base URL of the image search API
        """
        self.api_url = api_url
    
    def health_check(self):
        """Check the health status of the API"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "code": response.status_code, "message": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search_frame(self, frame, top_k=1, format="JPEG", quality=95):
        """
        Search for similar rotated images using a frame/numpy array
        
        Parameters:
        - frame: Numpy array representing the image (BGR or RGB)
        - top_k: Number of results to return
        - format: Image format for sending to API (JPEG, PNG)
        - quality: JPEG quality (1-100) if format is JPEG
        
        Returns:
        - Dict containing search results
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid numpy array")
        
        # Convert BGR to RGB if needed (assuming OpenCV format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # If OpenCV BGR format, convert to RGB for PIL
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Create PIL image from numpy array
            pil_image = Image.fromarray(frame)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=quality)
            buffer.seek(0)
            
            # Make the API request
            print(f"Sending frame to {self.api_url}/search...")
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/search",
                params={"top_k": top_k},
                files={"file": ("frame.jpg", buffer, f"image/{format.lower()}")}
            )
            total_time = time.time() - start_time
        
            # Check if the request was successful
            if response.status_code == 200:
                results = response.json()
                results["total_time"] = total_time
                return results
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        else:
            raise ValueError("Frame must be a 3-channel color image (BGR or RGB)")
    
    def extract_product_info(self, product_str):
        """
        Extract product name and barcode from product string
        
        Parameters:
        - product_str: The product string from API response
        
        Returns:
        - Tuple of (product_name, barcode)
        """
        # Split by underscore to separate product name and barcode
        parts = product_str.split('_', 1)
        
        if len(parts) == 2:
            product_name = parts[0]
            barcode = parts[1]
        else:
            # If no underscore is found
            product_name = product_str
            barcode = "Unknown"
            
        return product_name, barcode


def search_product_from_frame(frame=None, resize_dim=(256, 256), api_url="http://localhost:8000", top_k=1):
    """
    Function to search for a product using an image frame
    
    Parameters:
    - frame: Numpy array of the image 
    - resize_dim: Dimensions to resize the image to (width, height)
    - api_url: URL of the search API
    - top_k: Number of top results to return
    
    Returns:
    - Product information if found, None otherwise
    """
    # Initialize the client
    search_client = RotatedImageSearchFrameClient(api_url=api_url)
    
    # Check API health
    health = search_client.health_check()
    print(f"API Health: {health}")

    
    if frame is None:
        raise ValueError("No frame found")
    
    # Resize the frame if dimensions are provided
    if resize_dim:
        frame = cv2.resize(frame, resize_dim)
    
    # Search for similar images
    try:
        results = search_client.search_frame(frame, top_k=top_k)
        
        if results and results.get("results") and len(results["results"]) > 0:
            # Print results
            print(f"\nFound {len(results['results'])} similar images in {results['query_time']:.3f}s:")
            
            product_matches = []
            for i, item in enumerate(results["results"]):
                # Extract product name and barcode
                product_name, barcode = search_client.extract_product_info(item["product"])
                
                # Create a match object
                match = {
                    "product_name": product_name,
                    "barcode": barcode,
                    "rotation": item['rotation'],
                    "similarity": item['similarity'],
                    "path": item['path']
                }
                
                product_matches.append(match)
                
                # Print information
                print(f"{i+1}. Product Name: {product_name}")
                print(f"   Barcode: {barcode}")
                print(f"   Rotation: {item['rotation']}")
                print(f"   Similarity: {item['similarity']:.4f}")
                print(f"   Path: {item['path']}")
            
            return product_name, barcode
        else:
            print("No matching products found")
            return None
            
    except Exception as e:
        print(f"Error searching image: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Call the function with an image path
    frame = cv2.imread("images/ramen_rotated.jpg")
    if frame is None:
        print("Error loading image")
    
    product_name, barcode = search_product_from_frame(frame=frame, resize_dim=(256, 256), api_url="http://localhost:8000", top_k=1)
    
    if [product_name, barcode]:
        print(f"\nProduct: {product_name}")
        print(f"Barcode: {barcode}")