import requests
import time
import io
import numpy as np
from PIL import Image
import cv2
from collections import Counter


class ImageSearchFrameClient:
    def __init__(self, api_url="http://192.168.1.50:8000"):
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

    def search_frame(self, frame, top_k=8, format="JPEG", quality=95):
        """Search for similar rotated images using a frame/numpy array"""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid numpy array")

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=quality)
            buffer.seek(0)

            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/search",
                params={"top_k": top_k},
                files={"file": ("frame.jpg", buffer, f"image/{format.lower()}")}
            )
            total_time = time.time() - start_time

            if response.status_code == 200:
                results = response.json()
                results["total_time"] = total_time
                return results
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        else:
            raise ValueError("Frame must be a 3-channel color image (BGR or RGB)")



def search_product_from_frame(frame=None, resize_dim=(256, 256), api_url="http://localhost:8000", top_k=8):
    """Search for a product using an image frame and rank results by occurrence."""
    search_client = ImageSearchFrameClient(api_url=api_url)

    # Check API health
    health = search_client.health_check()
    print(f"API Health: {health}")

    if frame is None:
        raise ValueError("No frame provided")

    # Resize the frame if dimensions are provided
    if resize_dim:
        frame = cv2.resize(frame, resize_dim)

    try:
        results = search_client.search_frame(frame, top_k=top_k)

        if results and results.get("results"):
            print(f"\nFound {len(results['results'])} similar images in {results['query_time']:.3f}s:")

            product_matches = []
            product_occurrences = []

            for i, item in enumerate(results["results"]):
                product_name = item.get("product", "Unknown")
                barcode = item.get("barcode", "Unknown")
                rotation = item.get("rotation", 0)
                flavor = item.get("flavor", "Unknown")
                path = item.get("path", "Unknown")

                match = {
                    "product_name": product_name,
                    "barcode": barcode,
                    "rotation": rotation,
                    "flavor": flavor,
                    "path": path,
                    "similarity": item['similarity']
                }

                product_matches.append(match)
                product_occurrences.append((product_name, flavor, barcode))  # Track by name and barcode combo

                print(f"{i + 1}. Product Name: {product_name}")
                print(f"   Barcode: {barcode}")
                print(f"   Rotation: {rotation}")
                print(f"   Flavor: {flavor}")
                print(f"   Similarity: {item['similarity']:.4f}")
                print(f"   Path: {path}")

            # Rank products by occurrence count
            product_counter = Counter(product_occurrences)
            ranked_products = product_counter.most_common()

            print("\n--- Ranked Products by Occurrence ---")
            for i, ((product_name, flavor, barcode), count) in enumerate(ranked_products):
                print(f"{i + 1}. {product_name} - {flavor} (Barcode: {barcode}) - {count} occurrences")

            top_product = ranked_products[0][0] if ranked_products else None
            return top_product, ranked_products, product_matches
        else:
            print("No matching products found")
            return None, [], []

    except Exception as e:
        print(f"Error searching image: {e}")
        return None, [], []

# Example usage
if __name__ == "__main__":
    # Call the function with an image path
    frame = cv2.imread(r"C:\Users\Linoflaptech\Desktop\ImageSearchAPI\captured_images\frame_1745815067.jpg")
    if frame is None:
        print("Error loading image")
        exit(1)

    # top_product, all_matches, ranked_products = search_product_from_frame(
    #     frame=frame,
    #     resize_dim=(256, 256),
    #     api_url="http://192.168.3.8:8000",
    #     top_k=8
    # )
    ranked_products = search_product_from_frame(
        frame=frame,
        resize_dim=(256, 256),
        api_url="http://192.168.1.50:8000",
        top_k= 8
    )

    # if top_product:
    #     product_name, barcode = top_product
    #     print(f"\nBest match: {product_name}")
    #     print(f"Barcode: {barcode}")
    #
    #     # Calculate confidence based on occurrence percentage
    #     total_matches = len(all_matches)
    #     top_occurrences = ranked_products[0][1]
    #     confidence = (top_occurrences / total_matches) * 100
    #
    #     print(f"Confidence: {confidence:.1f}% ({top_occurrences}/{total_matches} matches)")
    # else:
    #     print("No product identified")
