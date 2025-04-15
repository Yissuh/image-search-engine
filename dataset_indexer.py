import os
from PIL import Image
import torch
import numpy as np
import faiss
import pickle
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F  # For normalization
import time

class RotatedImageIndexer:
    def __init__(self, dataset_dir, index_path="siglip2b-16-256-rotated.index", 
                 meta_path="siglip2b-16-256-rotated-metadata.pkl", device=None):
        self.dataset_dir = dataset_dir
        self.index_path = index_path
        self.meta_path = meta_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load SigLIP model and processor
        print(f"Loading SigLIP model on {self.device}...")
        self.processor = SiglipProcessor.from_pretrained("google/siglip2-base-patch16-256")
        self.model = SiglipModel.from_pretrained("google/siglip2-base-patch16-256").to(self.device)

        self.embeddings = []
        self.metadata = []

    def get_rotated_image_paths(self):
        """Get paths to all rotated lightglue images in the dataset"""
        image_paths = []
        product_folders = [os.path.join(self.dataset_dir, folder) for folder in os.listdir(self.dataset_dir) 
                          if os.path.isdir(os.path.join(self.dataset_dir, folder))]
        
        for folder in product_folders:
            # Get product name from folder path
            product_name = os.path.basename(folder)
            
            # Find all rotated images (0, 45, 90, 135, 180, 225, 270, 315)
            for file in os.listdir(folder):
                if "lightglue_" in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(folder, file)
                    # Store both path and product name
                    image_paths.append((full_path, product_name))
        
        return image_paths

    def compute_embedding(self, image_path):
        """Compute embedding for an image while handling transparency"""
        try:
            # Open image and convert to RGB (handling transparency)
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embeds = self.model.get_image_features(**inputs)
            return image_embeds.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def index_images(self):
        """Index all rotated images in the dataset"""
        rotated_image_paths = self.get_rotated_image_paths()

        print(f"Found {len(rotated_image_paths)} rotated images. Extracting embeddings...")

        for path, product_name in rotated_image_paths:
            # Extract rotation angle from filename
            filename = os.path.basename(path)
            rotation = "unknown"
            if "lightglue_" in filename:
                parts = filename.split("lightglue_")
                if len(parts) > 1:
                    angle_part = parts[1].split(".")[0].split("_")[0]
                    if angle_part.isdigit():
                        rotation = f"{angle_part}°"
            
            # Compute embedding
            embedding = self.compute_embedding(path)
            if embedding is not None:
                # Normalize embeddings to unit length (cosine similarity)
                embedding = F.normalize(torch.tensor(embedding), p=2, dim=-1).numpy()
                self.embeddings.append(embedding)
                
                # Store metadata (path, product name, rotation angle)
                self.metadata.append({
                    "path": path,
                    "product": product_name,
                    "rotation": rotation
                })
                
                # Print progress
                if len(self.embeddings) % 50 == 0:
                    print(f"Processed {len(self.embeddings)} images...")

        # Only proceed if we have embeddings
        if not self.embeddings:
            print("No valid embeddings were computed. Check your image paths.")
            return
            
        # Convert embeddings to numpy array
        embeddings_np = np.array(self.embeddings).astype("float32")
        
        # Use IndexFlatIP for cosine similarity
        print(f"Creating FAISS index with dimension {embeddings_np.shape[1]}...")
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)

        # Save index and metadata
        print(f"Saving index to {self.index_path}...")
        faiss.write_index(index, self.index_path)
        
        print(f"Saving metadata to {self.meta_path}...")
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Successfully indexed {len(self.metadata)} rotated images.")

# # ---- Run the indexer ----
# if __name__ == "__main__":
#     start_time = time.time()
#     indexer = RotatedImageIndexer(dataset_dir="./dataset")  # Update to your dataset directory
#     indexer.index_images()
#     print(f"Indexing completed in {time.time() - start_time:.2f} seconds.")