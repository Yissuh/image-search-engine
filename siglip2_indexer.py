import os
from PIL import Image
import torch
import numpy as np
import faiss
import pickle
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F  # For normalization

class ImageIndexer:
    def __init__(self, dataset_dir, index_path="siglip2b-16-256.index", meta_path="siglip2b-16-256-metadata.pkl", device=None):
        self.dataset_dir = dataset_dir
        self.index_path = index_path
        self.meta_path = meta_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load SigLIP model and processor (you can switch this to CLIP if preferred)
        self.processor = SiglipProcessor.from_pretrained("google/siglip2-base-patch16-256")
        self.model = SiglipModel.from_pretrained("google/siglip2-base-patch16-256").to(self.device)

        self.embeddings = []
        self.metadata = []

    def get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.dataset_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, fname)
                    image_paths.append(full_path)
        return image_paths

    def compute_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds.squeeze().cpu().numpy()

    def index_images(self):
        image_paths = self.get_image_paths()

        print(f"Found {len(image_paths)} images. Extracting embeddings...")

        for path in image_paths:
            embedding = self.compute_embedding(path)
            # Normalize embeddings to unit length (cosine similarity)
            embedding = F.normalize(torch.tensor(embedding), p=2, dim=-1).numpy()
            self.embeddings.append(embedding)
            self.metadata.append(path)  # or (path, class_name)

        embeddings_np = np.array(self.embeddings).astype("float32")
        # Use IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)

        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Indexed {len(self.metadata)} images into {self.index_path}.")

# ---- Run the indexer ----
if __name__ == "__main__":
    indexer = ImageIndexer(dataset_dir="./products")  # Change to your dataset directory
    indexer.index_images()
