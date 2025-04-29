import torch
import numpy as np
import faiss
import pickle
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F  # For normalization

class ImageSearcher:
    def __init__(self, index_path="siglip2b-16-256.index", meta_path="siglip2b-16-256-metadata.pkl", model_name="google/siglip2-base-patch16-256", device=None):
        self.index_path = index_path
        self.meta_path = meta_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the SigLIP model and processor
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)

        # Load the FAISS index and metadata (filenames)
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def compute_query_embedding(self, image_path):
        # Open the image and process it
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Extract image features (embedding) from the model
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        
        # Normalize the embedding (to unit length) for cosine similarity
        return F.normalize(image_embeds.squeeze(), p=2, dim=-1).cpu().numpy().astype("float32")

    def search(self, query_image_path, top_k=5):
        # Compute the query embedding
        query_embedding = self.compute_query_embedding(query_image_path)

        # Perform the search in the FAISS index
        D, I = self.index.search(np.array([query_embedding]), k=top_k)

        # Collect the search results
        results = []
        for dist, idx in zip(D[0], I[0]):
            result_path = self.metadata[idx]
            results.append((result_path, dist))

        return results

# ---- Example Run ----
if __name__ == "__main__":
    # Initialize the searcher
    searcher = ImageSearcher(index_path="siglip2b-16-256-rotated.index", meta_path="siglip2b-16-256-rotated-metadata.pkl")
    
    # Provide the query image path
    query_path = "images/choco_knots_cart.jpg"  # Replace with your actual query image path

    # Perform the search and retrieve the top K results
    results = searcher.search(query_path, top_k=5)

    # Print the results
    print("\nTop Matches:")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. {path} (Score: {score})")
