import torch
import numpy as np
import faiss
import pickle
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F
import cv2

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
    
    def compute_query_embedding_from_frame(self, frame):
        # Process the image directly from cv2 frame
        # OpenCV uses BGR, but most models expect RGB, so we need to convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame directly with the processor
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        
        # Extract image features (embedding) from the model
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        
        # Normalize the embedding (to unit length) for cosine similarity
        return F.normalize(image_embeds.squeeze(), p=2, dim=-1).cpu().numpy().astype("float32")
    

    def search(self, frame, top_k=5):
        # Compute the query embedding from frame
        query_embedding = self.compute_query_embedding_from_frame(frame)
        
        # Perform the search in the FAISS index
        D, I = self.index.search(np.array([query_embedding]), k=top_k)
        
        # Collect the search results
        results = []
        for dist, idx in zip(D[0], I[0]):
            result_path = self.metadata[idx]
            results.append((result_path, dist))
        
        return results