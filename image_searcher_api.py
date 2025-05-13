from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import torch
import numpy as np
import faiss
import pickle
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F
import cv2
import io
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import time
import os

HOST="192.168.3.8"

class ImageMetadata(BaseModel):
    path: str
    product: str
    rotation: str

class SearchResult(BaseModel):
    path: str
    product: str
    rotation: str
    similarity: float
    barcode: Optional[str]
    flavor: Optional[str]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time: float

class ImageSearcher:
    def __init__(self, index_path="./models/siglip2b-16-256-flavor.index",
                 meta_path="./models/siglip2b-16-256-flavor.pkl",
                 model_name="google/siglip2-base-patch16-256",
                 device=None):
        self.index_path = index_path
        self.meta_path = meta_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model on device: {self.device}")
        try:
            # Load the SigLIP model and processor
            print(f"Loading model: {model_name}")
            self.processor = SiglipProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = SiglipModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

            # Load the FAISS index and metadata
            print(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(self.index_path)

            print(f"Loading metadata from {meta_path}")
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            print(f"Loaded metadata with {len(self.metadata)} entries")
            if len(self.metadata) > 0:
                print(f"Sample metadata entry: {self.metadata[0]}")
        except Exception as e:
            print(f"Detailed initialization error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def compute_query_embedding_from_frame(self, frame):
        # Process the image directly from cv2 frame
        frame = cv2.resize(frame, (256, 256))  # resize depending on the model
        # OpenCV uses BGR, but most models expect RGB, so we need to convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame directly with the processor
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        
        # Extract image features (embedding) from the model
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        
        # Normalize the embedding (to unit length) for cosine similarity
        return F.normalize(image_embeds.squeeze(), p=2, dim=-1).cpu().numpy().astype("float32")
    
    def search(self, frame, top_k=3):
        # Compute the query embedding from frame
        query_embedding = self.compute_query_embedding_from_frame(frame)
        
        # Perform the search in the FAISS index
        D, I = self.index.search(np.array([query_embedding]), k=top_k)
        
        # Collect the search results
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.metadata):
                meta_entry = self.metadata[idx]
                # Add similarity score to the result
                result = {
                    "path": meta_entry["path"],
                    "product": meta_entry["product"],
                    "rotation": meta_entry["rotation"],
                    "similarity": float(dist),
                    "barcode": meta_entry.get("barcode"),  # add this line
                    "flavor": meta_entry.get("flavor")  # add this line
                }
                results.append(result)


            else:
                print(f"Warning: Index {idx} out of bounds for metadata of length {len(self.metadata)}")

        #debugging frame processed
        # Create the folder if it doesn't exist
        os.makedirs("best_images", exist_ok=True)

        # Save the frame with a timestamp
        timestamp = int(time.time())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"best_images/frame_{timestamp}.jpg", rgb_frame)
        return results



# Global variable to store our searcher instance
searcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the image searcher when the application starts"""
    global searcher
    try:
        searcher = ImageSearcher()
        print("Image searcher initialized successfully")
    except Exception as e:
        print(f"Error initializing image searcher: {e}")
        raise
    yield
    # Optionally add shutdown code here
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(title="SigLIP Rotated Image Search API", lifespan=lifespan)


@app.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {"message": "SigLIP Rotated Image Search API", "status": "running"}

@app.post("/search", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...), top_k: int = 3):
    """
    Upload an image and find similar rotated images from the index
    
    Parameters:
    - file: The image file to search for similar images
    - top_k: Number of similar images to return (default: 5)

    Returns:
    - List of similar images with their paths, products, rotation angles, and similarity scores
    """
    global searcher
    
    if searcher is None:
        raise HTTPException(status_code=500, detail="Image searcher not initialized")
    
    try:
        # Read image file
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform the search and time it
        start_time = time.time()
        results = searcher.search(img, top_k=top_k)
        query_time = time.time() - start_time
        
        return SearchResponse(results=results, query_time=query_time)
    
    except Exception as e:
        import traceback
        error_detail = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {
        "status": "healthy", 
        "device": searcher.device if searcher else "not initialized",
        "metadata_count": len(searcher.metadata) if searcher else 0
    }

def main():
    """Run the FastAPI application with Uvicorn"""

    uvicorn.run("image_searcher_api:app", host=HOST, port=8000, reload=False)

if __name__ == "__main__":
    main()