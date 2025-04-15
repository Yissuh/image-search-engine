# SigLIP-Based Image Similarity Search

This repository provides tools for indexing and searching product images using Google's SigLIP (Sigmoid Loss for Language Image Pre-Training) model embeddings. The system creates a searchable FAISS index of rotated product images and enables fast similarity searches.

## Components

### 1. `dataset_indexer.py`

Indexes rotated product images from a directory structure by:
- Loading images from product folders
- Computing embeddings using SigLIP
- Creating a FAISS index for fast similarity searches
- Storing metadata (file paths, product names, rotation angles)

### 2. `image_searcher.py`

Provides a reusable class for querying the index:
- Loads pre-trained SigLIP model and FAISS index
- Processes input images to generate compatible embeddings
- Performs similarity searches to find matching products

### 3. `frame_searcher.py`

Demo script showing how to use the system:
- Loads an example product image
- Searches for similar products in the index
- Displays top matches with similarity scores

## Usage

### Indexing a Dataset

```python
from dataset_indexer import RotatedImageIndexer

indexer = RotatedImageIndexer(dataset_dir="./dataset")
indexer.index_images()
```

### Searching for Similar Images

```python
from image_searcher import ImageSearcher
import cv2

# Initialize searcher with paths to your index files
searcher = ImageSearcher(
    index_path="models/siglip2b-16-256-rotated.index",
    meta_path="models/siglip2b-16-256-rotated-metadata.pkl"
)

# Load an image and search
frame = cv2.imread("query_image.jpg")
results = searcher.search(frame, top_k=5)

# Display results
for i, (metadata, score) in enumerate(results, 1):
    print(f"{i}. Product: {metadata['product']}, Rotation: {metadata['rotation']}, Score: {score}")
```

## Requirements

- PyTorch
- FAISS
- Transformers (Hugging Face)
- OpenCV
- Pillow

## Notes

- The system uses Google's SigLIP2-base-patch16-256 model for high-quality image embeddings
- Cosine similarity is used for comparing embeddings (via FAISS IndexFlatIP)
- The indexer handles images at various rotation angles to improve matching robustness