# Product Image Search Application

An application for capturing product images, removing backgrounds, generating rotated variants, and indexing them for fast similarity search using Google's SigLIP model and FAISS.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [1. Launching the GUI](#1-launching-the-gui)
  - [2. Product Image Capture](#2-product-image-capture)
  - [3. Dataset Indexing](#3-dataset-indexing)
- [Workflow Overview](#workflow-overview)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Credits](#credits)

---

## Features
- **Product Capture**: Take product photos via webcam, automatically remove backgrounds, and generate rotated versions (every 45°).
- **Dataset Indexer**: Indexes all captured images using the SigLIP model, creating a searchable FAISS index for similarity search.
- **Modern GUI**: Easy-to-use PyQt6 interface for product capture and dataset management.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your_repo_url>
   cd imagesearch
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have Python 3.8 or later.

---

## Getting Started

### 1. Launching the GUI
Run the main application:
```bash
python gui_app.py
```
This opens the Product Image Search System GUI.

### 2. Product Image Capture
- Go to the **Product Capture** tab.
- Enter the product name, flavor, and barcode.
- Select the camera (usually 0 for default webcam).
- Click **Test Camera** to check the feed.
- Click **Capture & Process**:
  - Captures the image from the camera.
  - Removes the background.
  - Saves rotated versions (0°, 45°, ..., 315°) in `dataset/ProductName_Flavor_Barcode/`.

### 3. Dataset Indexing
- Switch to the **Dataset Indexer** tab.
- Browse and select your `dataset` directory.
- Click **Run Indexer**:
  - All images are processed with the SigLIP model.
  - A FAISS index and metadata file are created for fast similarity search.

---

## Workflow Overview
1. **Capture Product Images:**
   - Use the GUI to capture and process product images with background removal and rotation.
2. **Index the Dataset:**
   - Use the GUI or run `dataset_indexer.py` directly to build the search index:
     ```bash
     python dataset_indexer.py
     ```
   - This creates index files (e.g., `siglip2b-16-256-flavor.index`, `siglip2b-16-256-flavor.pkl`).
3. **Search for Images:**
   - Use the provided search scripts (e.g., `image_searcher.py`) to find similar products by image.

---

## Troubleshooting
- **Camera Not Detected:**
  - Ensure your webcam is connected and not used by another application.
  - Try changing the camera index (0, 1, 2).
- **Missing Dependencies:**
  - Double-check your Python environment and install all packages from `requirements.txt`.
- **CUDA/CPU Issues:**
  - The application will use GPU if available; otherwise, it defaults to CPU.
- **Background Removal Fails:**
  - Ensure `rembg` and its dependencies are installed.

---

## Requirements
- Python 3.8+
- PyQt6
- OpenCV
- Pillow
- rembg
- torch
- transformers
- faiss-cpu
- numpy
- onnxruntime (only if using ONNX models)

Install all requirements with:
```bash
pip install -r requirements.txt
```

---

## Credits
- [Google SigLIP](https://huggingface.co/google/siglip2-base-patch16-256)
- [FAISS](https://github.com/facebookresearch/faiss)
- [rembg](https://github.com/danielgatis/rembg)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)

---

For more details, see the source files: `gui_app.py`, `dataset_indexer.py`, and `requirements.txt`.
